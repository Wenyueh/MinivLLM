"""Paged-attention decode benchmark.

Compares four implementations on identical inputs:

  Naive PyTorch   gathers K/V block by block in a Python loop, then dense attention
  Fast PyTorch    vectorised gather + batched matmul
  Triton (old)    one scalar iteration per token inside each chunk
  Triton (new)    whole chunk loaded as a single masked vector op

Every implementation is checked against a float32 reference before it is timed, so
a kernel that is fast because it reads the wrong memory shows up as NO in the
correct column rather than as a speedup.
"""

import argparse
import torch
import triton
import triton.language as tl


# =============================================================================
# Triton (old): scalar per-token loop inside each chunk
# =============================================================================

@triton.jit
def paged_attention_decode_kernel_old(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Paged attention decode kernel. Processes KV cache in chunks.

    Each chunk is walked one token at a time, so the block table is looked up per
    token and the within-block offset is token_idx % block_size.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N

        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10

            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                        if physical_block_idx != -1:
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                        block_offset * num_kv_heads * head_dim +
                                        kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)

                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            qk = tl.where(mask_n, qk, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            acc = acc * alpha
            l_i = l_i * alpha

            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                        if physical_block_idx != -1:
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                        block_offset * num_kv_heads * head_dim +
                                        kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset)

                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))

                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


# =============================================================================
# Triton (new): whole chunk loaded as one vector op
# =============================================================================

@triton.jit
def paged_attention_decode_kernel_new(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Paged attention decode kernel with the whole chunk loaded at once.

    The block table is gathered per token rather than per chunk, so a chunk may
    straddle any number of blocks and no relation between BLOCK_N and block_size
    is assumed. Each lane resolves its own token independently:

        token t  ->  block_tables[batch, t // block_size], slot t % block_size

    K/V for the whole chunk then come from one masked gather each.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N

        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            logical_block = offs_n // block_size
            # physical_block * block_size already moves the base pointer to the
            # start of the block, so what is added is the offset *within* the
            # block, not the global token index.
            offs_in_block = offs_n % block_size

            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

            # One block-table entry per token: a chunk is free to span several
            # blocks, and those blocks need not be adjacent in the cache.
            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                mask=in_range, other=-1)
            valid = in_range & (physical_block != -1)
            # Masked-out lanes still take part in the address arithmetic, so give
            # them block 0 to keep every computed offset inside the cache.
            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

            kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
                         + offs_in_block[None, :] * (num_kv_heads * head_dim)
                         + kv_head_idx * head_dim
                         + offs_d[:, None])

            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            k = tl.cast(k, tl.float32)
            score = tl.sum(q[:, None] * k, axis=0) * scale
            qk = tl.where(valid, score, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            acc = acc * alpha
            l_i = l_i * alpha

            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            v = tl.cast(v, tl.float32)
            weight = tl.where(valid, p, 0.0)
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
            l_i = l_i + tl.sum(weight)

            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def _make_triton_launcher(kernel):
    """Wrap a decode kernel behind a common signature."""
    def launch(
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        max_num_blocks = block_tables.shape[1]
        query = query.contiguous()
        output = torch.empty_like(query)

        BLOCK_N = 64 if head_dim <= 128 else 32
        grid = (batch_size, num_heads)

        kernel[grid](
            output, query, k_cache, v_cache, block_tables, context_lens,
            scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, block_size=block_size,
            max_num_blocks=max_num_blocks, BLOCK_N=BLOCK_N,
        )
        return output
    return launch


paged_attention_decode_triton_old = _make_triton_launcher(paged_attention_decode_kernel_old)
paged_attention_decode_triton_new = _make_triton_launcher(paged_attention_decode_kernel_new)


# =============================================================================
# PyTorch implementations
# =============================================================================

def decode_torch_optimized(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype

    max_context_len = context_lens.max().item()

    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    for i in range(batch_size):
        seq_len = context_lens[i].item()
        # Gather through the block table: logical token t lives at
        # (block_tables[i, t // block_size], t % block_size).
        idx = torch.arange(seq_len, device=device)
        phys = block_tables[i][idx // block_size].long()
        off = idx % block_size
        valid = phys != -1

        padded_k[i, :seq_len][valid] = k_cache[phys[valid], off[valid]]
        padded_v[i, :seq_len][valid] = v_cache[phys[valid], off[valid]]

    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)

    q = q.unsqueeze(2)
    padded_k = padded_k.transpose(1, 2)
    padded_v = padded_v.transpose(1, 2)

    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale

    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)

    return output


def naive_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """
    Naive decode implementation
    This reconstructs full K, V sequences and uses standard PyTorch attention.
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype

    max_context_len = context_lens.max().item()

    # Gather K, V into full sequences (inefficient for large contexts)
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)

    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        for block_idx in range(num_blocks_needed):
            block_id = block_tables[i, block_idx].item()
            if block_id == -1:
                break
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            padded_k[i, start:end] = k_cache[block_id, :end - start]
            padded_v[i, start:end] = v_cache[block_id, :end - start]

    # GQA
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)

    # Reshape and compute attention
    q = q.unsqueeze(2)  # (B, H, 1, D)
    padded_k = padded_k.transpose(1, 2)  # (B, H, N, D)
    padded_v = padded_v.transpose(1, 2)  # (B, H, N, D)

    # This is the inefficient part - materializes full attention matrix
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale

    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)

    return output


IMPLS = {
    "Naive PyTorch": naive_decode_attention,
    "Fast PyTorch": decode_torch_optimized,
    "Triton (old)": paged_attention_decode_triton_old,
    "Triton (new)": paged_attention_decode_triton_new,
}


# =============================================================================
# Test data, reference, timing
# =============================================================================

def setup_test_data(batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size, device='cuda'):
    """Setup test data for benchmarking"""
    # Query: (batch_size, num_heads, head_dim)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)

    # Calculate number of blocks needed
    max_num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks

    # KV Cache: (total_blocks, block_size, num_kv_heads, head_dim)
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    # Block tables: (batch_size, max_num_blocks).
    # Shuffled, not arange: a real BlockManager hands out whatever blocks are free,
    # so logical order never matches physical order. An identity mapping would let
    # a kernel that ignores the block table still produce the right answer.
    perm = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    block_tables = perm.reshape(batch_size, max_num_blocks)

    # Context lengths: (batch_size,)
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)

    # Scale
    scale = 1.0 / (head_dim ** 0.5)

    return q, k_cache, v_cache, block_tables, context_lens, scale


def reference_output(q, k_cache, v_cache, block_tables, context_lens,
                     scale, num_heads, num_kv_heads, head_dim, block_size):
    """Float32 attention over the logical sequence gathered from the paged cache."""
    batch_size = q.shape[0]
    out = torch.empty(batch_size, num_heads, head_dim, device=q.device, dtype=torch.float32)
    for b in range(batch_size):
        ctx = int(context_lens[b].item())
        idx = torch.arange(ctx, device=q.device)
        phys = block_tables[b][idx // block_size].long()
        off = idx % block_size
        K = k_cache[phys, off].float()   # (ctx, num_kv_heads, head_dim)
        V = v_cache[phys, off].float()
        for h in range(num_heads):
            kvh = h // (num_heads // num_kv_heads)
            s = (q[b, h].float() @ K[:, kvh].T) * scale
            out[b, h] = torch.softmax(s, dim=-1) @ V[:, kvh]
    return out


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark(batch_size, seq_len, num_heads=16, num_kv_heads=8,
              head_dim=128, block_size=256, num_iterations=100, warmup=10, tol=2e-2):
    """Check every implementation against a float32 reference, then time it."""

    print(f"\n{'='*74}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'='*74}")

    q, k_cache, v_cache, block_tables, context_lens, scale = setup_test_data(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size
    )
    ref = reference_output(q, k_cache, v_cache, block_tables, context_lens,
                           scale, num_heads, num_kv_heads, head_dim, block_size)

    header = f"{'implementation':<16} {'max_err':>10} {'correct':>8} {'ms':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    results = {}
    baseline_ms = None

    for name, impl in IMPLS.items():
        call = lambda impl=impl: impl(q, k_cache, v_cache, block_tables, context_lens,
                                      scale, num_heads, num_kv_heads, head_dim, block_size)
        try:
            out = call()
            err = (out.float() - ref).abs().max().item()
        except RuntimeError as e:
            # An out-of-bounds read poisons the CUDA context: every later launch in
            # this process would fail too, so report and stop rather than emit noise.
            print(f"{name:<16} {'CRASH':>10} {'no':>8}")
            print(f"\n{name} raised: {type(e).__name__}: {str(e).splitlines()[0]}")
            print("The CUDA context cannot be recovered in-process; "
                  "rerun with a smaller config to measure the remaining cases.")
            raise SystemExit(1)

        correct = err < tol
        ms = time_ms(call, warmup=warmup, iters=num_iterations)
        if baseline_ms is None:
            baseline_ms = ms
        results[name] = ms
        print(f"{name:<16} {err:>10.5f} {'yes' if correct else 'NO':>8} "
              f"{ms:>9.3f} {baseline_ms / ms:>7.2f}x")

    print(f"\nmax_err is vs a float32 reference; speedup is vs {next(iter(IMPLS))}.")
    print("A 'NO' in the correct column means the timing is meaningless -- that")
    print("implementation is not computing attention over the right tokens.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Paged attention decode benchmark.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    # 256 is what main.py / models/llama.py actually configure. Note that the
    # Attention class itself still defaults to 16.
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("PAGED ATTENTION DECODE BENCHMARK")
    print("Comparing: " + " | ".join(IMPLS))
    print("=" * 74)

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            benchmark(batch_size=batch_size, seq_len=seq_len,
                      num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
                      head_dim=args.head_dim, block_size=args.block_size,
                      num_iterations=args.iters, warmup=args.warmup)


if __name__ == "__main__":
    main()
