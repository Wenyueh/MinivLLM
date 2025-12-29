import torch
import time
import triton 
import triton.language as tl

@triton.jit
def paged_attention_decode_kernel(
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
    """Optimized paged attention kernel for decode phase."""
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


def paged_attention_decode_triton(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    query = query.contiguous()
    output = torch.empty_like(query)
    
    BLOCK_N = 64 if head_dim <= 128 else 32
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output, query, k_cache, v_cache, block_tables, context_lens,
        scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size, 
        max_num_blocks=max_num_blocks, BLOCK_N=BLOCK_N,
    )
    return output


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
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        valid_blocks = block_tables[i, :num_blocks_needed]
        valid_blocks = valid_blocks[valid_blocks != -1]
        
        if len(valid_blocks) > 0:
            gathered_k = k_cache[valid_blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
            gathered_v = v_cache[valid_blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
            
            padded_k[i, :seq_len] = gathered_k
            padded_v[i, :seq_len] = gathered_v
    
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
    all_k = []
    all_v = []
    
    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        seq_k_list = []
        seq_v_list = []
        for block_idx in range(num_blocks_needed):
            block_id = block_tables[i, block_idx].item()
            if block_id == -1:
                break
            seq_k_list.append(k_cache[block_id])
            seq_v_list.append(v_cache[block_id])
        
        if len(seq_k_list) > 0:
            seq_k = torch.cat(seq_k_list, dim=0)[:seq_len]
            seq_v = torch.cat(seq_v_list, dim=0)[:seq_len]
            all_k.append(seq_k)
            all_v.append(seq_v)
    
    # Pad sequences
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    
    for i, (k_seq, v_seq) in enumerate(zip(all_k, all_v)):
        seq_len = len(k_seq)
        padded_k[i, :seq_len] = k_seq
        padded_v[i, :seq_len] = v_seq
    
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
    
    # Block tables: (batch_size, max_num_blocks)
    block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch_size, max_num_blocks)
    
    # Context lengths: (batch_size,)
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    # Scale
    scale = 1.0 / (head_dim ** 0.5)
    
    return q, k_cache, v_cache, block_tables, context_lens, scale


def benchmark(batch_size, seq_len, num_heads=32, num_kv_heads=8, 
                                  head_dim=128, block_size=16, num_iterations=100):
    """Compare all three implementations"""
    
    print(f"\n{'='*70}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'='*70}")
    
    # Setup data
    q, k_cache, v_cache, block_tables, context_lens, scale = setup_test_data(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size
    )
    
    results = {}
    
    # 1. Naive implementation (your original?)
    print("\n1. Testing Naive PyTorch implementation...")
    for _ in range(10):  # warmup
        _ = naive_decode_attention(q, k_cache, v_cache, block_tables, context_lens,
                                   scale, num_heads, num_kv_heads, head_dim, block_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_naive = naive_decode_attention(q, k_cache, v_cache, block_tables, context_lens,
                                           scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / num_iterations
    results['Naive PyTorch'] = naive_time
    print(f"   Time: {naive_time*1000:.3f}ms")
    
    # 2. Optimized PyTorch
    print("\n2. Testing Optimized PyTorch implementation...")
    for _ in range(10):  # warmup
        _ = decode_torch_optimized(q, k_cache, v_cache, block_tables, context_lens,
                                   scale, num_heads, num_kv_heads, head_dim, block_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_pytorch = decode_torch_optimized(q, k_cache, v_cache, block_tables, context_lens,
                                            scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iterations
    results['Optimized PyTorch'] = pytorch_time
    print(f"   Time: {pytorch_time*1000:.3f}ms")
    
    # 3. Triton
    print("\n3. Testing Triton implementation...")
    for _ in range(10):  # warmup
        _ = paged_attention_decode_triton(q, k_cache, v_cache, block_tables, context_lens,
                                          scale, num_heads, num_kv_heads, head_dim, block_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_triton = paged_attention_decode_triton(q, k_cache, v_cache, block_tables, context_lens,
                                                   scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iterations
    results['Triton'] = triton_time
    print(f"   Time: {triton_time*1000:.3f}ms")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE PAGED ATTENTION DECODE BENCHMARK")
    print("Comparing: Naive PyTorch | Optimized PyTorch | Triton")
    print("="*70)
    
    benchmark(batch_size=2, seq_len=60, num_iterations=100)
    benchmark(batch_size=1, seq_len=512, num_iterations=100)
    benchmark(batch_size=16, seq_len=256, num_iterations=50)
    benchmark(batch_size=4, seq_len=2048, num_iterations=20)