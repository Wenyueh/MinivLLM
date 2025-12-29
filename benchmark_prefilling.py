import torch
import time
import triton 
import triton.language as tl

# ============================================================================
# 1. PyTorch Standard (O(N²) memory)
# ============================================================================
def pytorch_standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Standard PyTorch attention - O(N²) memory"""
    total_tokens = q.shape[0]
    output = torch.zeros_like(q)
    
    cu_seqlens_cpu = cu_seqlens.cpu().tolist()
    
    for i in range(len(cu_seqlens_cpu) - 1):
        start = cu_seqlens_cpu[i]
        end = cu_seqlens_cpu[i + 1]
        seq_len = end - start
        
        q_seq = q[start:end].transpose(0, 1)  # (num_heads, seq_len, head_dim)
        k_seq = k[start:end].transpose(0, 1)
        v_seq = v[start:end].transpose(0, 1)
        
        # GQA
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(num_groups, dim=0)
            v_seq = v_seq.repeat_interleave(num_groups, dim=0)
        
        # O(N²) attention matrix
        attn_scores = torch.matmul(q_seq, k_seq.transpose(1, 2)) * scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out_seq = torch.matmul(attn_probs, v_seq).transpose(0, 1)
        
        output[start:end] = out_seq
    
    return output


# ============================================================================
# 2. Naive Triton (O(N²) memory, limited to short sequences)
# ============================================================================
@triton.jit  
def naive_triton_attention_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Naive Triton: materializes full O(N²) attention matrix.
    Memory limited: BLOCK_SIZE^2 * 4 bytes < ~48KB
    For BLOCK_SIZE=64: 64*64*4 = 16KB ✓
    For BLOCK_SIZE=128: 128*128*4 = 64KB ✗ (exceeds limit)
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    if seq_len > BLOCK_SIZE:
        return  # Skip sequences that are too long
    
    # Load entire sequence
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, head_dim)
    mask = offs_m < seq_len
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + head_idx * head_dim + offs_d[None, :]
    k_ptrs = K + (seq_start + offs_m[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
    v_ptrs = V + (seq_start + offs_m[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
    
    q = tl.load(q_ptrs, mask=mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)
    
    # Compute full attention matrix - O(N²) memory!
    qk = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_SIZE, BLOCK_SIZE)
    
    # Apply causal mask
    causal_mask = offs_m[:, None] >= offs_m[None, :]
    seq_mask = mask[:, None] & mask[None, :]
    qk = tl.where(causal_mask & seq_mask, qk, float('-inf'))
    
    # Softmax
    qk_max = tl.max(qk, axis=1)
    qk_exp = tl.exp(qk - qk_max[:, None])
    qk_sum = tl.sum(tl.where(seq_mask, qk_exp, 0.0), axis=1)
    attn = qk_exp / qk_sum[:, None]
    
    # Output
    out = tl.dot(attn.to(v.dtype), v)
    
    # Store
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + head_idx * head_dim + offs_d[None, :]
    tl.store(o_ptrs, out, mask=mask[:, None])


def naive_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Naive Triton - limited by shared memory for attention matrix"""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    output = torch.empty_like(q)
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Determine BLOCK_SIZE based on shared memory limits
    # Attention matrix uses BLOCK_SIZE^2 * 4 bytes
    # Target: < 48KB for safety
    # BLOCK_SIZE = 64 -> 16KB ✓
    # BLOCK_SIZE = 128 -> 64KB ✗
    
    if head_dim <= 64:
        BLOCK_SIZE = 128  # Risky but might work
    else:
        BLOCK_SIZE = 64   # Safe choice
    
    # Round up max_seq_len to power of 2, but cap at BLOCK_SIZE
    actual_size = 2 ** ((max_seq_len - 1).bit_length())
    actual_size = min(actual_size, BLOCK_SIZE)
    
    if max_seq_len > BLOCK_SIZE:
        print(f"      WARNING: seq_len ({max_seq_len}) > BLOCK_SIZE ({BLOCK_SIZE}), results may be incorrect")
    
    grid = (num_seqs, num_heads)
    
    naive_triton_attention_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_SIZE=actual_size,
    )
    
    return output


# ============================================================================
# 3. Flash Attention (O(N) memory)
# ============================================================================
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash Attention - O(N) memory via online softmax"""
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    if start_m * BLOCK_M >= seq_len:
        return
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Online softmax - stores only O(BLOCK_M) values
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        qk = tl.dot(q, k) * scale
        
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    acc = acc / l_i[:, None]
    
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Flash Attention - online softmax optimization"""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    output = torch.empty_like(q)
    
    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 64, 64
    elif head_dim <= 128:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 16
    
    num_seqs = cu_seqlens.shape[0] - 1
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output


def find_crossover_point():
    """Find where flash attention becomes faster than naive"""
    
    print("\n" + "="*80)
    print("FINDING CROSSOVER POINT: When does Flash beat Naive?")
    print("="*80)
    
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_seqs = 2
    
    results = []
    
    # Test different sequence lengths
    seq_lengths = [16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512, 1024]
    
    for seq_len in seq_lengths:
        print(f"\nTesting seq_len = {seq_len}...")
        
        q, k, v, cu_seqlens, scale = setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim)
        
        # Naive Triton (if it fits)
        max_safe_seq = 64 if head_dim > 64 else 128
        if seq_len <= max_safe_seq:
            for _ in range(10):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            torch.cuda.synchronize()
            naive_time = (time.perf_counter() - start) / 50
        else:
            naive_time = None
        
        # Flash Attention
        for _ in range(10):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 50
        
        if naive_time:
            speedup = naive_time / flash_time
            winner = "Naive" if speedup < 1 else "Flash"
            print(f"  Naive: {naive_time*1000:.3f}ms | Flash: {flash_time*1000:.3f}ms | Winner: {winner} ({abs(speedup):.2f}x)")
            results.append((seq_len, naive_time, flash_time, winner))
        else:
            print(f"  Naive: SKIPPED | Flash: {flash_time*1000:.3f}ms | Winner: Flash (by default)")
            results.append((seq_len, None, flash_time, "Flash"))
    
    # Summary
    print("\n" + "="*80)
    print("CROSSOVER ANALYSIS")
    print("="*80)
    print(f"{'Seq Len':>10} | {'Naive (ms)':>12} | {'Flash (ms)':>12} | {'Winner':>10} | {'Speedup':>10}")
    print("-" * 80)
    
    crossover = None
    for seq_len, naive_time, flash_time, winner in results:
        if naive_time:
            speedup = naive_time / flash_time if flash_time < naive_time else flash_time / naive_time
            speedup_str = f"{speedup:.2f}x"
            naive_str = f"{naive_time*1000:.3f}"
        else:
            speedup_str = "N/A"
            naive_str = "OOM"
        
        flash_str = f"{flash_time*1000:.3f}"
        print(f"{seq_len:>10} | {naive_str:>12} | {flash_str:>12} | {winner:>10} | {speedup_str:>10}")
        
        # Find crossover point
        if crossover is None and winner == "Flash" and naive_time is not None:
            crossover = seq_len
    

def analyze_kernel_launches():
    """Show why naive has fewer kernel launches"""
    
    print("\n" + "="*80)
    print("KERNEL LAUNCH ANALYSIS")
    print("="*80)
    
    num_seqs = 2
    seq_len = 60
    num_heads = 32
    BLOCK_M = 32
    
    # Naive Triton
    naive_grid = (num_seqs, num_heads)
    naive_kernels = num_seqs * num_heads
    
    # Flash Attention  
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    flash_grid = (num_blocks_m, num_heads, num_seqs)
    flash_kernels = num_blocks_m * num_heads * num_seqs
    
    print(f"\nFor {num_seqs} sequences × {seq_len} tokens:")
    print(f"  Naive Triton grid:    {naive_grid}")
    print(f"  Naive total kernels:  {naive_kernels}")
    print(f"\n  Flash Attention grid: {flash_grid}")
    print(f"  Flash total kernels:  {flash_kernels}")
    print(f"\n  Ratio: Flash launches {flash_kernels/naive_kernels:.1f}x more kernels")
    print(f"\n  Each kernel launch has ~5-20μs overhead")
    print(f"  Extra overhead: ~{(flash_kernels - naive_kernels) * 10}μs = {(flash_kernels - naive_kernels) * 0.01:.2f}ms")


# ============================================================================
# Benchmark
# ============================================================================

def setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim):
    total_tokens = num_seqs * seq_len
    device = 'cuda'
    
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    
    cu_seqlens = torch.tensor([i * seq_len for i in range(num_seqs + 1)], 
                              device=device, dtype=torch.int32)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    return q, k, v, cu_seqlens, scale


def benchmark(num_seqs, seq_len, num_heads=32, num_kv_heads=8, head_dim=128, num_iter=50):
    print(f"\n{'='*80}")
    print(f"Benchmark: {num_seqs} seqs × {seq_len} tokens (total: {num_seqs*seq_len} tokens)")
    print(f"Heads: {num_heads}/{num_kv_heads}, Dim: {head_dim}")
    print(f"{'='*80}")
    
    q, k, v, cu_seqlens, scale = setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim)
    
    results = {}
    outputs = {}
    
    # 1. PyTorch
    print("\n[1/3] PyTorch Standard (O(N²) memory)...")
    for _ in range(5):
        _ = pytorch_standard_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        outputs['pytorch'] = pytorch_standard_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    torch.cuda.synchronize()
    t = (time.perf_counter() - start) / num_iter
    results['PyTorch (O(N²))'] = t
    print(f"      {t*1000:.3f} ms")
    
    # 2. Naive Triton (limited to seq_len ≤ 128 for head_dim=128)
    max_safe_seq = 64 if head_dim > 64 else 128
    if seq_len <= max_safe_seq:
        print(f"\n[2/3] Naive Triton (O(N²), materializes full attention)...")
        for _ in range(5):
            _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iter):
            outputs['naive'] = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
        torch.cuda.synchronize()
        t = (time.perf_counter() - start) / num_iter
        results['Naive Triton (O(N²))'] = t
        print(f"      {t*1000:.3f} ms")
    else:
        print(f"\n[2/3] Naive Triton: SKIPPED (seq_len={seq_len} > {max_safe_seq}, would exceed shared memory)")
    
    # 3. Flash
    print("\n[3/3] Flash Attention (O(N), online softmax)...")
    for _ in range(5):
        _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        outputs['flash'] = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    torch.cuda.synchronize()
    t = (time.perf_counter() - start) / num_iter
    results['Flash Attention (O(N))'] = t
    print(f"      {t*1000:.3f} ms")
    


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PREFILL ATTENTION BENCHMARK")
    print("Comparing: PyTorch (O(N²)) | Naive Triton (O(N²)) | Flash (O(N))")
    print("="*80)
    
    benchmark(num_seqs=2, seq_len=60, num_iter=100)
    benchmark(num_seqs=4, seq_len=64, num_iter=100)
    benchmark(num_seqs=2, seq_len=1024, num_iter=30)
    benchmark(num_seqs=1, seq_len=4096, num_iter=10)

    # Now analyze the crossover point
    find_crossover_point()
    
    # Explain why
    analyze_kernel_launches()
    