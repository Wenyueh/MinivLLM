import triton 
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr, 
    value_ptr,
    k_cache_ptr, 
    v_cache_ptr,
    k_scale_ptr,  # 新增
    v_scale_ptr,  # 新增
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    USE_INT8: tl.constexpr  # 编译时常量
):
    """
    Store keys and values into paged KV cache.
    支持 FP16 存储和 INT8 动态量化存储。
    """
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    if slot_idx == -1:
        return
    
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    head_idx = tl.program_id(1)
    
    head_offsets = tl.arange(0, head_dim)
    input_offset = (token_idx * num_kv_heads * head_dim + 
                    head_idx * head_dim + 
                    head_offsets)

    # 加载原始 FP16/BF16 数据
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    # 计算 Cache 物理地址
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + 
                   block_offset * num_kv_heads * head_dim + 
                   head_idx * head_dim + 
                   head_offsets)
    
    if USE_INT8:
        # === INT8 量化逻辑 ===
        # 计算 Scale: max(abs(val))
        # 注意：Triton 的 max 支持向量归约
        k_abs_max = tl.max(tl.abs(key))
        v_abs_max = tl.max(tl.abs(value))
        
        # 防止除0，并计算 scale
        # 标准公式: val_int8 = round(val / scale)
        # scale = max_val / 127.0
        k_scale = k_abs_max / 127.0
        v_scale = v_abs_max / 127.0
        
        # 如果全0，scale设为1避免NaN
        k_scale = tl.where(k_scale == 0, 1.0, k_scale)
        v_scale = tl.where(v_scale == 0, 1.0, v_scale)
        
        # 量化并转换为 int8 (Triton 会处理 round)
        key_int8 = (key / k_scale).to(tl.int8)
        value_int8 = (value / v_scale).to(tl.int8)
        
        # 存储 INT8 数据
        tl.store(k_cache_ptr + cache_offset, key_int8)
        tl.store(v_cache_ptr + cache_offset, value_int8)
        
        # 存储 Scale (Scale 的形状: [block, block_offset, head])
        # 这里 head_offsets 是向量，但 scale 是标量，所以我们取任意一个偏移量作为基准
        # 实际上我们只需要存一次标量，但为了符合物理结构，我们将其广播或存入特定位置
        # 为了配合 kernel 的简单性，我们假设 scale cache 的布局也是每 token 每 head 一个值
        # offset 计算: block * block_size * num_heads + offset * num_heads + head
        scale_offset = (block_idx * block_size * num_kv_heads + 
                        block_offset * num_kv_heads + 
                        head_idx)
        
        # 将标量转换为向量以便存储 (Triton 的 store 通常需要匹配宽度，或者使用标量存储)
        # 这里我们简单地将 scale 转换回 fp16 存储
        tl.store(k_scale_ptr + scale_offset, k_scale.to(tl.float16))
        tl.store(v_scale_ptr + scale_offset, v_scale.to(tl.float16))
    else:
        # === FP16 默认逻辑 ===
        tl.store(k_cache_ptr + cache_offset, key)
        tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int,
    k_scale: torch.Tensor = None,
    v_scale: torch.Tensor = None
):
    num_tokens, num_kv_heads, head_dim = key.shape
    
    if not key.is_contiguous(): key = key.contiguous()
    if not value.is_contiguous(): value = value.contiguous()
    
    grid = (num_tokens, num_kv_heads)
    
    # 判断是否使用 INT8
    use_int8 = (k_cache.dtype == torch.int8)
    
    # 由于 Triton Jit 需要 constexpr，我们在调用时进行分支
    # 或者传递 constexpr 参数。这里我们通过两个 wrapper 简化，或者直接传递 bool 让编译器推导
    # 为了性能，建议分开两个 kernel，但为了代码简洁，这里使用 constexpr 参数
    
    store_kvcache_kernel[grid](
        key, value,
        k_cache, v_cache,
        k_scale, v_scale, # 传入 scale tensor
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        USE_INT8=use_int8
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    start_m = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len:
        return
    
    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Mask for valid positions
        mask_n = offs_n < seq_len
        
        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Apply causal mask: only attend to positions <= current position
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Rescale previous accumulator
        acc = acc * alpha[:, None]
        
        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs
    
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_varlen_kernel[grid](
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


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr, # 新增
    v_scale_ptr, # 新增
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_INT8: tl.constexpr
):
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
            
            # 预处理循环：加载 K 并计算 Score
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # 加载 K Cache
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            
                            if USE_INT8:
                                # === INT8 反量化读取 ===
                                k_vec_int8 = tl.load(k_cache_ptr + k_offset)
                                # 加载对应的 Scale
                                scale_offset = (physical_block_idx * block_size * num_kv_heads +
                                               block_offset * num_kv_heads + kv_head_idx)
                                k_s = tl.load(k_scale_ptr + scale_offset)
                                # 反量化: int8 * scale -> float32
                                k_vec = k_vec_int8.to(tl.float32) * k_s.to(tl.float32)
                            else:
                                k_vec = tl.load(k_cache_ptr + k_offset).to(tl.float32)
                            
                            score = tl.sum(q.to(tl.float32) * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            
            qk = tl.where(mask_n, qk, -1e10)
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            
            acc = acc * alpha
            l_i = l_i * alpha
            
            # 预处理循环：加载 V 并累加
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
                            
                            if USE_INT8:
                                # === INT8 反量化读取 ===
                                v_vec_int8 = tl.load(v_cache_ptr + v_offset)
                                scale_offset = (physical_block_idx * block_size * num_kv_heads +
                                               block_offset * num_kv_heads + kv_head_idx)
                                v_s = tl.load(v_scale_ptr + scale_offset)
                                v_vec = v_vec_int8.to(tl.float32) * v_s.to(tl.float32)
                            else:
                                v_vec = tl.load(v_cache_ptr + v_offset).to(tl.float32)

                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
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
    k_scale: torch.Tensor = None,
    v_scale: torch.Tensor = None
) -> torch.Tensor:
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    query = query.contiguous()
    output = torch.empty_like(query)
    BLOCK_N = 64 if head_dim <= 128 else 32
    grid = (batch_size, num_heads)
    
    use_int8 = (k_cache.dtype == torch.int8)
    
    paged_attention_decode_kernel[grid](
        output, query, k_cache, v_cache,
        k_scale, v_scale,
        block_tables, context_lens,
        scale=scale,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size,
        max_num_blocks=max_num_blocks, BLOCK_N=BLOCK_N,
        USE_INT8=use_int8
    )
    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])
        # 新增 scale 缓存引用
        self.k_scale = self.v_scale = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        k_scale, v_scale = self.k_scale, self.v_scale

        # Store current k, v into cache
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # ... (处理形状逻辑同原文件) ...
            if k.dim() == 4:
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            # 调用新的 store_kvcache，传入 scale
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size, k_scale, v_scale)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill 逻辑：注意这里 Flash Attention 需要读取的是 FP16/BF16 的 K/V
            # 当前逻辑中，k 和 v 还是刚计算出来的 FP16，直接使用即可
            # 不需要从 Cache 读取，所以 Prefill 阶段不需要修改反量化逻辑
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None: raise ValueError("cu_seqlens_q missing")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # Decode 逻辑：需要从 Cache 读取，传入 scale
            o = paged_attention_decode(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                k_scale, # 传入 scale
                v_scale
            )
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")