from myvllm.layers import *

from typing import Tuple

import torch 
import torch.nn as nn

class LlamaAttn(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-5,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        self.total_num_heads = num_qo_heads
        self.num_heads = num_qo_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_qo_heads
        # self.num_kv_heads is per-GPU value (divided by tp_size)
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_qo_heads

        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            bias=has_attn_bias,
        )

        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        
        # Llama 3.2 does not have q_norm or k_norm

        self.rotary_emb = RotaryEmbedding(
            base=rope_base,
            rotary_embedding=head_dim,
            max_position=max_position_embeddings,
            is_llama3=True
        )
        self.attention = Attention(
            num_heads=num_qo_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
        )
        self.o_proj = RowParallelLinear(
            input_size= head_dim * num_qo_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Input: x shape (B, N, hidden_size) - REPLICATED on all GPUs

        # ===== QKV Projection (Column Parallel - THIS IS WHERE SHARDING HAPPENS) =====
        # Output shape PER GPU: (B, N, head_dim * (num_heads + 2*num_kv_heads))
        # where num_heads = total_num_heads/tp_size
        #       num_kv_heads = total_num_kv_heads/tp_size
        qkv = self.qkv_projection(x)

        # ===== Split QKV =====
        # q_size = head_dim * num_heads           - Per-GPU size!
        # kv_size = head_dim * num_kv_heads       - Per-GPU size!
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Handle both batched (3D) and varlen (2D) inputs
        # Varlen: q shape: (total_tokens, q_size) where q_size = num_heads * head_dim
        # Batched: q shape: (B, N, q_size)
        if q.dim() == 2:
            # Varlen mode: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            # Batched mode: (B, N, q_size) -> (B, N, num_heads, head_dim)
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k) 

        o = self.attention(q, k, v)
        # o shape: (B*N, num_heads, head_dim)     - Per-GPU, different heads per GPU

        # ===== Output Projection (Row Parallel - COMMUNICATION HAPPENS HERE by dist.all_reduce) =====
        o = self.o_proj(o)
        # Input: (B*N, num_heads * head_dim) sharded across GPUs
        # Output: (B*N, hidden_size) REPLICATED on all GPUs (after all_reduce)

        return o
    
# LlamaMLP
# gate_up
# activateion
# gate_down
class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x

# LlamaDecoderLayer
# input_layernorm, also consider residual
# self_attn
# layer_norm post attention
# mlp
class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_qo_heads: int = 32,
        num_kv_heads: int = 8,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-05,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        intermediate_size: int = 8192,
        ffn_bias: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma, eps=rms_norm_epsilon)
        self.self_attn = LlamaAttn(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            has_attn_bias=has_attn_bias,
            rms_norm_epsilon=rms_norm_epsilon,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            block_size=block_size,
        )
        self.post_attention_layernorm = LayerNorm(gamma, eps=rms_norm_epsilon)
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x  # Save BEFORE normalization
            x = self.input_layernorm(x)
        # Compute positions based on context (respecting sequence boundaries for batched prefill)
        from myvllm.utils import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # For batched prefill, create positions that restart at 0 for each sequence
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            # For single sequence prefill, use sequential positions
            positions = torch.arange(x.size(0), device=x.device)
        else:
            # For decode, use context_lens - 1 as positions (current position for each sequence)
            positions = context.context_lens - 1

        x = self.self_attn(x, positions=positions)
        # Residual connection always on for attention output
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual
    
# LlamaModel
# embedding
# layers stack
# final layer norm
class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_qo_heads: int = 32,
        num_kv_heads: int = 8,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-5,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        intermediate_size: int = 8192,
        ffn_bias: bool = False,
        num_layers: int = 16,
        block_size: int = 256,
    ):
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                has_attn_bias=has_attn_bias,
                rms_norm_epsilon=rms_norm_epsilon,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma, eps=rms_norm_epsilon)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(
            self,
            vocab_size: int = 128256,
            hidden_size: int = 2048,
            head_dim: int = 64,
            num_qo_heads: int = 32,
            num_kv_heads: int = 8,
            has_attn_bias: bool = False,
            rms_norm_epsilon: float = 1e-5,
            rope_base: int = 500000,
            max_position_embeddings: int = 131072,
            intermediate_size: int = 8192,
            ffn_bias: bool = False,
            num_layers: int = 16,
            block_size: int = 256,
            tie_word_embeddings: bool = True
        ):
        super().__init__()
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            has_attn_bias=has_attn_bias,
            rms_norm_epsilon=rms_norm_epsilon,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
