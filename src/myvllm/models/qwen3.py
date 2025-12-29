from myvllm.layers import *
import torch 
import torch.nn as nn

# Qwen3Attention: 
# qkv projection
# if not qkv_bias: then rms_norm
# apply rotary embedding to q, k
# attention
# output projection
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.qkv_projection = QKVColumnParallelLinear(
            input_size=head_dim * self.total_num_heads,
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        self.rms_norm = LayerNorm(torch.ones(head_dim))

        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads
        )

        self.output_projection = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
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

        # only applied to q and k because these two participates in the attention_weight computation
        # to remove the possibility that there is big number in q or k that causes instability in softmax
        if self.qkv_bias is False:
            q = self.rms_norm(q)  
            k = self.rms_norm(k)

        q, k = self.rotary_emb(positions, q, k) 

        o = self.attention(q, k, v)
        # o shape: (B*N, num_heads, head_dim)     - Per-GPU, different heads per GPU

        # ===== Output Projection (Row Parallel - COMMUNICATION HAPPENS HERE by dist.all_reduce) =====
        o = self.output_projection(o)
        # Input: (B*N, num_heads * head_dim) sharded across GPUs
        # Output: (B*N, hidden_size) REPLICATED on all GPUs (after all_reduce)

        return o

# Qwen3MLP
# gate_up
# activateion
# gate_down
class Qwen3MLP(nn.Module):
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
        self.gate_down = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_down(self.activation(self.gate_up(x)))
        return x


# Qwen3DecoderLayer
# input_layernorm, also consider residual
# self_attn
# layer_norm post attention
# mlp
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position
        )
        self.post_attention_layernorm = LayerNorm(gamma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            x = self.input_layernorm(x)
            residual = x
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

# Qwen3Model
# embedding
# layers stack
# final layer norm
class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
    ):
        super().__init__()
        self.embedding_layer = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim = hidden_size
        )
        self.layer_stack = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.final_layernorm = LayerNorm(gamma)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(input_ids)
        residual = None
        for layer in self.layer_stack:
            x, residual = layer(x, residual)
        x, _ = self.final_layernorm(x, residual)
        return x



# Qwen3ForCausalLM
# add lm_head on top of Qwen3Model
class Qwen3ForCausalLM(nn.Module):
    packed_module_mapping = {
        "q_proj": ('q_proj', 'q'),
        "k_proj": ('k_proj', 'k'),
        "v_proj": ('v_proj', 'v'),
        "gate_up": ('gate_up_proj', '0'),
        "gate_down": ('gate_down_proj', '1'),
    }
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.qwen3_model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.qwen3_model.embedding_layer.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.qwen3_model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    model = Qwen3ForCausalLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    )
    input_ids = torch.randint(0, 50257, (2, 16)).cuda()
    output = model(input_ids)
    print(output.shape)