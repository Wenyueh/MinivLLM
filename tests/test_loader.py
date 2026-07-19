import os
import sys

import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.utils.loader import load_weights


def test_qwen3_hf_names_use_packed_weight_loaders(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    model = Qwen3ForCausalLM(
        vocab_size=8,
        hidden_size=8,
        num_heads=4,
        head_dim=2,
        num_kv_heads=2,
        intermediate_size=8,
        num_layers=1,
        qkv_bias=False,
        ffn_bias=False,
        tie_word_embeddings=True,
    )

    tensors = {
        "model.embed_tokens.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "model.layers.0.input_layernorm.weight": torch.full((8,), 1.1),
        "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 1.2),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8) + 100,
        "model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8) + 200,
        "model.layers.0.self_attn.q_norm.weight": torch.full((2,), 1.3),
        "model.layers.0.self_attn.k_norm.weight": torch.full((2,), 1.4),
        "model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 300,
        "model.layers.0.mlp.gate_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 400,
        "model.layers.0.mlp.up_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 500,
        "model.layers.0.mlp.down_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 600,
        "model.norm.weight": torch.full((8,), 1.5),
    }

    loaded_names = load_weights(model, tensors.items())

    qkv = model.model.layers[0].self_attn.qkv_projection.weight
    torch.testing.assert_close(qkv[:4], tensors["model.layers.0.self_attn.q_proj.weight"][4:8])
    torch.testing.assert_close(qkv[4:6], tensors["model.layers.0.self_attn.k_proj.weight"][2:4])
    torch.testing.assert_close(qkv[6:8], tensors["model.layers.0.self_attn.v_proj.weight"][2:4])

    gate_up = model.model.layers[0].mlp.gate_up.weight
    torch.testing.assert_close(gate_up[:4], tensors["model.layers.0.mlp.gate_proj.weight"][4:8])
    torch.testing.assert_close(gate_up[4:8], tensors["model.layers.0.mlp.up_proj.weight"][4:8])

    torch.testing.assert_close(
        model.model.layers[0].self_attn.o_proj.weight,
        tensors["model.layers.0.self_attn.o_proj.weight"][:, 4:8],
    )
    torch.testing.assert_close(
        model.model.layers[0].mlp.down_proj.weight,
        tensors["model.layers.0.mlp.down_proj.weight"][:, 4:8],
    )
    torch.testing.assert_close(
        model.model.embed_tokens.weight,
        tensors["model.embed_tokens.weight"][4:8],
    )
    assert "model.layers.0.self_attn.qkv_projection.weight" in loaded_names
    assert "model.layers.0.mlp.gate_up.weight" in loaded_names


def test_loads_transformers_qwen3_state_dict(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    config = Qwen3Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        tie_word_embeddings=True,
    )
    hf_model = HFQwen3ForCausalLM(config)
    model = Qwen3ForCausalLM(
        vocab_size=8,
        hidden_size=8,
        num_heads=4,
        head_dim=2,
        num_kv_heads=2,
        intermediate_size=8,
        num_layers=1,
        qkv_bias=False,
        ffn_bias=False,
        tie_word_embeddings=True,
    )

    hf_weights = hf_model.state_dict()
    load_weights(model, hf_weights.items())

    qkv = model.model.layers[0].self_attn.qkv_projection.weight
    expected_qkv = torch.cat([
        hf_weights["model.layers.0.self_attn.q_proj.weight"],
        hf_weights["model.layers.0.self_attn.k_proj.weight"],
        hf_weights["model.layers.0.self_attn.v_proj.weight"],
    ])
    torch.testing.assert_close(qkv, expected_qkv)

    gate_up = model.model.layers[0].mlp.gate_up.weight
    expected_gate_up = torch.cat([
        hf_weights["model.layers.0.mlp.gate_proj.weight"],
        hf_weights["model.layers.0.mlp.up_proj.weight"],
    ])
    torch.testing.assert_close(gate_up, expected_gate_up)
