import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvllm.models.llama import LlamaAttn


@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1), (4, 0), (4, 3)])
def test_attention_metadata_matches_rank_local_qkv(monkeypatch, world_size, rank):
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    layer = LlamaAttn(
        hidden_size=64,
        head_dim=8,
        num_qo_heads=8,
        num_kv_heads=4,
        max_position_embeddings=16,
    )
    projected = layer.qkv_projection(torch.zeros(3, 64))
    q, k, v = projected.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1)

    assert q.shape[-1] == layer.attention.num_heads * layer.attention.head_dim
    assert k.shape[-1] == layer.attention.num_kv_heads * layer.attention.head_dim
    assert v.shape == k.shape
