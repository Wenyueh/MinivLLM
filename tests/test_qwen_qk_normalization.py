import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvllm.models.qwen3 import Qwen3Attention


class IdentityRotary(nn.Module):
    def forward(self, positions, query, key):
        return query, key


class CaptureAttention(nn.Module):
    def forward(self, query, key, value):
        self.query = query.detach().clone()
        self.key = key.detach().clone()
        return query.flatten(-2)


@pytest.mark.parametrize("qkv_bias", [False, True])
@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 8)])
def test_qk_are_normalized_with_and_without_projection_bias(
    monkeypatch, qkv_bias, shape
):
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    layer = Qwen3Attention(
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        num_kv_heads=1,
        qkv_bias=qkv_bias,
        max_position=16,
    )
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.rand(parameter.shape, generator=generator) + 0.1)
    layer.rotary_emb = IdentityRotary()
    layer.attention = CaptureAttention()
    inputs = torch.rand(shape, generator=generator)
    qkv = layer.qkv_projection(inputs)
    query, key, _ = qkv.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    query = query.reshape(*shape[:-1], layer.num_heads, layer.head_dim)
    key = key.reshape(*shape[:-1], layer.num_kv_heads, layer.head_dim)

    def reference(value, norm):
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.eps)
            * norm.weight
        )

    with torch.compiler.set_stance("force_eager"):
        layer(inputs, positions=torch.arange(shape[-2]))

    torch.testing.assert_close(layer.attention.query, reference(query, layer.q_norm))
    torch.testing.assert_close(layer.attention.key, reference(key, layer.k_norm))
