import torch
import pytest

from aiquant.models.transformer_trend.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        batch, heads, seq_len, d_k = 2, 4, 10, 16
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)

        out, weights = scaled_dot_product_attention(q, k, v)

        assert out.shape == (batch, heads, seq_len, d_k)
        assert weights.shape == (batch, heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        q = torch.randn(1, 2, 5, 8)
        k = torch.randn(1, 2, 5, 8)
        v = torch.randn(1, 2, 5, 8)

        _, weights = scaled_dot_product_attention(q, k, v)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_gradient_flow(self):
        q = torch.randn(1, 1, 3, 4, requires_grad=True)
        k = torch.randn(1, 1, 3, 4, requires_grad=True)
        v = torch.randn(1, 1, 3, 4, requires_grad=True)

        out, _ = scaled_dot_product_attention(q, k, v)
        out.sum().backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestMultiHeadAttention:
    def test_output_shape(self):
        d_model, n_heads = 64, 4
        mha = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(2, 10, d_model)
        out = mha(x, x, x)

        assert out.shape == (2, 10, d_model)

    def test_different_seq_lengths(self):
        d_model, n_heads = 32, 4
        mha = MultiHeadAttention(d_model, n_heads)

        for seq_len in [1, 5, 20, 60]:
            x = torch.randn(1, seq_len, d_model)
            out = mha(x, x, x)
            assert out.shape == (1, seq_len, d_model)

    def test_attention_weights_stored(self):
        mha = MultiHeadAttention(16, 2)
        x = torch.randn(1, 5, 16)
        mha(x, x, x)

        assert mha.attn_weights is not None
        assert mha.attn_weights.shape == (1, 2, 5, 5)
