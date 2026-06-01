"""从零实现 Scaled Dot-Product Attention 和 Multi-Head Attention

参考《Attention Is All You Need》(Vaswani et al., 2017) 公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-Head Attention 将 Q/K/V 分别投影到 h 个子空间，
各头独立计算注意力后拼接，再经线性投影输出。
对应参考书第 5.2 节 Multi-Head Attention。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention 计算。

    Args:
        query: (batch, n_heads, seq_len, d_k)
        key:   (batch, n_heads, seq_len, d_k)
        value: (batch, n_heads, seq_len, d_k)
        mask:  可选掩码张量，为 0 的位置被替换为 -inf（不参与注意力）
        dropout: 可选的注意力权重 Dropout

    Returns:
        output:           (batch, n_heads, seq_len, d_k)
        attention_weights: (batch, n_heads, seq_len, seq_len)，可用于可视化
    """
    d_k = query.size(-1)
    # QK^T / sqrt(d_k)：缩放防止点积数值过大导致 softmax 梯度消失
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 掩码位置设为 -inf，softmax 后趋近于 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # softmax 归一化得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 加权求和 Value 向量
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头自注意力模块。

    将输入同时投影到 n_heads 个子空间，各头独立计算注意力，
    最终将所有头的输出拼接后经线性变换输出。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model:  模型维度
            n_heads:  注意力头数，必须整除 d_model
            dropout:  注意力权重的 Dropout 比例
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q、K、V 和输出的线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 多头拼接后的输出投影

        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None  # 保存最近一次的注意力权重

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: 可选掩码

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 线性投影后重排为 (batch, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 各头并行计算注意力，保存权重供可视化使用
        out, self.attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # 将 n_heads 个输出拼接：(batch, seq_len, d_model)，再经线性投影
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)
