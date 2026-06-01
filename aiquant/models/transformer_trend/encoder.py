"""Transformer Encoder Layer 和 Encoder Stack

每个 Encoder Layer 由两个子层构成（均带有残差连接和 LayerNorm）：
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network (FFN)

结构（对应参考书第 5.1 节）：
  输入 x
    → Self-Attention(x, x, x)
    → Add & LayerNorm（残差连接防止梯度消失）
    → FFN（两层线性变换 + GELU 激活）
    → Add & LayerNorm
  → 输出
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """单个 Transformer Encoder 层（Pre-LN 风格采用 Post-LN）。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model:  模型维度
            n_heads:  多头注意力的头数
            d_ff:     前馈网络中间层维度（通常为 d_model 的 4 倍）
            dropout:  Dropout 比例
        """
        super().__init__()
        # 子层1：多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 子层2：位置感知前馈网络（两层线性 + GELU）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),          # GELU 在 Transformer 中效果优于 ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # FFN 后的层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, d_model)
            mask: 可选注意力掩码

        Returns:
            输出张量，形状与输入相同
        """
        # 残差连接1：自注意力 + Add & Norm
        attn_out = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 残差连接2：FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Stack：将多个 Encoder Layer 串行堆叠。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        """
        Args:
            d_model:  模型维度
            n_heads:  注意力头数
            d_ff:     FFN 中间层维度
            n_layers: Encoder Layer 堆叠层数
            dropout:  Dropout 比例
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """依次经过所有 Encoder Layer。

        Args:
            x:    (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
