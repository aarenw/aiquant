"""正弦位置编码（Sinusoidal Positional Encoding）

Transformer 本身不具有序列顺序感知能力，位置编码将时间步信息
注入到每个位置的向量中，使模型能区分不同时间步的输入。

编码公式（对应参考书第 5.1 节）：
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

偶数维度用 sin，奇数维度用 cos，不同频率组合使每个位置拥有唯一编码。
位置编码以 buffer 形式注册（不参与梯度更新）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码模块。"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度，与输入向量维度一致
            max_len: 支持的最大序列长度（通常略大于 seq_len）
            dropout: 位置编码后的 Dropout 比例
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算所有位置的编码矩阵，形状 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # position: (max_len, 1)，表示各时间步的位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: 频率衰减项，维度越高频率越低
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度：cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，方便广播到 batch

        # 注册为 buffer：随模型保存但不参与梯度计算
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将位置编码叠加到输入特征上。

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + PE：形状与输入相同
        """
        x = x + self.pe[:, : x.size(1)]  # 截取与当前序列等长的位置编码
        return self.dropout(x)
