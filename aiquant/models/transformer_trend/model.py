"""TrendTransformer：基于注意力机制的股票趋势预测模型

完整网络结构（输入为 60 天 × 33 特征的时间序列）：

    (batch, seq_len, n_features)
        ↓ Linear 输入投影
    (batch, seq_len, d_model)
        ↓ 正弦位置编码
    (batch, seq_len, d_model)
        ↓ Transformer Encoder × n_layers
          每层：Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm
    (batch, seq_len, d_model)
        ↓ 时间维度均值池化（将序列信息聚合为单一向量）
    (batch, d_model)
        ↓ MLP 分类头（两层线性 + GELU + Dropout）
    (batch, n_classes)  →  0=下跌 / 1=震荡 / 2=上涨
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig
from .positional import PositionalEncoding
from .encoder import TransformerEncoder


class TrendTransformer(nn.Module):
    """股票趋势预测 Transformer 模型。"""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: 模型超参数，详见 TransformerConfig
        """
        super().__init__()
        self.config = config

        # ── 输入投影层 ──────────────────────────────────────────────────────
        # 将原始特征维度 n_features 映射到统一的模型维度 d_model
        self.input_projection = nn.Linear(config.n_features, config.d_model)

        # ── 位置编码 ────────────────────────────────────────────────────────
        # 为每个时间步注入位置信息，使模型感知时序顺序
        self.positional_encoding = PositionalEncoding(
            config.d_model, max_len=config.seq_len + 10, dropout=config.dropout
        )

        # ── Transformer Encoder ─────────────────────────────────────────────
        # n_layers 层 Multi-Head Self-Attention + FFN
        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        # ── 分类头 ──────────────────────────────────────────────────────────
        # 池化后经两层 MLP 输出各类别的 logits
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout * 2),  # 分类头使用更大 Dropout 防过拟合
            nn.Linear(config.d_model // 2, config.n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化所有权重矩阵，加速收敛。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, n_features) —— 输入时间序列特征

        Returns:
            logits: (batch, n_classes) —— 各类别的未归一化得分
        """
        # 1. 特征维度投影到 d_model
        x = self.input_projection(x)

        # 2. 叠加正弦位置编码，注入时序位置信息
        x = self.positional_encoding(x)

        # 3. 通过 Transformer Encoder，提取时序依赖关系
        x = self.encoder(x)

        # 4. 时间维度均值池化，将 (batch, seq_len, d_model) → (batch, d_model)
        x = x.mean(dim=1)

        # 5. 分类头输出 logits
        return self.classifier(x)

    def get_attention_weights(self) -> list[torch.Tensor]:
        """获取各 Encoder 层的注意力权重，用于可视化分析。

        Returns:
            list of (batch, n_heads, seq_len, seq_len)，长度为 n_layers
        """
        weights = []
        for layer in self.encoder.layers:
            if layer.self_attn.attn_weights is not None:
                weights.append(layer.self_attn.attn_weights.detach())
        return weights

    def count_parameters(self) -> int:
        """统计可训练参数总量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
