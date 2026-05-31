"""TrendTransformer: 基于注意力机制的股票趋势预测模型"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig
from .positional import PositionalEncoding
from .encoder import TransformerEncoder


class TrendTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 输入投影: n_features → d_model
        self.input_projection = nn.Linear(config.n_features, config.d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(
            config.d_model, max_len=config.seq_len + 10, dropout=config.dropout
        )

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout * 2),
            nn.Linear(config.d_model // 2, config.n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            logits: (batch, n_classes)
        """
        # 投影到 d_model 维
        x = self.input_projection(x)

        # 添加位置编码
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.encoder(x)

        # 时间维度均值池化
        x = x.mean(dim=1)

        # 分类
        return self.classifier(x)

    def get_attention_weights(self) -> list[torch.Tensor]:
        """获取各层的注意力权重，用于可视化。"""
        weights = []
        for layer in self.encoder.layers:
            if layer.self_attn.attn_weights is not None:
                weights.append(layer.self_attn.attn_weights.detach())
        return weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
