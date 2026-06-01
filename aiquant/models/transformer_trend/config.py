"""Transformer 趋势预测模型的超参数配置"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """TrendTransformer 模型结构超参数。

    默认值适用于约 33 维特征、60 天回看窗口的场景。
    """
    n_features: int = 33   # 输入特征维度（由特征工程管线决定）
    d_model: int = 64      # 模型内部表示维度
    n_heads: int = 4       # Multi-Head Attention 的头数（需整除 d_model）
    n_layers: int = 3      # Transformer Encoder 堆叠层数
    d_ff: int = 256        # 前馈网络中间层维度（通常为 d_model 的 4 倍）
    dropout: float = 0.1   # Dropout 比例，用于正则化
    seq_len: int = 60      # 输入序列长度（与 DataConfig.seq_len 保持一致）
    n_classes: int = 3     # 输出类别数：0=DOWN, 1=SIDEWAYS, 2=UP
