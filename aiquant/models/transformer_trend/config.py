from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_features: int = 33
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    seq_len: int = 60
    n_classes: int = 3
