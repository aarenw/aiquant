from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 123
    timeout: int = 60
    readonly: bool = True


@dataclass
class DataConfig:
    start_year: int = 2010
    end_year: int = 2025
    bar_size: str = "1 day"
    what_to_show: str = "ADJUSTED_LAST"
    use_rth: bool = True
    seq_len: int = 60
    horizon: int = 5
    trend_threshold: float = 0.01
    norm_window: int = 252
    train_end: str = "2020-12-31"
    val_end: str = "2022-12-31"


@dataclass
class TrainConfig:
    batch_size: int = 256
    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    seed: int = 42
    device: torch.device = field(default_factory=get_device)
