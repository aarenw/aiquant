"""全局配置：路径常量、训练设备检测、IB 连接参数及训练超参数"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # 原始 parquet 文件目录
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"           # 股票列表缓存
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"  # 模型权重保存目录


def get_device() -> torch.device:
    """自动检测可用设备：优先 Apple MPS，其次 CUDA，最后 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


@dataclass
class IBConfig:
    """Interactive Brokers TWS/Gateway 连接配置。"""
    host: str = "127.0.0.1"   # TWS/Gateway 主机地址
    port: int = 7497           # 7497=TWS模拟盘, 7496=TWS实盘, 4002=Gateway
    client_id: int = 123       # API 客户端 ID，同一时间不可重复
    timeout: int = 60          # 单次请求超时秒数
    readonly: bool = True      # 只读模式，仅下载数据时建议开启


@dataclass
class DataConfig:
    """数据获取与预处理配置。"""
    start_year: int = 2010          # 历史数据起始年份
    end_year: int = 2025            # 历史数据终止年份
    bar_size: str = "1 day"         # K 线周期
    what_to_show: str = "ADJUSTED_LAST"  # 全复权价格（含拆股+分红调整）
    use_rth: bool = True            # True=仅使用正常交易时段数据
    seq_len: int = 60               # 模型输入序列长度（交易日数）
    horizon: int = 5                # 预测未来 N 个交易日的趋势
    trend_threshold: float = 0.01  # 趋势判断阈值：±1% 以外才算涨/跌
    norm_window: int = 252          # 滚动 Z-score 标准化窗口（约1年）
    train_end: str = "2020-12-31"   # 训练集截止日期
    val_end: str = "2022-12-31"     # 验证集截止日期（之后为测试集）


@dataclass
class TrainConfig:
    """模型训练超参数配置。"""
    batch_size: int = 256           # 每批样本数
    max_epochs: int = 100           # 最大训练轮数
    lr: float = 1e-3                # 初始学习率
    weight_decay: float = 1e-4     # AdamW 权重衰减
    patience: int = 10              # 早停耐心值（验证集 F1 连续不提升的轮数）
    max_grad_norm: float = 1.0      # 梯度裁剪阈值
    label_smoothing: float = 0.1   # 标签平滑系数，缓解过拟合
    seed: int = 42                  # 随机种子，保证可复现
    device: torch.device = field(default_factory=get_device)
