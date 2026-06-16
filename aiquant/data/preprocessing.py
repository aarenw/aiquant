"""数据预处理：标签创建、标准化、滑动窗口序列、时间分割

处理流程：
  1. create_trend_labels()  → ZigZag 提取极值，生成三分类趋势标签
  2. rolling_zscore_normalize() → 滚动 Z-score 标准化（无前视偏差）
  3. create_sequences()     → 滑动窗口生成 (seq_len, n_features) 序列
  4. time_based_split()     → 按时间切分 train/val/test（非随机切分）

标签定义（ZigZag + target2，预测下一交易日趋势翻转）：
  - UP_TO_DOWN (0)：  target2 由正转负（由涨转跌）
  - CONTINUE (1)：    target2 符号不变（维持趋势）
  - DOWN_TO_UP (2)：  target2 由负转正（由跌转涨）

  参考 MetaQuotes NeuroBook ZigZag 标签构造法。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aiquant.config import DataConfig
from aiquant.data.zigzag import create_zigzag_labels


class TrendLabel:
    """趋势标签常量定义。"""
    UP_TO_DOWN = 0  # 由涨转跌
    CONTINUE = 1    # 维持趋势
    DOWN_TO_UP = 2  # 由跌转涨
    NAMES = ["UP_TO_DOWN", "CONTINUE", "DOWN_TO_UP"]


def create_trend_labels(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    depth: int = 5,
    backstep: int = 3,
    threshold: float = 0.01,
    atr_k: float = 0.5,
) -> pd.Series:
    """基于 ZigZag 极值与 target2 符号翻转创建三分类标签。

    Args:
        close:     收盘价 Series
        high:      最高价 Series（可选，默认用 close）
        low:       最低价 Series（可选，默认用 close）
        depth:     ZigZag Depth
        backstep:  ZigZag Backstep
        threshold: Deviation 固定比例回退值
        atr_k:     ATR 自适应 Deviation 系数

    Returns:
        标签 Series：0=UP_TO_DOWN, 1=CONTINUE, 2=DOWN_TO_UP, -1=无效
    """
    return create_zigzag_labels(
        close, high, low, depth, backstep, threshold, atr_k,
    )


def rolling_zscore_normalize(
    df: pd.DataFrame, feature_cols: list[str], window: int = 252
) -> pd.DataFrame:
    """滚动 Z-score 标准化，彻底避免前视偏差。

    公式：z_t = (x_t - mean(x_{t-window:t})) / std(x_{t-window:t})
    仅使用当前时刻之前的数据计算均值和标准差，不泄露未来信息。
    异常值裁剪至 [-3, 3] 防止极端值影响训练。

    Args:
        df:           含特征列的 DataFrame
        feature_cols: 需要标准化的特征列名列表
        window:       滚动窗口大小（默认 252 约 1 年）

    Returns:
        特征已标准化的 DataFrame（原始 OHLCV 列不变）
    """
    result = df.copy()
    for col in feature_cols:
        rolling_mean = result[col].rolling(window, min_periods=window // 2).mean()
        rolling_std = result[col].rolling(window, min_periods=window // 2).std()
        rolling_std = rolling_std.replace(0, 1)  # 避免除以零
        result[col] = (result[col] - rolling_mean) / rolling_std
        result[col] = result[col].clip(-3, 3)  # 裁剪极端值
    return result


def create_sequences(
    features: np.ndarray, labels: np.ndarray, seq_len: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """使用滑动窗口从时间序列创建训练样本。

    对于每个时间步 i（i ≥ seq_len），取 [i-seq_len, i) 作为输入特征序列，
    取第 i 个时间步的标签作为目标。

    Args:
        features: (n_samples, n_features) 特征矩阵（时间顺序）
        labels:   (n_samples,) 标签数组

    Returns:
        X: (n_sequences, seq_len, n_features)  输入序列
        y: (n_sequences,)                       目标标签
    """
    n = len(features)
    if n <= seq_len:
        return np.empty((0, seq_len, features.shape[1])), np.empty(0, dtype=np.int64)

    X = np.stack([features[i : i + seq_len] for i in range(n - seq_len)])
    y = labels[seq_len:]  # 序列结束日对应的下一交易日趋势标签
    return X, y


@dataclass
class SplitData:
    """存储 train/val/test 三个子集的特征和标签。"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def time_based_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    config: DataConfig | None = None,
) -> SplitData:
    """基于时间的数据分割，保证无数据泄露。

    分割策略（默认）：
      Train:  2010-01-01 ~ 2020-12-31
      Val:    2021-01-01 ~ 2022-12-31
      Test:   2023-01-01 ~ 2025-12-31

    注意：每个分区内部独立生成滑动窗口序列，不跨分区，
    避免训练数据中出现来自验证/测试时段的未来信息。

    Args:
        df:          含日期、特征列和标签列的 DataFrame
        feature_cols: 特征列名列表
        label_col:   标签列名
        config:      数据配置，含分割日期

    Returns:
        SplitData：含三个分区的 (X, y) 数组
    """
    config = config or DataConfig()
    seq_len = config.seq_len

    valid_mask = df[label_col] >= 0
    df_valid = df[valid_mask].reset_index(drop=True)

    dates = pd.to_datetime(df_valid["date"])
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end

    features = df_valid[feature_cols].values.astype(np.float32)
    labels = df_valid[label_col].values.astype(np.int64)

    def _split_partition(mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """对单个分区提取特征和标签，生成滑动窗口序列。"""
        idx = mask.values.nonzero()[0]
        if len(idx) < seq_len + 1:
            return (
                np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )
        f = features[idx]
        la = labels[idx]
        return create_sequences(f, la, seq_len)

    X_train, y_train = _split_partition(train_mask)
    X_val, y_val = _split_partition(val_mask)
    X_test, y_test = _split_partition(test_mask)

    return SplitData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


def prepare_single_stock(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """对单只股票完成标签创建 + 标准化的完整预处理流程。

    Args:
        df:           原始 OHLCV + 特征工程后的 DataFrame
        feature_cols: 特征列名列表
        config:       数据配置

    Returns:
        清洗后的 DataFrame（含 label 列，已过滤 NaN 和无效标签）
    """
    config = config or DataConfig()

    df = df.copy()
    df["label"] = create_trend_labels(
        df["close"], df["high"], df["low"],
        config.zigzag_depth, config.zigzag_backstep,
        config.trend_threshold, config.atr_k,
    )
    df = rolling_zscore_normalize(df, feature_cols, config.norm_window)

    initial_nans = df[feature_cols].isna().any(axis=1)
    valid_mask = ~initial_nans & (df["label"] >= 0)
    df = df[valid_mask].reset_index(drop=True)

    return df
