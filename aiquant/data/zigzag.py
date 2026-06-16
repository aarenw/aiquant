"""ZigZag 指标：过滤噪声、提取极值并生成趋势标签

实现参考 MetaQuotes《Neural Networks for Algorithmic Trading with MQL5》：
  - 用 ZigZag 识别显著极值（Depth / Deviation / Backstep）
  - 反向遍历得到 target2 = 未来最近极值 - close（运动强度与方向）
  - 由 target2 符号在相邻 bar 间的翻转定义三分类趋势标签

https://www.mql5.com/en/neurobook/index/realization/task
https://www.mql5.com/en/neurobook/index/realization/initial_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def compute_deviation(
    close: np.ndarray,
    high: np.ndarray | None,
    low: np.ndarray | None,
    threshold: float,
    atr_k: float,
) -> np.ndarray:
    """计算 ZigZag Deviation（绝对价格幅度），优先 ATR 自适应。"""
    n = len(close)
    if high is not None and low is not None:
        atr = talib.ATR(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            timeperiod=14,
        )
        atr_norm = np.where(close != 0, atr / close, threshold)
        pct = np.maximum(atr_k * atr_norm, 0.005)
        return pct * close
    return np.full(n, threshold * close)


def compute_zigzag(
    high: np.ndarray,
    low: np.ndarray,
    depth: int,
    backstep: int,
    deviation: np.ndarray,
) -> np.ndarray:
    """MQL5 风格 ZigZag：在 pivot bar 写入极值价格，其余为 NaN。

    Args:
        high:       最高价数组
        low:        最低价数组
        depth:      寻找极值的最小 K 线窗口
        backstep:   相邻极值之间的最小 K 线间隔
        deviation:  确认 pivot 所需的最小反向幅度（绝对价格）

    Returns:
        与输入等长的数组，pivot 位置为极值价格，其余 NaN
    """
    n = len(high)
    zz = np.full(n, np.nan)
    if n == 0:
        return zz

    depth = max(1, depth)
    backstep = max(1, backstep)

    leg = -1  # 1=上涨腿（追踪高点），-1=下跌腿（追踪低点）
    cand_idx = 0
    cand_price = low[0]
    last_pivot_idx = 0

    for i in range(depth, n):
        dev = deviation[i]
        if dev <= 0 or np.isnan(dev):
            continue

        if leg == 1:
            if high[i] >= cand_price:
                cand_idx = i
                cand_price = high[i]
            if cand_price - low[i] >= dev and i - last_pivot_idx >= backstep:
                zz[cand_idx] = cand_price
                last_pivot_idx = cand_idx
                leg = -1
                cand_idx = i
                cand_price = low[i]
        else:
            if low[i] <= cand_price:
                cand_idx = i
                cand_price = low[i]
            if high[i] - cand_price >= dev and i - last_pivot_idx >= backstep:
                zz[cand_idx] = cand_price
                last_pivot_idx = cand_idx
                leg = 1
                cand_idx = i
                cand_price = high[i]

    return zz


def compute_target2(close: np.ndarray, zz: np.ndarray) -> np.ndarray:
    """按书中算法反向传播最近未来极值，得到 target2 = extremum - close。"""
    n = len(close)
    target2 = np.full(n, np.nan)
    extremum = np.nan

    for i in range(n - 2, -1, -1):
        pivot = zz[i + 1]
        if not np.isnan(pivot):
            extremum = pivot
        if not np.isnan(extremum):
            target2[i] = extremum - close[i]

    return target2


def create_zigzag_labels(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    depth: int = 5,
    backstep: int = 3,
    threshold: float = 0.01,
    atr_k: float = 0.5,
) -> pd.Series:
    """基于 ZigZag target2 符号翻转生成三分类趋势标签。

    在 bar t：
      - target2[t] > 0 且 target2[t+1] < 0 → 由涨转跌（UP_TO_DOWN）
      - target2[t] < 0 且 target2[t+1] > 0 → 由跌转涨（DOWN_TO_UP）
      - 其余 → 维持趋势（CONTINUE）

    Returns:
        0=UP_TO_DOWN, 1=CONTINUE, 2=DOWN_TO_UP, -1=无效
    """
    c = close.values.astype(np.float64)
    h = high.values.astype(np.float64) if high is not None else c.copy()
    l = low.values.astype(np.float64) if low is not None else c.copy()

    dev = compute_deviation(c, h, l, threshold, atr_k)
    zz = compute_zigzag(h, l, depth, backstep, dev)
    target2 = compute_target2(c, zz)

    n = len(c)
    labels = np.full(n, 1, dtype=np.int64)  # CONTINUE

    for i in range(n - 1):
        t0, t1 = target2[i], target2[i + 1]
        if np.isnan(t0) or np.isnan(t1):
            labels[i] = -1
        elif t0 > 0 and t1 < 0:
            labels[i] = 0  # UP_TO_DOWN
        elif t0 < 0 and t1 > 0:
            labels[i] = 2  # DOWN_TO_UP

    labels[-1] = -1
    return pd.Series(labels, index=close.index, dtype=np.int64)
