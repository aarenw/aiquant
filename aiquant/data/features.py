"""特征工程管线：原始 OHLCV → 完整特征矩阵

特征分为四类，合计约 37 列：

1. 价格衍生特征（~13 列）：
   - 对数收益率（1 列）
   - 多周期收益率 1/5/10/20 日（4 列）
   - 高低幅 (H-L)/C（1 列）
   - 开收幅 (C-O)/O（1 列）
   - 量比 V/MA20(V)（1 列）
   - 均线偏离率 (C-SMAn)/SMAn，n=5/10/20/50（4 列）
   - 20 日滚动波动率（1 列）

2. 技术指标（~18 列）：
   RSI、MACD×3、布林带×2、ADX×3、ATR、随机指标×2、CCI、OBV变化率、
   Williams %R、ROC

3. 量价关系特征（3 列）：
   - 量价一致性（price_vol_agree）：+1 量价同向，-1 量价背离
   - 成交量动量（volume_momentum）：相对 5 日均量的偏离
   - 短/长期波动率比（vol_ratio_5_20）：识别波动率扩张/收缩

4. 时间特征（2 列）：
   - 星期（归一化到 [0, 1]）
   - 月份（归一化到 [0, 1]）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiquant.data.indicators import compute_all_indicators

FEATURE_COLUMNS: list[str] = []  # 动态由 get_feature_columns() 填充


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始 OHLCV 数据构建全部特征。

    Args:
        df: 含 date/open/high/low/close/volume 的 DataFrame

    Returns:
        新增全部特征列的 DataFrame（前几行因 lookback 含 NaN，后续流程会清理）
    """
    # 先计算技术指标
    result = compute_all_indicators(df)

    c = result["close"]
    o = result["open"]
    h = result["high"]
    l = result["low"]  # noqa: E741
    v = result["volume"]

    # ── 价格收益率特征 ────────────────────────────────────────────────────
    # 对数收益率：ln(C_t / C_{t-1})，近似等于百分比收益率，但数学性质更好
    result["log_return"] = np.log(c / c.shift(1))

    # 多周期百分比收益率：反映不同时间尺度的动量
    for n in [1, 5, 10, 20]:
        result[f"return_{n}d"] = c.pct_change(n)

    # ── 价格结构特征 ──────────────────────────────────────────────────────
    # 高低幅：衡量当日波动幅度（已归一化）
    result["hl_range"] = (h - l) / c
    # 开收幅：衡量当日涨跌及收盘位置
    result["co_range"] = (c - o) / o

    # ── 成交量特征 ────────────────────────────────────────────────────────
    # 量比：当日成交量相对 20 日均量的倍数，>2 为放量信号
    vol_ma20 = v.rolling(20).mean()
    result["volume_ratio"] = v / vol_ma20

    # ── 趋势特征（均线偏离率）────────────────────────────────────────────
    # 价格相对各均线的偏离程度，反映趋势强弱和均值回归信号
    for n in [5, 10, 20, 50]:
        sma = c.rolling(n).mean()
        result[f"sma_bias_{n}"] = (c - sma) / sma

    # ── 波动率特征 ────────────────────────────────────────────────────────
    # 20 日滚动对数收益率标准差，反映近期价格不确定性
    result["volatility_20"] = result["log_return"].rolling(20).std()

    # ── 量价关系特征 ──────────────────────────────────────────────────────
    # 量价一致性：价格与成交量方向是否一致（同向=趋势可靠，背离=趋势衰竭）
    price_up = (c - c.shift(1)) > 0
    vol_up = v > v.shift(1)
    result["price_vol_agree"] = ((price_up == vol_up).astype(float) * 2 - 1)

    # 成交量动量：当日成交量相对 5 日均量的偏离（量能加速为正，减速为负）
    vol_ma5 = v.rolling(5).mean()
    result["volume_momentum"] = (v - vol_ma5) / vol_ma5.where(vol_ma5 > 0, 1)

    # 波动率比：5 日短期波动率 / 20 日长期波动率，>1 表示波动率正在扩张
    vol_5 = result["log_return"].rolling(5).std()
    result["vol_ratio_5_20"] = vol_5 / (result["volatility_20"] + 1e-8)

    # ── 时间特征 ──────────────────────────────────────────────────────────
    # 归一化到 [0, 1]，使模型感知周内/年内周期性规律
    if pd.api.types.is_datetime64_any_dtype(result["date"]):
        result["day_of_week"] = result["date"].dt.dayofweek / 4.0  # 0=周一, 1=周五
        result["month"] = result["date"].dt.month / 12.0
    else:
        dates = pd.to_datetime(result["date"])
        result["day_of_week"] = dates.dt.dayofweek / 4.0
        result["month"] = dates.dt.month / 12.0

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """返回所有用于模型训练的特征列名（排除原始 OHLCV 和 date 列）。

    Args:
        df: 经过 engineer_features 处理的 DataFrame

    Returns:
        特征列名列表（约 34 列）
    """
    exclude = {"date", "open", "high", "low", "close", "volume", "average", "barCount"}
    return [col for col in df.columns if col not in exclude]
