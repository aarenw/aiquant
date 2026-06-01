"""使用 ta-lib 计算技术指标

计算以下 11 类技术指标，共约 18 列：
  - RSI(14)：相对强弱指数，衡量超买/超卖状态
  - MACD(12,26,9)：趋势跟踪指标，含 MACD 线、信号线、柱状图
  - Bollinger Bands(20,2)：计算 %B 和带宽，衡量价格位置和波动范围
  - ADX(14)：趋势强度，含 +DI/-DI 方向指数
  - ATR(14)：真实波幅（归一化为价格比例）
  - Stochastic(14,3,3)：随机振荡器 %K/%D，判断超买/超卖
  - CCI(20)：商品通道指数，衡量价格偏离统计均值的程度
  - OBV：成交量变化率，反映资金流向
  - Williams %R(14)：与 RSI 互补的超买/超卖指标
  - ROC(10)：价格变化率，衡量动量

输入 DataFrame 必须包含 open, high, low, close, volume 列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算全部技术指标并追加到 DataFrame 的新列中。

    Args:
        df: 含 open/high/low/close/volume 的原始 OHLCV DataFrame

    Returns:
        新增指标列的 DataFrame（原始列保持不变）
    """
    # 提取 OHLCV 数组，ta-lib 要求 float64
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)  # noqa: E741
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    result = df.copy()

    # ── RSI（相对强弱指数）────────────────────────────────────────────────
    # 值域 [0, 100]；>70 超买，<30 超卖
    result["rsi_14"] = talib.RSI(c, timeperiod=14)

    # ── MACD（指数平滑异同移动平均线）─────────────────────────────────────
    # macd_hist > 0 且递增：看涨动能增强
    macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    result["macd"] = macd
    result["macd_signal"] = macd_signal
    result["macd_hist"] = macd_hist

    # ── Bollinger Bands（布林带）──────────────────────────────────────────
    # %B：价格在布林带内的相对位置（0~1，>1 超买，<0 超卖）
    # 带宽：波动性指标，带宽扩大预示趋势可能启动
    bb_upper, bb_mid, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_range = bb_upper - bb_lower
    result["bb_pctb"] = np.where(bb_range != 0, (c - bb_lower) / bb_range, 0.5)
    result["bb_width"] = np.where(bb_mid != 0, bb_range / bb_mid, 0.0)

    # ── ADX（平均方向指数）────────────────────────────────────────────────
    # ADX > 25：趋势较强；+DI > -DI：上升趋势
    result["adx_14"] = talib.ADX(h, l, c, timeperiod=14)
    result["plus_di"] = talib.PLUS_DI(h, l, c, timeperiod=14)
    result["minus_di"] = talib.MINUS_DI(h, l, c, timeperiod=14)

    # ── ATR（真实波幅，归一化为价格的百分比）──────────────────────────────
    atr = talib.ATR(h, l, c, timeperiod=14)
    result["atr_norm"] = np.where(c != 0, atr / c, 0.0)

    # ── Stochastic（随机振荡器）───────────────────────────────────────────
    # %K > 80 超买，%K < 20 超卖；%K 上穿 %D 看涨
    slowk, slowd = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
    result["stoch_k"] = slowk
    result["stoch_d"] = slowd

    # ── CCI（商品通道指数）────────────────────────────────────────────────
    # > +100 超买，< -100 超卖
    result["cci_20"] = talib.CCI(h, l, c, timeperiod=20)

    # ── OBV（能量潮）变化率 ───────────────────────────────────────────────
    # OBV 上升 + 价格上涨：量价齐升，趋势可靠
    obv = talib.OBV(c, v)
    result["obv_roc"] = pd.Series(obv).pct_change().values

    # ── Williams %R（威廉指标）────────────────────────────────────────────
    # 值域 [-100, 0]；> -20 超买，< -80 超卖（与 RSI 反向）
    result["willr_14"] = talib.WILLR(h, l, c, timeperiod=14)

    # ── ROC（价格变化率）──────────────────────────────────────────────────
    # 衡量 10 日动量，正值趋势向上
    result["roc_10"] = talib.ROC(c, timeperiod=10)

    return result
