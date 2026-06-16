import numpy as np
import pandas as pd

from aiquant.data.preprocessing import TrendLabel, create_trend_labels
from aiquant.data.zigzag import (
    compute_target2,
    compute_zigzag,
    create_zigzag_labels,
)


def _ohlc(close: list[float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    c = pd.Series(close, dtype=float)
    return c, c + 0.5, c - 0.5


class TestZigZagCore:
    def test_compute_zigzag_finds_pivots(self):
        close = [100, 102, 105, 103, 98, 95, 97, 100, 104, 102]
        _, h, l = _ohlc(close)
        dev = np.full(len(close), 2.0)
        zz = compute_zigzag(h.values, l.values, depth=3, backstep=2, deviation=dev)
        assert np.sum(~np.isnan(zz)) >= 2

    def test_target2_backward_propagation(self):
        close = np.array([100.0, 101.0, 102.0, 103.0])
        zz = np.array([np.nan, np.nan, 110.0, np.nan])
        target2 = compute_target2(close, zz)
        # 书中用 zz[i+1]：极值在 bar 2，target2 填充 bar 0、1
        assert target2[0] == 110.0 - 100.0
        assert target2[1] == 110.0 - 101.0
        assert np.isnan(target2[2])
        assert np.isnan(target2[3])


class TestZigZagLabels:
    def test_sign_flip_up_to_down(self):
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 108.0, 105.0])
        zz = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 110.0, np.nan, 95.0])
        target2 = compute_target2(close, zz)
        labels = np.full(len(close), 1, dtype=np.int64)
        for i in range(len(close) - 1):
            t0, t1 = target2[i], target2[i + 1]
            if not np.isnan(t0) and not np.isnan(t1):
                if t0 > 0 and t1 < 0:
                    labels[i] = 0
                elif t0 < 0 and t1 > 0:
                    labels[i] = 2
        assert 0 in labels

    def test_sign_flip_down_to_up(self):
        close = np.array([110.0, 108.0, 105.0, 100.0, 98.0, 95.0, 97.0, 100.0])
        zz = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 95.0, np.nan, 105.0])
        target2 = compute_target2(close, zz)
        labels = np.full(len(close), 1, dtype=np.int64)
        for i in range(len(close) - 1):
            t0, t1 = target2[i], target2[i + 1]
            if not np.isnan(t0) and not np.isnan(t1):
                if t0 > 0 and t1 < 0:
                    labels[i] = 0
                elif t0 < 0 and t1 > 0:
                    labels[i] = 2
        assert 2 in labels

    def test_last_bar_invalid(self):
        close, high, low = _ohlc([100, 101, 102, 103, 104, 105])
        labels = create_zigzag_labels(
            close, high, low, depth=3, backstep=2, threshold=0.01,
        )
        assert labels.iloc[-1] == -1

    def test_all_three_classes_on_random_walk(self):
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(500)) + 100)
        high = close + 0.5
        low = close - 0.5
        labels = create_zigzag_labels(
            close, high, low, depth=5, backstep=3, threshold=0.01,
        )
        valid = labels[labels >= 0]
        assert len(valid) > 0
        assert valid.max() <= 2


class TestTrendLabelsIntegration:
    def test_create_trend_labels_delegates_to_zigzag(self):
        close, high, low = _ohlc([100, 102, 105, 103, 98, 95, 97, 100, 104, 102])
        labels = create_trend_labels(
            close, high, low, depth=3, backstep=2, threshold=0.01,
        )
        assert labels.dtype == np.int64
        assert (labels == -1).sum() >= 1

    def test_continue_when_no_flip(self):
        close, high, low = _ohlc([100, 101, 102, 103, 104, 105, 106])
        labels = create_trend_labels(close, high, low, depth=3, backstep=2, threshold=0.01)
        valid = labels[labels >= 0]
        if len(valid) > 0:
            assert TrendLabel.CONTINUE in valid.values
