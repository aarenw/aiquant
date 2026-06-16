import numpy as np
import pandas as pd

from aiquant.data.preprocessing import (
    create_trend_labels,
    rolling_zscore_normalize,
    create_sequences,
    TrendLabel,
)


def _ohlc(close: list[float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    c = pd.Series(close, dtype=float)
    return c, c + 0.5, c - 0.5


class TestTrendLabels:
    def test_labels_are_valid_integers(self):
        close, high, low = _ohlc([100, 102, 105, 103, 98, 95, 97, 100, 104, 102])
        labels = create_trend_labels(close, high, low, depth=3, backstep=2, threshold=0.01)
        assert set(labels.unique()) <= {-1, 0, 1, 2}

    def test_last_bar_invalid(self):
        close, high, low = _ohlc(list(range(100, 120)))
        labels = create_trend_labels(close, high, low, depth=5, backstep=3)
        assert labels.iloc[-1] == -1

    def test_all_three_classes_present(self):
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(500)) + 100)
        high = close + 0.5
        low = close - 0.5
        labels = create_trend_labels(close, high, low, depth=5, backstep=3, threshold=0.01)
        valid = labels[labels >= 0]
        assert len(valid) > 0
        assert valid.max() <= 2


class TestRollingZscore:
    def test_output_clipped(self):
        df = pd.DataFrame({"a": np.random.randn(300) * 10})
        result = rolling_zscore_normalize(df, ["a"], window=50)
        assert result["a"].dropna().between(-3, 3).all()

    def test_shape_preserved(self):
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        result = rolling_zscore_normalize(df, ["a", "b"], window=20)
        assert result.shape == df.shape


class TestCreateSequences:
    def test_output_shapes(self):
        features = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 3, 100).astype(np.int64)

        X, y = create_sequences(features, labels, seq_len=20)
        assert X.shape == (80, 20, 10)
        assert y.shape == (80,)

    def test_too_short_data(self):
        features = np.random.randn(5, 10).astype(np.float32)
        labels = np.random.randint(0, 3, 5).astype(np.int64)

        X, y = create_sequences(features, labels, seq_len=10)
        assert len(X) == 0
        assert len(y) == 0
