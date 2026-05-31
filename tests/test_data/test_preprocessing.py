import numpy as np
import pandas as pd
import pytest

from aiquant.data.preprocessing import (
    create_trend_labels,
    rolling_zscore_normalize,
    create_sequences,
    TrendLabel,
)


class TestTrendLabels:
    def test_basic_labels(self):
        close = pd.Series([100, 102, 98, 105, 95, 100, 103, 97, 101, 99, 100])
        labels = create_trend_labels(close, horizon=1, threshold=0.01)

        for i in range(len(close) - 1):
            ret = close.iloc[i + 1] / close.iloc[i] - 1
            if ret > 0.01:
                assert labels.iloc[i] == TrendLabel.UP
            elif ret < -0.01:
                assert labels.iloc[i] == TrendLabel.DOWN
            else:
                assert labels.iloc[i] == TrendLabel.SIDEWAYS

    def test_last_rows_invalid(self):
        close = pd.Series(range(100, 120))
        labels = create_trend_labels(close, horizon=5)
        assert (labels.iloc[-5:] == -1).all()

    def test_all_three_classes_present(self):
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(500)) + 100)
        labels = create_trend_labels(close, horizon=5, threshold=0.01)
        valid = labels[labels >= 0]
        assert set(valid.unique()) == {0, 1, 2}


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
