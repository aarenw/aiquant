import numpy as np
import pytest

from aiquant.training.metrics import compute_metrics


class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        m = compute_metrics(y_true, y_pred)

        assert m.accuracy == 1.0
        assert m.macro_f1 == 1.0

    def test_random_predictions(self):
        np.random.seed(0)
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        m = compute_metrics(y_true, y_pred)

        assert 0 < m.accuracy < 1
        assert 0 < m.macro_f1 < 1

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])
        m = compute_metrics(y_true, y_pred)

        assert m.confusion.shape == (3, 3)
        assert m.confusion.sum() == len(y_true)

    def test_report_not_empty(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert len(m.report) > 0
