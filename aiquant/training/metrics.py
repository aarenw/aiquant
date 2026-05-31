"""分类指标计算"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    per_class_f1: list[float] = field(default_factory=list)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    report: str = ""


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
) -> ClassificationMetrics:
    class_names = class_names or ["DOWN", "SIDEWAYS", "UP"]

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        per_class_f1=f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
        report=classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    )
