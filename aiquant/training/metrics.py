"""分类指标计算

提供基于 scikit-learn 的多分类评估指标：
- 准确率 (Accuracy)
- Macro F1 / Precision / Recall（对类别不平衡更鲁棒）
- 混淆矩阵
- 分类报告（含各类别详细指标）

三分类定义：0=UP_TO_DOWN（由涨转跌）, 1=CONTINUE（维持趋势）, 2=DOWN_TO_UP（由跌转涨）
"""

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

from aiquant.data.preprocessing import TrendLabel


@dataclass
class ClassificationMetrics:
    """分类评估指标汇总。"""
    accuracy: float = 0.0           # 总体准确率
    macro_f1: float = 0.0           # Macro-averaged F1（各类别等权平均）
    macro_precision: float = 0.0    # Macro Precision
    macro_recall: float = 0.0       # Macro Recall
    per_class_f1: list[float] = field(default_factory=list)  # 各类别 F1
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # 混淆矩阵
    report: str = ""                # 完整的 sklearn 分类报告文本


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """计算全套分类评估指标。

    Args:
        y_true:      真实标签数组
        y_pred:      预测标签数组
        class_names: 类别名称列表，默认为 TrendLabel.NAMES

    Returns:
        ClassificationMetrics 数据类
    """
    class_names = class_names or TrendLabel.NAMES

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        # zero_division=0：当某类别无预测时 F1 记为 0，避免警告
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        per_class_f1=f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
        report=classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    )
