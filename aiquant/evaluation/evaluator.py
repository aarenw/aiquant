"""模型评估

在测试集（或任意 DataLoader）上运行推理，计算分类指标。
支持直接传入模型或从 checkpoint 文件加载后评估。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aiquant.training.metrics import compute_metrics, ClassificationMetrics
from aiquant.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """在给定数据集上评估模型性能。

    Args:
        model:       已加载权重的模型（需已移至 device）
        data_loader: 数据集 DataLoader
        device:      推理设备

    Returns:
        metrics: 分类指标汇总
        y_true:  真实标签数组
        y_pred:  预测标签数组
    """
    model.eval()
    all_preds = []
    all_labels = []

    for X, y in data_loader:
        X = X.to(device)
        logits = model(X)
        # 取 logits 最大值对应类别作为预测结果
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)

    return metrics, y_true, y_pred


def evaluate_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """从 checkpoint 文件加载权重后评估模型。

    Args:
        model:           模型实例（结构需与 checkpoint 匹配）
        checkpoint_path: checkpoint 文件路径
        data_loader:     数据集 DataLoader
        device:          推理设备

    Returns:
        与 evaluate_model 相同
    """
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    return evaluate_model(model, data_loader, device)
