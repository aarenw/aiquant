"""模型评估"""

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
    """在给定数据集上评估模型。

    Returns:
        metrics: 分类指标
        y_true: 真实标签
        y_pred: 预测标签
    """
    model.eval()
    all_preds = []
    all_labels = []

    for X, y in data_loader:
        X = X.to(device)
        logits = model(X)
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
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    return evaluate_model(model, data_loader, device)
