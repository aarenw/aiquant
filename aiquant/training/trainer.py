"""训练循环

训练策略：
- 优化器：AdamW（带权重衰减，相比 Adam 更适合 Transformer）
- 学习率调度：CosineAnnealingWarmRestarts（余弦退火 + 周期重启）
- 损失函数：CrossEntropyLoss + 类别权重（处理类别不平衡）+ 标签平滑
- 梯度裁剪：防止梯度爆炸（max_norm=1.0）
- 早停：验证集 Macro F1 连续 patience 轮不提升时停止
- Checkpoint：每次验证集 F1 创新高时保存模型
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiquant.config import TrainConfig
from aiquant.training.metrics import compute_metrics, ClassificationMetrics
from aiquant.training.checkpoint import save_checkpoint, cleanup_checkpoints

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """记录每个 epoch 的训练/验证指标，用于绘制曲线和分析过拟合。"""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_metrics: list[ClassificationMetrics] = field(default_factory=list)
    best_epoch: int = 0      # 最优 F1 所在的 epoch
    best_metric: float = 0.0  # 最优验证集 Macro F1


class Trainer:
    """封装完整训练流程的 Trainer 类。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Args:
            model:         待训练的 TrendTransformer 模型
            train_loader:  训练集 DataLoader
            val_loader:    验证集 DataLoader
            config:        训练超参数，默认使用 TrainConfig
            class_weights: 各类别的损失权重张量，用于处理类别不平衡
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainConfig()

        self.model.to(self.config.device)

        # AdamW 优化器：相比 Adam 在权重衰减上更正确（不对偏置/归一化层衰减）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # OneCycleLR：线性 warmup（前 10%）+ 余弦退火，无周期重启避免训练震荡
        # 需在每个 batch 后调用 step()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.lr,
            epochs=self.config.max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        # 交叉熵损失 + 类别权重 + 标签平滑
        if class_weights is not None:
            class_weights = class_weights.to(self.config.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing,
        )

        self.history = TrainHistory()

    def _train_epoch(self) -> float:
        """执行一个 epoch 的训练，返回平均训练损失。"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X, y in self.train_loader:
            X = X.to(self.config.device)
            y = y.to(self.config.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()

            # 梯度裁剪：防止梯度爆炸，对 Transformer 尤为重要
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()  # OneCycleLR 在每个 batch 后更新

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, ClassificationMetrics]:
        """在验证集上推理，返回平均损失和分类指标。"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        for X, y in self.val_loader:
            X = X.to(self.config.device)
            y = y.to(self.config.device)

            logits = self.model(X)
            loss = self.criterion(logits, y)

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        metrics = compute_metrics(y_true, y_pred)

        return avg_loss, metrics

    def train(self) -> TrainHistory:
        """执行完整训练流程（含早停和 checkpoint 保存）。

        Returns:
            TrainHistory：包含每个 epoch 的损失和指标历史
        """
        patience_counter = 0
        logger.info(
            "训练开始：最大 %d 轮，设备=%s，参数量=%s",
            self.config.max_epochs,
            self.config.device,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate()

            # 记录本轮指标
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.val_metrics.append(val_metrics)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | train_loss=%.4f, val_loss=%.4f, "
                "acc=%.4f, macro_f1=%.4f, lr=%.6f",
                epoch, self.config.max_epochs,
                train_loss, val_loss,
                val_metrics.accuracy, val_metrics.macro_f1, lr,
            )

            # 验证集 Macro F1 创历史新高：保存 checkpoint 并重置早停计数
            if val_metrics.macro_f1 > self.history.best_metric:
                self.history.best_metric = val_metrics.macro_f1
                self.history.best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics.macro_f1
                )
                cleanup_checkpoints()  # 保留最新 3 个 checkpoint
            else:
                patience_counter += 1

            # 触发早停
            if patience_counter >= self.config.patience:
                logger.info(
                    "早停触发：第 %d 轮停止（最优轮次=%d，最优 F1=%.4f）",
                    epoch, self.history.best_epoch, self.history.best_metric,
                )
                break

        return self.history


def compute_class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """根据训练集标签分布计算类别权重（频率越低权重越高）。

    公式：weight[i] = total / (n_classes * count[i])

    Args:
        labels:    训练集标签数组
        n_classes: 类别总数

    Returns:
        (n_classes,) float32 权重张量
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)  # 避免除以零
    weights = len(labels) / (n_classes * counts)
    return torch.from_numpy(weights)
