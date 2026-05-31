"""训练循环"""

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
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_metrics: list[ClassificationMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_metric: float = 0.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainConfig()

        self.model.to(self.config.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        if class_weights is not None:
            class_weights = class_weights.to(self.config.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing,
        )

        self.history = TrainHistory()

    def _train_epoch(self) -> float:
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

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, ClassificationMetrics]:
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
        patience_counter = 0
        logger.info(
            "Training started: %d epochs, device=%s, params=%s",
            self.config.max_epochs,
            self.config.device,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate()
            self.scheduler.step()

            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.val_metrics.append(val_metrics)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d - train_loss: %.4f, val_loss: %.4f, "
                "acc: %.4f, macro_f1: %.4f, lr: %.6f",
                epoch, self.config.max_epochs,
                train_loss, val_loss,
                val_metrics.accuracy, val_metrics.macro_f1, lr,
            )

            if val_metrics.macro_f1 > self.history.best_metric:
                self.history.best_metric = val_metrics.macro_f1
                self.history.best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics.macro_f1
                )
                cleanup_checkpoints()
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch: %d, best F1: %.4f)",
                    epoch, self.history.best_epoch, self.history.best_metric,
                )
                break

        return self.history


def compute_class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = len(labels) / (n_classes * counts)
    return torch.from_numpy(weights)
