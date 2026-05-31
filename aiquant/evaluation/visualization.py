"""训练与评估可视化"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from aiquant.training.metrics import ClassificationMetrics
from aiquant.training.trainer import TrainHistory
from aiquant.data.preprocessing import TrendLabel


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TrendLabel.NAMES,
        yticklabels=TrendLabel.NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Acc={metrics.accuracy:.3f}, F1={metrics.macro_f1:.3f})")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: TrainHistory,
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history.train_losses) + 1)

    # Loss curves
    axes[0].plot(epochs, history.train_losses, label="Train")
    axes[0].plot(epochs, history.val_losses, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    val_acc = [m.accuracy for m in history.val_metrics]
    axes[1].plot(epochs, val_acc)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")

    # F1
    val_f1 = [m.macro_f1 for m in history.val_metrics]
    axes[2].plot(epochs, val_f1)
    axes[2].axhline(y=history.best_metric, color="r", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title(f"Macro F1 (best={history.best_metric:.4f} @epoch {history.best_epoch})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_attention_weights(
    attention_weights: list[np.ndarray],
    layer_idx: int = 0,
    head_idx: int = 0,
    sample_idx: int = 0,
    save_path: Path | None = None,
) -> None:
    """可视化某一层某一头的注意力权重矩阵。"""
    w = attention_weights[layer_idx]
    if isinstance(w, np.ndarray):
        attn = w[sample_idx, head_idx]
    else:
        attn = w[sample_idx, head_idx].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attn, cmap="viridis", ax=ax)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
