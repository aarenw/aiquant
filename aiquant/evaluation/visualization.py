"""训练与评估结果可视化

提供三类图表：
1. 混淆矩阵热力图：展示各类别预测的准确性和易混淆类别
2. 训练曲线：损失、准确率、Macro F1 随 epoch 的变化
3. 注意力权重热力图：可视化模型在各时间步上的注意力分布
"""

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
    """绘制混淆矩阵热力图。

    Args:
        metrics:   包含混淆矩阵的 ClassificationMetrics
        save_path: 保存路径，None 则不保存（只显示）
    """
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
    ax.set_xlabel("Predicted（预测）")
    ax.set_ylabel("Actual（真实）")
    ax.set_title(f"Confusion Matrix  Acc={metrics.accuracy:.3f}  F1={metrics.macro_f1:.3f}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: TrainHistory,
    save_path: Path | None = None,
) -> None:
    """绘制训练过程曲线（损失、准确率、Macro F1）。

    Args:
        history:   TrainHistory 对象，含每个 epoch 的指标
        save_path: 保存路径，None 则不保存
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history.train_losses) + 1)

    # 子图1：训练/验证损失曲线
    axes[0].plot(epochs, history.train_losses, label="Train")
    axes[0].plot(epochs, history.val_losses, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    # 子图2：验证集准确率
    val_acc = [m.accuracy for m in history.val_metrics]
    axes[1].plot(epochs, val_acc, color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")

    # 子图3：验证集 Macro F1，红虚线标注最优值
    val_f1 = [m.macro_f1 for m in history.val_metrics]
    axes[2].plot(epochs, val_f1, color="orange")
    axes[2].axhline(y=history.best_metric, color="r", linestyle="--", alpha=0.5, label="best")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title(f"Macro F1  best={history.best_metric:.4f} @epoch {history.best_epoch}")
    axes[2].legend()

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
    """可视化某一层某一头的注意力权重矩阵。

    颜色越深表示注意力越集中在该时间步（Key 位置），
    可用于分析模型重点关注哪些历史时间段。

    Args:
        attention_weights: 来自 model.get_attention_weights() 的列表
        layer_idx:  选择第几层 Encoder（从 0 开始）
        head_idx:   选择第几个注意力头（从 0 开始）
        sample_idx: 选择 batch 中第几个样本
        save_path:  保存路径
    """
    w = attention_weights[layer_idx]
    if isinstance(w, np.ndarray):
        attn = w[sample_idx, head_idx]
    else:
        attn = w[sample_idx, head_idx].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attn, cmap="viridis", ax=ax)
    ax.set_xlabel("Key Position（历史时间步）")
    ax.set_ylabel("Query Position（当前时间步）")
    ax.set_title(f"Attention Weights  Layer={layer_idx}  Head={head_idx}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
