"""模型 Checkpoint 的保存、加载与清理

命名规则：model_epoch{epoch:03d}_f1{metric:.4f}.pt
默认保留最近 3 个 checkpoint，自动删除旧文件。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from aiquant.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    path: Path | None = None,
) -> Path:
    """保存模型权重、优化器状态及元信息到 checkpoint 文件。

    Args:
        model:     待保存的模型
        optimizer: 当前优化器（含动量等状态）
        epoch:     当前训练轮次
        metric:    当前最优指标值（验证集 Macro F1）
        path:      保存路径，默认使用命名规则生成

    Returns:
        实际保存的文件路径
    """
    path = path or CHECKPOINT_DIR / f"model_epoch{epoch:03d}_f1{metric:.4f}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        },
        path,
    )
    logger.info("Checkpoint saved: %s (epoch=%d, metric=%.4f)", path.name, epoch, metric)
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """从 checkpoint 文件恢复模型（和优化器）状态。

    Args:
        path:      checkpoint 文件路径
        model:     目标模型（权重将被覆盖）
        optimizer: 可选，同时恢复优化器状态（继续训练时需要）
        device:    加载目标设备

    Returns:
        checkpoint 文件中的原始 dict（含 epoch、metric 等元信息）
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint: %s (epoch=%d, metric=%.4f)",
        path.name, ckpt["epoch"], ckpt["metric"],
    )
    return ckpt


def cleanup_checkpoints(directory: Path | None = None, keep_top: int = 3) -> None:
    """按修改时间删除旧 checkpoint，只保留最新的 keep_top 个。

    Args:
        directory: checkpoint 所在目录，默认使用全局配置
        keep_top:  保留的最新文件数量
    """
    directory = directory or CHECKPOINT_DIR
    if not directory.exists():
        return
    # 按修改时间升序排列，删除最旧的文件
    ckpts = sorted(directory.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep_top]:
        p.unlink()
        logger.debug("Removed old checkpoint: %s", p.name)
