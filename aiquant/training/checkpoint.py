"""模型 checkpoint 保存/加载"""

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
    directory = directory or CHECKPOINT_DIR
    if not directory.exists():
        return
    ckpts = sorted(directory.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep_top]:
        p.unlink()
        logger.debug("Removed old checkpoint: %s", p.name)
