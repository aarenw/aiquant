"""PyTorch Dataset 和 DataLoader 封装

将预处理后的 NumPy 数组包装为 PyTorch Dataset，
并提供创建 DataLoader 的工厂函数。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrendDataset(Dataset):
    """股票趋势数据集。

    将 (特征序列, 标签) 的 NumPy 数组封装为 PyTorch Dataset，
    支持 DataLoader 的自动批处理和多进程加载。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, seq_len, n_features) float32 特征矩阵
            y: (n_samples,) int64 趋势标签（0=由涨转跌, 1=维持趋势, 2=由跌转涨）
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """创建训练集和验证集的 DataLoader。

    Args:
        X_train, y_train: 训练集特征和标签
        X_val, y_val:     验证集特征和标签
        batch_size:       批大小
        num_workers:      数据加载进程数（macOS/MPS 建议保持 0）

    Returns:
        (train_loader, val_loader)
    """
    train_ds = TrendDataset(X_train, y_train)
    val_ds = TrendDataset(X_val, y_val)

    # 训练集：打乱顺序以打破股票间的时间相关性
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # 验证集：不打乱顺序，保持评估确定性
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
