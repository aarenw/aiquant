"""随机种子固定，确保实验结果可复现

同时设置 Python random、NumPy 和 PyTorch 的随机种子。
对于 CUDA 设备额外开启确定性模式（会略微降低运行速度）。
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现。

    Args:
        seed: 随机种子，默认 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # 关闭非确定性算法
        torch.backends.cudnn.benchmark = False       # 关闭自动寻优
