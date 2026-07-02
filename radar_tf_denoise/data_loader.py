"""ARIM 第一版时频图数据集。"""

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from stft_utils import TFConfig, stft_magnitude


class RadarTFDataset(Dataset):
    """读取 arim_train.npy / arim_test.npy，并在线计算 STFT 幅度图。

    每个样本返回:
        x: 带干扰时频图，形状 [1, H, W]
        y: 干净时频图，形状 [1, H, W]

    默认 STFT 参数下，1024 点信号输出 [1, 256, 29]。
    """

    def __init__(
        self,
        path: str,
        max_samples: Optional[int] = None,
        tf_config: TFConfig = TFConfig(),
    ) -> None:
        self.path = path
        self.tf_config = tf_config

        data = np.load(path, allow_pickle=True)[()]
        if "sb" not in data or "sb0" not in data:
            raise KeyError("数据文件必须包含 'sb' 和 'sb0' 字段")

        self.sb = data["sb"]
        self.sb0 = data["sb0"]
        if max_samples is not None:
            self.sb = self.sb[:max_samples]
            self.sb0 = self.sb0[:max_samples]

    def __len__(self) -> int:
        return len(self.sb)

    def __getitem__(self, index: int):
        x = stft_magnitude(self.sb[index], self.tf_config)
        y = stft_magnitude(self.sb0[index], self.tf_config)

        x = torch.from_numpy(x).unsqueeze(0).float()
        y = torch.from_numpy(y).unsqueeze(0).float()
        return x, y

    def raw_sample(self, index: int) -> Dict[str, np.ndarray]:
        """评估画图时读取原始复数信号。"""
        return {"sb": self.sb[index], "sb0": self.sb0[index]}
