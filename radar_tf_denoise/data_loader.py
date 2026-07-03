"""ARIM 第一版时频图数据集。"""

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from baselines_cfar import cfar_mask_time, dilate_mask
from stft_utils import TFConfig, complex_stft, complex_to_channels, stft_complex_channels, stft_magnitude


def build_best_cfar_mask(zxx: np.ndarray) -> np.ndarray:
    """使用 soft hybrid 搜索得到的最佳 CFAR 参数生成 mask。

    v1.2 的设计目标是只让网络修复干扰区域，因此这里固定使用当前 best hybrid 参数:
    pfa=1e-3, dilation_iter=1, train_cells=4, guard_cells=1。
    """
    mask = cfar_mask_time(
        np.abs(zxx) ** 2,
        train_cells=4,
        guard_cells=1,
        pfa=1e-3,
        threshold_scale=None,
    )
    return dilate_mask(mask, freq_radius=1, time_radius=1).astype(np.float32)


class RadarTFDataset(Dataset):
    """读取 arim_train.npy / arim_test.npy，并在线计算 STFT 特征。

    mode="magnitude" 时，每个样本返回:
        x: 带干扰时频图，形状 [1, H, W]
        y: 干净时频图，形状 [1, H, W]

    mode="complex" 时，每个样本返回:
        x: STFT(sb) 实部/虚部，形状 [2, H, W]
        y: STFT(sb0) 实部/虚部，形状 [2, H, W]

    mode="complex_mask_residual" 或 mode="complex_mask_clean" 时，每个样本返回:
        x: STFT(sb) 实部/虚部 + CFAR mask，形状 [3, H, W]
        y: STFT(sb0) 实部/虚部，形状 [2, H, W]
        noisy_channels: STFT(sb) 实部/虚部，形状 [2, H, W]
        mask: CFAR mask，形状 [1, H, W]

    默认 STFT 参数下，1024 点信号输出 [1, 256, 29]。
    """

    def __init__(
        self,
        path: str,
        max_samples: Optional[int] = None,
        tf_config: TFConfig = TFConfig(),
        mode: str = "magnitude",
    ) -> None:
        self.path = path
        self.tf_config = tf_config
        if mode not in {"magnitude", "complex", "complex_mask_residual", "complex_mask_clean"}:
            raise ValueError("mode 必须是 'magnitude'、'complex'、'complex_mask_residual' 或 'complex_mask_clean'")
        self.mode = mode

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
        if self.mode == "magnitude":
            x = stft_magnitude(self.sb[index], self.tf_config)
            y = stft_magnitude(self.sb0[index], self.tf_config)

            x = torch.from_numpy(x).unsqueeze(0).float()
            y = torch.from_numpy(y).unsqueeze(0).float()
            return x, y

        if self.mode in {"complex_mask_residual", "complex_mask_clean"}:
            _, _, noisy_zxx = complex_stft(self.sb[index], self.tf_config)
            _, _, clean_zxx = complex_stft(self.sb0[index], self.tf_config)
            noisy_channels = complex_to_channels(noisy_zxx)
            clean_channels = complex_to_channels(clean_zxx)
            mask = build_best_cfar_mask(noisy_zxx)[None, :, :]

            # input 的第 3 个通道是 CFAR mask；v1.2 预测 residual，v1.4 预测 clean STFT。
            x = np.concatenate([noisy_channels, mask], axis=0)
            return (
                torch.from_numpy(x).float(),
                torch.from_numpy(clean_channels).float(),
                torch.from_numpy(noisy_channels).float(),
                torch.from_numpy(mask).float(),
            )

        x = stft_complex_channels(self.sb[index], self.tf_config)
        y = stft_complex_channels(self.sb0[index], self.tf_config)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y

    def raw_sample(self, index: int) -> Dict[str, np.ndarray]:
        """评估画图时读取原始复数信号。"""
        return {"sb": self.sb[index], "sb0": self.sb0[index]}
