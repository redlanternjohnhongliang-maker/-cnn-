"""STFT 工具函数。

默认参数面向 ARIM 第一版的 1024 点复数 beat signal：
- nperseg = 128
- noverlap = 96
- nfft = 256
- return_onesided = False

因此默认二维时频图尺寸为：
- H = 256 个频率 bin
- W = floor((1024 - 128) / (128 - 96)) + 1 = 29 个时间帧
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class TFConfig:
    fs: float = 40e6
    nperseg: int = 128
    noverlap: int = 96
    nfft: int = 256
    window: str = "hamming"
    log_compress: bool = True
    log_scale: float = 1.0
    normalize: bool = True
    norm_value: float = 8.0

    @property
    def hop_length(self) -> int:
        return self.nperseg - self.noverlap


def expected_tf_shape(signal_length: int = 1024, cfg: TFConfig = TFConfig()) -> Tuple[int, int]:
    """返回默认 STFT 幅度图尺寸 H, W。"""
    h = cfg.nfft
    w = (signal_length - cfg.nperseg) // cfg.hop_length + 1
    return h, w


def complex_stft(x: np.ndarray, cfg: TFConfig = TFConfig()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算复数 STFT。

    输入:
        x: 一维复数雷达 beat signal，形状 [1024]

    输出:
        f: 频率轴，形状 [H]
        t: 时间轴，形状 [W]
        zxx: 复数 STFT，形状 [H, W]
    """
    x = np.asarray(x).reshape(-1)
    f, t, zxx = signal.stft(
        x,
        fs=cfg.fs,
        window=cfg.window,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        nfft=cfg.nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    return f, t, zxx


def magnitude_to_model_space(mag: np.ndarray, cfg: TFConfig = TFConfig()) -> np.ndarray:
    """把原始 STFT 幅度转成模型训练空间。"""
    out = np.asarray(mag, dtype=np.float32)
    if cfg.log_compress:
        out = np.log1p(cfg.log_scale * out)
    if cfg.normalize:
        out = out / cfg.norm_value
    return out.astype(np.float32)


def model_space_to_magnitude(value: np.ndarray, cfg: TFConfig = TFConfig()) -> np.ndarray:
    """把模型输出近似还原到原始 STFT 幅度空间，评估画图时使用。"""
    out = np.asarray(value, dtype=np.float32)
    if cfg.normalize:
        out = out * cfg.norm_value
    if cfg.log_compress:
        out = np.expm1(out) / cfg.log_scale
    return np.maximum(out, 0.0).astype(np.float32)


def stft_magnitude(x: np.ndarray, cfg: TFConfig = TFConfig()) -> np.ndarray:
    """计算二维 STFT 幅度图。

    输入:
        x: 一维复数信号，形状 [1024]

    输出:
        mag: 二维幅度图，形状 [H, W]，默认 [256, 29]
    """
    _, _, zxx = complex_stft(x, cfg)
    mag = np.abs(zxx).astype(np.float32)
    return magnitude_to_model_space(mag, cfg)


def complex_to_channels(zxx: np.ndarray) -> np.ndarray:
    """把复数 STFT 转成两通道实数张量。

    输入:
        zxx: 复数 STFT，形状 [H, W]

    输出:
        channels: 两通道数组，形状 [2, H, W]
            channels[0] 是实部
            channels[1] 是虚部
    """
    zxx = np.asarray(zxx)
    return np.stack([zxx.real, zxx.imag], axis=0).astype(np.float32)


def channels_to_complex(channels: np.ndarray) -> np.ndarray:
    """把 [2, H, W] 实部/虚部通道还原成复数 STFT。"""
    channels = np.asarray(channels)
    if channels.shape[0] != 2:
        raise ValueError(f"复数通道输入必须是 [2,H,W]，当前形状: {channels.shape}")
    return channels[0].astype(np.float64) + 1j * channels[1].astype(np.float64)


def stft_complex_channels(x: np.ndarray, cfg: TFConfig = TFConfig()) -> np.ndarray:
    """计算复数 STFT，并输出 [2,H,W] 的实部/虚部通道。"""
    _, _, zxx = complex_stft(x, cfg)
    return complex_to_channels(zxx)


def istft_from_magnitude_and_phase(
    magnitude: np.ndarray,
    phase_source: np.ndarray,
    signal_length: int,
    cfg: TFConfig = TFConfig(),
) -> np.ndarray:
    """用预测幅度和带干扰输入相位近似重建时域信号。

    第一版模型只输出幅度图，没有预测相位。为了画距离谱，这里借用输入
    sb 的 STFT 相位做一个近似重建，后续可替换为相位恢复或复数网络。
    """
    raw_mag = model_space_to_magnitude(magnitude, cfg)
    phase = np.exp(1j * np.angle(phase_source))
    zxx = raw_mag * phase
    _, x_rec = signal.istft(
        zxx,
        fs=cfg.fs,
        window=cfg.window,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        nfft=cfg.nfft,
        input_onesided=False,
        boundary=False,
    )
    x_rec = np.asarray(x_rec)
    if len(x_rec) < signal_length:
        x_rec = np.pad(x_rec, (0, signal_length - len(x_rec)))
    return x_rec[:signal_length]


def istft_from_channels(
    channels: np.ndarray,
    signal_length: int,
    cfg: TFConfig = TFConfig(),
) -> np.ndarray:
    """用预测的实部/虚部 STFT 做 ISTFT，恢复时域复数信号。

    STFT 和 ISTFT 使用同一套 window、nperseg、noverlap、nfft、fs 参数。
    """
    zxx = channels_to_complex(channels)
    _, x_rec = signal.istft(
        zxx,
        fs=cfg.fs,
        window=cfg.window,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        nfft=cfg.nfft,
        input_onesided=False,
        boundary=False,
    )
    x_rec = np.asarray(x_rec)
    if len(x_rec) < signal_length:
        x_rec = np.pad(x_rec, (0, signal_length - len(x_rec)))
    return x_rec[:signal_length]


def range_spectrum(x: np.ndarray, nfft: int = 2048) -> np.ndarray:
    """计算一维距离谱幅度，返回正频率半边。"""
    spec = np.abs(np.fft.fft(np.asarray(x).reshape(-1), nfft))
    return spec[: nfft // 2]
