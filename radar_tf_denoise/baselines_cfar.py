"""传统 CFAR-Z / CFAR-AC 基线。

参考实现:
https://github.com/JianpingWang-TUD/InterferenceMitigation_CFAR

本脚本不训练网络，只在带干扰信号 sb 的复数 STFT 上做 1D CFAR 干扰检测与修复：
- CFAR-Z: 检测到的干扰 mask 区域置零。
- CFAR-AC: mask 区域用同频 bin 的非干扰平均幅度替换，相位保持不变。

参考 MATLAB 代码使用 phased.CFARDetector 和 Image Processing Toolbox 的 octagon
膨胀。这里为了在 Python 环境中可复现，使用 CA-CFAR 的 Pfa 阈值公式和一个近似
octagon 结构元素。ARIM v0/v1 的 STFT 时间帧只有 29 帧，所以默认训练/保护单元
按本数据尺寸缩小，不能直接照搬参考代码中的 150/50。
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Sequence, Tuple

# Windows/Conda 下多个科学计算库可能重复加载 OpenMP，这里只做评估画图兜底。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from metrics import METRIC_FIELDNAMES, SUMMARY_FIELDNAMES, compute_metrics_for_sample, summarize_metrics
from stft_utils import (
    TFConfig,
    complex_stft,
    complex_to_channels,
    istft_from_channels,
    magnitude_to_model_space,
    range_spectrum,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def default_paths() -> Dict[str, str]:
    """基于脚本位置生成默认路径，避免写死绝对路径。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "output_dir": os.path.join(script_dir, "outputs_baseline_cfar"),
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="ARIM CFAR-Z / CFAR-AC 传统基线")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="输出目录")
    parser.add_argument("--num_samples", type=int, default=200, help="评估样本数")
    parser.add_argument("--seed", type=int, default=707, help="随机种子，默认和 evaluate.py 保持一致")
    parser.add_argument("--train_cells", type=int, default=4, help="CUT 两侧合计之前每侧的训练单元数")
    parser.add_argument("--guard_cells", type=int, default=1, help="CUT 两侧每侧保护单元数")
    parser.add_argument("--pfa", type=float, default=1e-4, help="CA-CFAR 虚警概率，参考代码使用 1e-4")
    parser.add_argument(
        "--threshold_scale",
        type=float,
        default=None,
        help="手动阈值倍数；不传时使用 CA-CFAR 的 Pfa 阈值公式",
    )
    parser.add_argument("--time_dilate", type=int, default=1, help="mask 时间维膨胀半径")
    parser.add_argument("--freq_dilate", type=int, default=1, help="mask 频率维膨胀半径")
    parser.add_argument("--num_images", type=int, default=6, help="保存可视化图数量")
    return parser.parse_args()


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """读取 ARIM 测试集字段。"""
    data = np.load(path, allow_pickle=True)[()]
    required = ["sb", "sb0", "distances", "amplitudes"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"数据文件缺少字段: {missing}")
    return data


def select_indices(total_count: int, num_samples: int, seed: int) -> List[int]:
    """按固定随机种子选择样本，使 CFAR 和神经网络评估样本一致。"""
    indices = list(range(total_count))
    random.seed(seed)
    random.shuffle(indices)
    if num_samples is not None and num_samples > 0:
        indices = indices[:num_samples]
    return indices


def cfar_alpha(num_train: int, pfa: float) -> float:
    """CA-CFAR 阈值系数。

    对指数分布功率噪声，常用阈值为:
        alpha = N * (Pfa^(-1/N) - 1)
    其中 N 是当前 CUT 可用训练单元数量。边界位置训练单元更少，因此按实际 N 计算。
    """
    if num_train <= 0:
        return float("inf")
    pfa = float(np.clip(pfa, 1e-12, 1.0 - 1e-12))
    return float(num_train * (pfa ** (-1.0 / num_train) - 1.0))


def cfar_mask_time(
    power: np.ndarray,
    train_cells: int = 4,
    guard_cells: int = 1,
    pfa: float = 1e-4,
    threshold_scale: float = None,
) -> np.ndarray:
    """沿时间维做简化 1D CA-CFAR 检测。

    参考仓库先计算 ``abs(sig_TF).^2``，再对每个频率 bin 沿时间帧做 1D CFAR。
    这里输入同样是功率谱 ``power = abs(STFT)^2``。对每个 CUT，取左右训练单元
    估计局部背景功率，超过阈值则判为干扰。
    """
    power = np.asarray(power, dtype=np.float64)
    h, w = power.shape
    mask = np.zeros((h, w), dtype=bool)

    for t in range(w):
        left_start = max(0, t - guard_cells - train_cells)
        left_end = max(0, t - guard_cells)
        right_start = min(w, t + guard_cells + 1)
        right_end = min(w, t + guard_cells + 1 + train_cells)

        train_parts = []
        if left_end > left_start:
            train_parts.append(power[:, left_start:left_end])
        if right_end > right_start:
            train_parts.append(power[:, right_start:right_end])
        if not train_parts:
            continue

        train_values = np.concatenate(train_parts, axis=1)
        background = np.mean(train_values, axis=1)
        if threshold_scale is None:
            alpha = cfar_alpha(train_values.shape[1], pfa)
        else:
            alpha = float(threshold_scale)
        threshold = alpha * (background + 1e-12)
        mask[:, t] = power[:, t] > threshold

    return mask


def octagon_like_structure(freq_radius: int, time_radius: int) -> np.ndarray:
    """生成近似 MATLAB strel('octagon', r) 的二维结构元素。

    ARIM 的 STFT 时间帧很少，严格八边形半径 12 会覆盖过多区域；这里按给定半径
    生成一个矩形切角的近似八边形，用于扩大 CFAR 检出的干扰区域。
    """
    if freq_radius <= 0 and time_radius <= 0:
        return np.ones((1, 1), dtype=bool)

    fy = max(0, int(freq_radius))
    tx = max(0, int(time_radius))
    yy, xx = np.ogrid[-fy : fy + 1, -tx : tx + 1]
    rect = (np.abs(yy) <= fy) & (np.abs(xx) <= tx)
    if fy == 0 or tx == 0:
        return rect.astype(bool)

    y_norm = np.abs(yy) / max(fy, 1)
    x_norm = np.abs(xx) / max(tx, 1)
    # 介于矩形和菱形之间，比完整矩形少膨胀四角，更接近 octagon。
    octagon = rect & ((x_norm + y_norm) <= 1.5)
    return octagon.astype(bool)


def dilate_mask(mask: np.ndarray, freq_radius: int = 1, time_radius: int = 1) -> np.ndarray:
    """对 CFAR mask 做膨胀，覆盖干扰附近的泄漏区域。"""
    if freq_radius <= 0 and time_radius <= 0:
        return mask
    structure = octagon_like_structure(freq_radius, time_radius)
    return binary_dilation(mask, structure=structure)


def apply_cfar_z(zxx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CFAR-Z: 检测到的 mask 区域直接置零。"""
    out = np.array(zxx, copy=True)
    out[mask] = 0.0
    return out


def apply_cfar_ac(zxx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CFAR-AC: 同频非干扰平均幅度替换，保留原相位。

    这一步和参考仓库 ``Util_CFAR_IM/ampCorrect.m`` 的逻辑一致。
    """
    mag = np.abs(zxx)
    phase = np.exp(1j * np.angle(zxx))
    repaired_mag = np.array(mag, copy=True)

    for freq_idx in range(mag.shape[0]):
        valid = ~mask[freq_idx]
        if np.any(valid):
            fill_value = float(np.mean(mag[freq_idx, valid]))
        else:
            fill_value = float(np.mean(mag[freq_idx]))
        repaired_mag[freq_idx, mask[freq_idx]] = fill_value

    return repaired_mag * phase


def stft_to_signal(zxx: np.ndarray, signal_length: int, cfg: TFConfig) -> np.ndarray:
    """把处理后的复数 STFT 通过 ISTFT 还原到时域。"""
    return istft_from_channels(complex_to_channels(zxx), signal_length, cfg)


def model_space_mag(zxx: np.ndarray, cfg: TFConfig) -> np.ndarray:
    """把复数 STFT 幅度转成与 v0/v0.1 可比的模型空间幅度图。"""
    return magnitude_to_model_space(np.abs(zxx), cfg)


def write_metrics(path: str, rows: Sequence[Dict[str, float]]) -> None:
    """写逐样本指标 CSV。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: str, rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """写整体汇总指标 CSV。"""
    summary = summarize_metrics(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary)
    return summary


def save_case_figure(
    path: str,
    sample_id: int,
    noisy_zxx: np.ndarray,
    clean_zxx: np.ndarray,
    mask: np.ndarray,
    z_zxx: np.ndarray,
    ac_zxx: np.ndarray,
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    z_signal: np.ndarray,
    ac_signal: np.ndarray,
) -> None:
    """保存 CFAR-Z / CFAR-AC 可视化图。"""
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    fig.suptitle(f"CFAR-Z / CFAR-AC 基线 | sample_id={sample_id}", fontsize=14)

    images: List[Tuple[np.ndarray, str]] = [
        (np.abs(noisy_zxx), "带干扰 STFT 幅度"),
        (mask.astype(float), "CFAR 干扰 mask"),
        (np.abs(clean_zxx), "干净 STFT 幅度"),
        (np.abs(z_zxx), "CFAR-Z 输出幅度"),
        (np.abs(ac_zxx), "CFAR-AC 输出幅度"),
    ]
    for idx, (image, title) in enumerate(images):
        ax = fig.add_subplot(grid[idx // 3, idx % 3])
        im = ax.imshow(image, aspect="auto", origin="lower", cmap="turbo")
        ax.set_title(title)
        ax.set_xlabel("time frame")
        ax.set_ylabel("frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    eps = 1e-12
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(20 * np.log10(range_spectrum(noisy_signal) + eps), label="带干扰 noisy", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(clean_signal) + eps), label="干净 clean", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(z_signal) + eps), label="CFAR-Z", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(ac_signal) + eps), label="CFAR-AC", linewidth=1.0)
    ax.set_title("距离谱对比")
    ax.set_xlabel("range bin")
    ax.set_ylabel("magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def print_selected_summary(title: str, summary: Dict[str, float]) -> None:
    """终端打印关键汇总指标。"""
    keys = [
        "aggregate_spectrum_mae_improvement_ratio",
        "spectrum_improved_rate",
        "mean_label_pred_peak_keep_ratio",
        "median_label_pred_peak_keep_ratio",
        "mean_label_peak_error_pred",
        "mean_label_peak_error_improvement",
        "label_peak_error_improved_rate",
        "pred_to_clean_noise_floor_ratio",
    ]
    print(title)
    for key in keys:
        print(f"{key}: {summary[key]:.6f}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_dataset(args.test_path)
    cfg = TFConfig()
    indices = select_indices(len(data["sb"]), args.num_samples, args.seed)

    rows_z = []
    rows_ac = []

    for out_idx, sample_id in enumerate(indices):
        noisy_signal = data["sb"][sample_id]
        clean_signal = data["sb0"][sample_id]
        distances = data["distances"][sample_id]
        amplitudes = data["amplitudes"][sample_id]

        _, _, noisy_zxx = complex_stft(noisy_signal, cfg)
        _, _, clean_zxx = complex_stft(clean_signal, cfg)

        # 参考 MATLAB 代码使用 abs(sig_TF).^2 作为 CFAR 输入。
        power = np.abs(noisy_zxx) ** 2
        mask = cfar_mask_time(
            power,
            train_cells=args.train_cells,
            guard_cells=args.guard_cells,
            pfa=args.pfa,
            threshold_scale=args.threshold_scale,
        )
        mask = dilate_mask(mask, freq_radius=args.freq_dilate, time_radius=args.time_dilate)

        z_zxx = apply_cfar_z(noisy_zxx, mask)
        ac_zxx = apply_cfar_ac(noisy_zxx, mask)
        z_signal = stft_to_signal(z_zxx, len(noisy_signal), cfg)
        ac_signal = stft_to_signal(ac_zxx, len(noisy_signal), cfg)

        noisy_tf = model_space_mag(noisy_zxx, cfg)
        clean_tf = model_space_mag(clean_zxx, cfg)
        z_tf = model_space_mag(z_zxx, cfg)
        ac_tf = model_space_mag(ac_zxx, cfg)

        rows_z.append(
            compute_metrics_for_sample(
                sample_id=sample_id,
                noisy_tf=noisy_tf,
                clean_tf=clean_tf,
                pred_tf=z_tf,
                noisy_signal=noisy_signal,
                clean_signal=clean_signal,
                pred_signal=z_signal,
                distances=distances,
                amplitudes=amplitudes,
            )
        )
        rows_ac.append(
            compute_metrics_for_sample(
                sample_id=sample_id,
                noisy_tf=noisy_tf,
                clean_tf=clean_tf,
                pred_tf=ac_tf,
                noisy_signal=noisy_signal,
                clean_signal=clean_signal,
                pred_signal=ac_signal,
                distances=distances,
                amplitudes=amplitudes,
            )
        )

        if out_idx < args.num_images:
            fig_path = os.path.join(args.output_dir, f"cfar_case_{out_idx:03d}_sample_{sample_id:05d}.png")
            save_case_figure(
                fig_path,
                sample_id,
                noisy_zxx,
                clean_zxx,
                mask,
                z_zxx,
                ac_zxx,
                noisy_signal,
                clean_signal,
                z_signal,
                ac_signal,
            )

    metrics_z_path = os.path.join(args.output_dir, "metrics_cfar_z.csv")
    summary_z_path = os.path.join(args.output_dir, "summary_cfar_z.csv")
    metrics_ac_path = os.path.join(args.output_dir, "metrics_cfar_ac.csv")
    summary_ac_path = os.path.join(args.output_dir, "summary_cfar_ac.csv")
    write_metrics(metrics_z_path, rows_z)
    write_metrics(metrics_ac_path, rows_ac)
    summary_z = write_summary(summary_z_path, rows_z)
    summary_ac = write_summary(summary_ac_path, rows_ac)

    print(f"输出目录: {args.output_dir}")
    print(f"CFAR-Z metrics: {metrics_z_path}")
    print(f"CFAR-Z summary: {summary_z_path}")
    print(f"CFAR-AC metrics: {metrics_ac_path}")
    print(f"CFAR-AC summary: {summary_ac_path}")
    print(f"CFAR 参数: train_cells={args.train_cells}, guard_cells={args.guard_cells}, pfa={args.pfa}, "
          f"threshold_scale={args.threshold_scale}, freq_dilate={args.freq_dilate}, time_dilate={args.time_dilate}")
    print_selected_summary("CFAR-Z 关键汇总指标", summary_z)
    print_selected_summary("CFAR-AC 关键汇总指标", summary_ac)


if __name__ == "__main__":
    main()
