"""分析 v1.2 底噪偏高的误差来源。

本脚本只做推理和统计，不训练模型、不修改模型文件。

主要判断两件事:
1. CFAR mask 是否漏检高能量干扰，导致非 mask 区域保留 noisy STFT。
2. mask 内 residual 网络是否修复不够干净。
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Sequence

# Windows/Conda 下 torch、scipy、matplotlib 可能重复加载 OpenMP。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines_cfar import cfar_mask_time, dilate_mask
from metrics import label_target_indices
from model_unet import SmallUNet
from stft_utils import TFConfig, channels_to_complex, complex_stft, complex_to_channels, istft_from_channels, range_spectrum

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


BASE_FIELDNAMES = [
    "sample_id",
    "mask_ratio",
    "high_interference_coverage",
    "target_mask_overlap_ratio",
    "mask_in_noisy_mae",
    "mask_out_noisy_mae",
    "mask_in_v11_mae",
    "mask_out_v11_mae",
]


def metric_fieldnames(pred_label: str) -> List[str]:
    """根据待分析模型标签生成逐样本 CSV 列名。"""
    return [
        *BASE_FIELDNAMES,
        f"mask_in_{pred_label}_mae",
        f"mask_out_{pred_label}_mae",
    ]


def summary_fieldnames(pred_label: str) -> List[str]:
    """根据待分析模型标签生成 summary CSV 列名。"""
    return [f"mean_{name}" for name in metric_fieldnames(pred_label) if name != "sample_id"]


def default_paths() -> Dict[str, str]:
    """生成默认路径，便于从脚本目录直接运行。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "v12_checkpoint": os.path.join(script_dir, "checkpoints_v12_full", "best_model.pth"),
        "v11_checkpoint": os.path.join(script_dir, "checkpoints_v11", "best_model.pth"),
        "output_dir": os.path.join(script_dir, "outputs_v12_error_analysis"),
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="分析 v1.2 底噪偏高来源")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--v12_checkpoint", default=paths["v12_checkpoint"], help="v1.2 best_model.pth 路径")
    parser.add_argument("--v11_checkpoint", default=paths["v11_checkpoint"], help="v1.1 best_model.pth 路径")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="输出目录")
    parser.add_argument("--pred_label", default="v12", help="待分析模型标签，例如 v12 或 v13")
    parser.add_argument("--num_samples", type=int, default=200, help="分析样本数")
    parser.add_argument("--num_images", type=int, default=6, help="保存可视化图数量")
    parser.add_argument("--seed", type=int, default=707, help="随机种子，默认和 evaluate.py 一致")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")

    # v1.2 当前固定 CFAR 参数。
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--train_cells", type=int, default=4)
    parser.add_argument("--guard_cells", type=int, default=1)
    parser.add_argument("--dilation_iter", type=int, default=1)
    parser.add_argument("--target_freq_radius", type=int, default=2, help="目标频率 bin 邻域半径")
    return parser.parse_args()


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """读取 ARIM 测试数据。"""
    data = np.load(path, allow_pickle=True)[()]
    required = ["sb", "sb0", "distances", "amplitudes"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"数据文件缺少字段: {missing}")
    return data


def select_indices(total_count: int, num_samples: int, seed: int) -> List[int]:
    """使用固定随机种子选择样本，保证和之前评估可比。"""
    indices = list(range(total_count))
    random.seed(seed)
    random.shuffle(indices)
    if num_samples is not None and num_samples > 0:
        indices = indices[:num_samples]
    return indices


def load_model(checkpoint_path: str, in_channels: int, out_channels: int, device: torch.device) -> tuple[SmallUNet, TFConfig]:
    """加载指定通道数的 U-Net。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = TFConfig(**checkpoint.get("tf_config", {}))
    model = SmallUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def build_v12_mask(
    noisy_zxx: np.ndarray,
    pfa: float,
    train_cells: int,
    guard_cells: int,
    dilation_iter: int,
) -> np.ndarray:
    """使用 v1.2 当前参数生成 CFAR mask。"""
    mask = cfar_mask_time(
        np.abs(noisy_zxx) ** 2,
        train_cells=train_cells,
        guard_cells=guard_cells,
        pfa=pfa,
        threshold_scale=None,
    )
    for _ in range(dilation_iter):
        mask = dilate_mask(mask, freq_radius=1, time_radius=1)
    return mask.astype(bool)


def predict_v11(model: SmallUNet, noisy_zxx: np.ndarray, device: torch.device) -> np.ndarray:
    """v1.1: 输入 noisy 复数 STFT，输出预测复数 STFT。"""
    x = torch.from_numpy(complex_to_channels(noisy_zxx)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_channels = model(x).squeeze(0).cpu().numpy()
    return channels_to_complex(pred_channels)


def predict_v12(model: SmallUNet, noisy_zxx: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    """v1.2: 输入 noisy 实部/虚部 + mask，输出 noisy + mask * residual。"""
    noisy_channels = complex_to_channels(noisy_zxx)
    mask_channel = mask.astype(np.float32)[None, :, :]
    x = np.concatenate([noisy_channels, mask_channel], axis=0)
    x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)
    noisy_tensor = torch.from_numpy(noisy_channels).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_channel).unsqueeze(0).float().to(device)
    with torch.no_grad():
        residual = model(x_tensor)
        pred_channels = (noisy_tensor + mask_tensor * residual).squeeze(0).cpu().numpy()
    return channels_to_complex(pred_channels)


def masked_mae(pred_mag: np.ndarray, clean_mag: np.ndarray, mask: np.ndarray) -> float:
    """计算指定 mask 区域内的幅度 MAE；区域为空时返回 NaN。"""
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred_mag[mask] - clean_mag[mask])))


def high_interference_region(noisy_zxx: np.ndarray, clean_zxx: np.ndarray, top_ratio: float = 0.1) -> np.ndarray:
    """取 abs(noisy)-abs(clean) 最大的前 10% 时频点作为高能量干扰区域。"""
    diff = np.maximum(np.abs(noisy_zxx) - np.abs(clean_zxx), 0.0)
    flat = diff.reshape(-1)
    count = max(1, int(np.ceil(flat.size * top_ratio)))
    top_indices = np.argpartition(flat, -count)[-count:]
    region = np.zeros(flat.size, dtype=bool)
    region[top_indices] = True
    return region.reshape(diff.shape)


def target_region_from_labels(
    distances: Optional[np.ndarray],
    amplitudes: Optional[np.ndarray],
    label_length_default: int,
    nfft: int,
    time_frames: int,
    freq_radius: int = 2,
) -> np.ndarray:
    """把 2048 维目标标签粗略映射到 STFT 频率 bin 邻域。

    ARIM 标签通常是 2048 点距离谱 bin；STFT 默认 nfft=256。
    这里用 target_idx * nfft / label_len 映射到 STFT bin，并取频率邻域 ±freq_radius。
    """
    target_indices, label_length = label_target_indices(distances, amplitudes)
    label_length = max(label_length, label_length_default)
    region = np.zeros((nfft, time_frames), dtype=bool)
    if len(target_indices) == 0:
        return region

    mapped_bins = np.rint(target_indices.astype(np.float64) * nfft / label_length).astype(np.int64) % nfft
    for freq_bin in mapped_bins:
        for offset in range(-freq_radius, freq_radius + 1):
            region[(freq_bin + offset) % nfft, :] = True
    return region


def save_metrics(path: str, rows: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    """保存逐样本误差来源指标。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> Dict[str, float]:
    """对逐样本指标求均值，自动忽略 NaN。"""
    summary = {}
    for key in fieldnames:
        if key == "sample_id":
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return summary


def save_summary(path: str, summary: Dict[str, float], fieldnames: Sequence[str]) -> None:
    """保存整体 summary。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def save_case_figure(
    path: str,
    sample_id: int,
    noisy_zxx: np.ndarray,
    clean_zxx: np.ndarray,
    v11_zxx: np.ndarray,
    pred_zxx: np.ndarray,
    mask: np.ndarray,
    high_region: np.ndarray,
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    cfg: TFConfig,
    pred_label: str,
) -> None:
    """保存单样本可视化图。"""
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    fig.suptitle(f"{pred_label} error source | sample_id={sample_id}", fontsize=14)

    images = [
        (np.abs(noisy_zxx), "noisy 幅度时频图"),
        (np.abs(clean_zxx), "clean 幅度时频图"),
        (mask.astype(float), "CFAR mask"),
        (high_region.astype(float), "high interference region"),
        (np.abs(v11_zxx), "v1.1 输出"),
        (np.abs(pred_zxx), f"{pred_label} 输出"),
    ]
    for idx, (image, title) in enumerate(images):
        ax = fig.add_subplot(grid[idx // 4, idx % 4])
        im = ax.imshow(image, aspect="auto", origin="lower", cmap="turbo")
        ax.set_title(title)
        ax.set_xlabel("time frame")
        ax.set_ylabel("frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 距离谱对比使用和前面评估一致的 2048 点 FFT 幅度。
    v11_time = istft_from_channels(complex_to_channels(v11_zxx), len(noisy_signal), cfg)
    pred_time = istft_from_channels(complex_to_channels(pred_zxx), len(noisy_signal), cfg)

    eps = 1e-12
    ax = fig.add_subplot(grid[1, 2:4])
    ax.plot(20 * np.log10(range_spectrum(noisy_signal) + eps), label="noisy", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(clean_signal) + eps), label="clean", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(v11_time) + eps), label="v1.1", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(pred_time) + eps), label=pred_label, linewidth=1.0)
    ax.set_title("距离谱对比")
    ax.set_xlabel("range bin")
    ax.set_ylabel("magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_dataset(args.test_path)
    v12_model, cfg = load_model(args.v12_checkpoint, in_channels=3, out_channels=2, device=device)
    v11_model, _ = load_model(args.v11_checkpoint, in_channels=2, out_channels=2, device=device)
    indices = select_indices(len(data["sb"]), args.num_samples, args.seed)
    fieldnames = metric_fieldnames(args.pred_label)
    summary_names = summary_fieldnames(args.pred_label)

    rows = []
    for out_idx, sample_id in enumerate(indices):
        noisy_signal = data["sb"][sample_id]
        clean_signal = data["sb0"][sample_id]
        _, _, noisy_zxx = complex_stft(noisy_signal, cfg)
        _, _, clean_zxx = complex_stft(clean_signal, cfg)

        mask = build_v12_mask(
            noisy_zxx,
            pfa=args.pfa,
            train_cells=args.train_cells,
            guard_cells=args.guard_cells,
            dilation_iter=args.dilation_iter,
        )
        high_region = high_interference_region(noisy_zxx, clean_zxx, top_ratio=0.1)
        target_region = target_region_from_labels(
            data["distances"][sample_id],
            data["amplitudes"][sample_id],
            label_length_default=2048,
            nfft=cfg.nfft,
            time_frames=noisy_zxx.shape[1],
            freq_radius=args.target_freq_radius,
        )

        v11_zxx = predict_v11(v11_model, noisy_zxx, device)
        pred_zxx = predict_v12(v12_model, noisy_zxx, mask, device)

        noisy_mag = np.abs(noisy_zxx)
        clean_mag = np.abs(clean_zxx)
        v11_mag = np.abs(v11_zxx)
        pred_mag = np.abs(pred_zxx)
        mask_out = ~mask

        row = {
            "sample_id": int(sample_id),
            "mask_ratio": float(np.mean(mask)),
            "high_interference_coverage": float(np.mean(mask[high_region])) if np.any(high_region) else float("nan"),
            "target_mask_overlap_ratio": float(np.mean(mask[target_region])) if np.any(target_region) else float("nan"),
            "mask_in_noisy_mae": masked_mae(noisy_mag, clean_mag, mask),
            "mask_out_noisy_mae": masked_mae(noisy_mag, clean_mag, mask_out),
            "mask_in_v11_mae": masked_mae(v11_mag, clean_mag, mask),
            "mask_out_v11_mae": masked_mae(v11_mag, clean_mag, mask_out),
            f"mask_in_{args.pred_label}_mae": masked_mae(pred_mag, clean_mag, mask),
            f"mask_out_{args.pred_label}_mae": masked_mae(pred_mag, clean_mag, mask_out),
        }
        rows.append(row)

        if out_idx < args.num_images:
            fig_path = os.path.join(args.output_dir, f"error_source_case_{out_idx:03d}_sample_{sample_id:05d}.png")
            save_case_figure(
                fig_path,
                sample_id,
                noisy_zxx,
                clean_zxx,
                v11_zxx,
                pred_zxx,
                mask,
                high_region,
                noisy_signal,
                clean_signal,
                cfg,
                args.pred_label,
            )

    metrics_path = os.path.join(args.output_dir, "error_source_metrics.csv")
    summary_path = os.path.join(args.output_dir, "error_source_summary.csv")
    save_metrics(metrics_path, rows, fieldnames)
    summary = summarize_rows(rows, fieldnames)
    save_summary(summary_path, summary, summary_names)

    print(f"输出目录: {args.output_dir}")
    print(f"逐样本指标: {metrics_path}")
    print(f"汇总指标: {summary_path}")
    print(
        "CFAR 参数: "
        f"pfa={args.pfa}, dilation_iter={args.dilation_iter}, "
        f"train_cells={args.train_cells}, guard_cells={args.guard_cells}"
    )
    print(f"{args.pred_label} 误差来源 summary:")
    for key in [
        "mean_mask_ratio",
        "mean_high_interference_coverage",
        "mean_target_mask_overlap_ratio",
        f"mean_mask_in_{args.pred_label}_mae",
        f"mean_mask_out_{args.pred_label}_mae",
        "mean_mask_in_v11_mae",
        "mean_mask_out_v11_mae",
    ]:
        print(f"{key}: {summary[key]:.6f}")


if __name__ == "__main__":
    main()
