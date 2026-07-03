"""v1.1 + CFAR mask 无训练后处理实验。

目的:
验证 v1.1 在部分样本上变差，是否来自网络改动了非干扰区域。

做法:
1. 对带干扰信号计算复数 STFT，得到 Z_noisy。
2. 用 v1.1 预测复数 STFT，得到 Z_pred。
3. 用 CFAR 在 abs(Z_noisy)^2 上检测干扰 mask。
4. 只在 mask 区域使用 v1.1 输出，非 mask 区域保留原始 Z_noisy:
       Z_hybrid = M * Z_pred + (1 - M) * Z_noisy
5. 对 Z_hybrid 做 ISTFT，并复用现有 metrics.py 计算指标。
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Optional

# Windows/Conda 下 torch、scipy、matplotlib 可能重复加载 OpenMP。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines_cfar import cfar_mask_time, dilate_mask
from metrics import METRIC_FIELDNAMES, SUMMARY_FIELDNAMES, compute_metrics_for_sample, summarize_metrics
from model_unet import SmallUNet
from stft_utils import (
    TFConfig,
    channels_to_complex,
    complex_stft,
    complex_to_channels,
    istft_from_channels,
    range_spectrum,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


KEY_METRICS = [
    "aggregate_spectrum_mae_improvement_ratio",
    "spectrum_improved_rate",
    "mean_label_pred_peak_keep_ratio",
    "mean_label_peak_error_pred",
    "label_peak_error_improved_rate",
    "pred_to_clean_noise_floor_ratio",
]


def default_paths() -> Dict[str, str]:
    """生成默认输入输出路径，便于直接从脚本目录运行。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "checkpoint": os.path.join(script_dir, "checkpoints_v11", "best_model.pth"),
        "output_dir": os.path.join(script_dir, "outputs_hybrid_v11_cfar"),
        "v11_summary": os.path.join(script_dir, "outputs_v11", "summary_metrics.csv"),
        "cfar_ac_summary": os.path.join(script_dir, "outputs_baseline_cfar", "summary_cfar_ac.csv"),
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="v1.1 + CFAR mask 无训练后处理实验")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--checkpoint", default=paths["checkpoint"], help="v1.1 best_model.pth 路径")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="输出目录")
    parser.add_argument("--num_samples", type=int, default=200, help="评估样本数")
    parser.add_argument("--seed", type=int, default=707, help="随机种子，默认和 evaluate.py 一致")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--num_images", type=int, default=6, help="保存可视化图数量")

    # CFAR 参数沿用 baselines_cfar.py 的默认值。
    parser.add_argument("--train_cells", type=int, default=4, help="CFAR 每侧训练单元数")
    parser.add_argument("--guard_cells", type=int, default=1, help="CFAR 每侧保护单元数")
    parser.add_argument("--pfa", type=float, default=1e-4, help="CFAR 虚警概率")
    parser.add_argument("--threshold_scale", type=float, default=None, help="手动阈值倍数；默认用 Pfa 公式")
    parser.add_argument("--time_dilate", type=int, default=1, help="mask 时间维膨胀半径")
    parser.add_argument("--freq_dilate", type=int, default=1, help="mask 频率维膨胀半径")

    parser.add_argument("--v11_summary", default=paths["v11_summary"], help="v1.1 summary_metrics.csv 路径")
    parser.add_argument("--cfar_ac_summary", default=paths["cfar_ac_summary"], help="CFAR-AC summary CSV 路径")
    return parser.parse_args()


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """读取测试数据和标签。"""
    data = np.load(path, allow_pickle=True)[()]
    required = ["sb", "sb0", "distances", "amplitudes"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"数据文件缺少字段: {missing}")
    return data


def select_indices(total_count: int, num_samples: int, seed: int) -> List[int]:
    """使用固定随机种子抽样，保证和 v1.1/CFAR 基线可比。"""
    indices = list(range(total_count))
    random.seed(seed)
    random.shuffle(indices)
    if num_samples is not None and num_samples > 0:
        indices = indices[:num_samples]
    return indices


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SmallUNet, TFConfig]:
    """加载 v1.1 复数两通道 U-Net。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = TFConfig(**checkpoint.get("tf_config", {}))
    model = SmallUNet(in_channels=2, out_channels=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def predict_v11_zxx(model: SmallUNet, noisy_zxx: np.ndarray, device: torch.device) -> np.ndarray:
    """用 v1.1 预测复数 STFT。"""
    x = torch.from_numpy(complex_to_channels(noisy_zxx)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_channels = model(x).squeeze(0).cpu().numpy()
    return channels_to_complex(pred_channels)


def make_cfar_mask(
    noisy_zxx: np.ndarray,
    train_cells: int,
    guard_cells: int,
    pfa: float,
    threshold_scale: Optional[float],
    freq_dilate: int,
    time_dilate: int,
) -> np.ndarray:
    """在 abs(Z_noisy)^2 上生成并膨胀 CFAR 干扰 mask。"""
    power = np.abs(noisy_zxx) ** 2
    mask = cfar_mask_time(
        power,
        train_cells=train_cells,
        guard_cells=guard_cells,
        pfa=pfa,
        threshold_scale=threshold_scale,
    )
    return dilate_mask(mask, freq_radius=freq_dilate, time_radius=time_dilate)


def write_metrics_csv(path: str, rows: List[Dict[str, float]]) -> None:
    """保存逐样本指标。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str, rows: List[Dict[str, float]]) -> Dict[str, float]:
    """保存整体汇总指标。"""
    summary = summarize_metrics(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary)
    return summary


def read_summary(path: str) -> Optional[Dict[str, float]]:
    """读取已有 summary CSV；缺失时返回 None。"""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return {key: float(value) for key, value in row.items()}


def save_comparison_csv(path: str, summaries: Dict[str, Dict[str, float]]) -> None:
    """保存 hybrid、v1.1、CFAR-AC 的关键指标对比。"""
    fieldnames = ["method"] + KEY_METRICS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, summary in summaries.items():
            row = {"method": method}
            row.update({key: summary[key] for key in KEY_METRICS})
            writer.writerow(row)


def save_case_figure(
    path: str,
    sample_id: int,
    noisy_zxx: np.ndarray,
    clean_zxx: np.ndarray,
    pred_zxx: np.ndarray,
    hybrid_zxx: np.ndarray,
    mask: np.ndarray,
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    pred_signal: np.ndarray,
    hybrid_signal: np.ndarray,
) -> None:
    """保存单样本可视化图。"""
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    fig.suptitle(f"v1.1 + CFAR mask 后处理 | sample_id={sample_id}", fontsize=14)

    images = [
        (np.abs(noisy_zxx), "带干扰时频图"),
        (mask.astype(float), "CFAR mask"),
        (np.abs(pred_zxx), "v1.1 输出时频图"),
        (np.abs(hybrid_zxx), "hybrid 输出时频图"),
        (np.abs(clean_zxx), "干净时频图"),
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
    ax.plot(20 * np.log10(range_spectrum(noisy_signal) + eps), label="noisy", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(clean_signal) + eps), label="clean", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(pred_signal) + eps), label="v1.1", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(hybrid_signal) + eps), label="hybrid", linewidth=1.0)
    ax.set_title("距离谱对比")
    ax.set_xlabel("range bin")
    ax.set_ylabel("magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def print_key_summary(title: str, summary: Dict[str, float]) -> None:
    """打印用户关心的关键指标。"""
    print(title)
    for key in KEY_METRICS:
        print(f"{key}: {summary[key]:.6f}")


def print_comparison_table(summaries: Dict[str, Dict[str, float]]) -> None:
    """在终端打印方法对比表。"""
    print("方法对比")
    print("method," + ",".join(KEY_METRICS))
    for method, summary in summaries.items():
        values = ",".join(f"{summary[key]:.6f}" for key in KEY_METRICS)
        print(f"{method},{values}")


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
    model, cfg = load_model(args.checkpoint, device)
    indices = select_indices(len(data["sb"]), args.num_samples, args.seed)

    rows = []
    mask_ratios = []

    for out_idx, sample_id in enumerate(indices):
        noisy_signal = data["sb"][sample_id]
        clean_signal = data["sb0"][sample_id]
        distances = data["distances"][sample_id]
        amplitudes = data["amplitudes"][sample_id]

        _, _, noisy_zxx = complex_stft(noisy_signal, cfg)
        _, _, clean_zxx = complex_stft(clean_signal, cfg)
        pred_zxx = predict_v11_zxx(model, noisy_zxx, device)

        mask = make_cfar_mask(
            noisy_zxx,
            train_cells=args.train_cells,
            guard_cells=args.guard_cells,
            pfa=args.pfa,
            threshold_scale=args.threshold_scale,
            freq_dilate=args.freq_dilate,
            time_dilate=args.time_dilate,
        )
        mask_ratios.append(float(np.mean(mask)))

        # 关键后处理：只在 CFAR mask 区域采用网络预测，非干扰区域保留原始带干扰 STFT。
        hybrid_zxx = np.where(mask, pred_zxx, noisy_zxx)

        pred_signal = istft_from_channels(complex_to_channels(pred_zxx), len(noisy_signal), cfg)
        hybrid_signal = istft_from_channels(complex_to_channels(hybrid_zxx), len(noisy_signal), cfg)

        rows.append(
            compute_metrics_for_sample(
                sample_id=sample_id,
                noisy_tf=complex_to_channels(noisy_zxx),
                clean_tf=complex_to_channels(clean_zxx),
                pred_tf=complex_to_channels(hybrid_zxx),
                noisy_signal=noisy_signal,
                clean_signal=clean_signal,
                pred_signal=hybrid_signal,
                distances=distances,
                amplitudes=amplitudes,
            )
        )

        if out_idx < args.num_images:
            fig_path = os.path.join(args.output_dir, f"hybrid_case_{out_idx:03d}_sample_{sample_id:05d}.png")
            save_case_figure(
                fig_path,
                sample_id,
                noisy_zxx,
                clean_zxx,
                pred_zxx,
                hybrid_zxx,
                mask,
                noisy_signal,
                clean_signal,
                pred_signal,
                hybrid_signal,
            )

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    write_metrics_csv(metrics_path, rows)
    hybrid_summary = write_summary_csv(summary_path, rows)

    summaries = {"hybrid_v11_cfar": hybrid_summary}
    v11_summary = read_summary(args.v11_summary)
    cfar_ac_summary = read_summary(args.cfar_ac_summary)
    if v11_summary is not None:
        summaries["v1.1"] = v11_summary
    else:
        print(f"未找到 v1.1 summary: {args.v11_summary}")
    if cfar_ac_summary is not None:
        summaries["CFAR-AC"] = cfar_ac_summary
    else:
        print(f"未找到 CFAR-AC summary: {args.cfar_ac_summary}")

    comparison_path = os.path.join(args.output_dir, "comparison_summary.csv")
    save_comparison_csv(comparison_path, summaries)

    print(f"输出目录: {args.output_dir}")
    print(f"逐样本指标: {metrics_path}")
    print(f"汇总指标: {summary_path}")
    print(f"对比指标: {comparison_path}")
    print(f"平均 CFAR mask 覆盖比例: {float(np.mean(mask_ratios)):.6f}")
    print(
        "CFAR 参数: "
        f"train_cells={args.train_cells}, guard_cells={args.guard_cells}, pfa={args.pfa}, "
        f"threshold_scale={args.threshold_scale}, freq_dilate={args.freq_dilate}, time_dilate={args.time_dilate}"
    )
    print_key_summary("hybrid_v11_cfar 关键汇总指标", hybrid_summary)
    print_comparison_table(summaries)


if __name__ == "__main__":
    main()
