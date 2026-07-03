"""生成 v0.1 和 v1.1 的同样本可视化对比图。

脚本只加载已有模型、测试集和 metrics.csv，不训练、不修改模型。
"""

import argparse
import csv
import os

# Windows/Conda 下多个科学计算库可能重复加载 OpenMP，这里只做评估画图兜底。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import RadarTFDataset
from model_unet import SmallUNet
from stft_utils import (
    TFConfig,
    complex_stft,
    istft_from_channels,
    istft_from_magnitude_and_phase,
    range_spectrum,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SUMMARY_FIELDNAMES = [
    "case_type",
    "rank",
    "sample_id",
    "image_file",
    "v01_label_peak_keep_ratio",
    "v11_label_peak_keep_ratio",
    "v01_label_peak_error_pred",
    "v11_label_peak_error_pred",
    "v01_pred_noise_floor",
    "v11_pred_noise_floor",
    "v11_minus_v01_peak_error",
]


def default_paths():
    """基于脚本位置生成默认路径，避免写死绝对路径。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "v01_checkpoint": os.path.join(script_dir, "checkpoints_v01", "best_model.pth"),
        "v11_checkpoint": os.path.join(script_dir, "checkpoints_v11", "best_model.pth"),
        "v01_metrics": os.path.join(script_dir, "outputs_v01", "metrics.csv"),
        "v11_metrics": os.path.join(script_dir, "outputs_v11", "metrics.csv"),
        "output_dir": os.path.join(script_dir, "outputs_compare_v01_v11"),
    }


def parse_args():
    paths = default_paths()
    parser = argparse.ArgumentParser(description="生成 v0.1 / v1.1 同样本对比图")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--v01_checkpoint", default=paths["v01_checkpoint"], help="v0.1 best_model.pth")
    parser.add_argument("--v11_checkpoint", default=paths["v11_checkpoint"], help="v1.1 best_model.pth")
    parser.add_argument("--v01_metrics", default=paths["v01_metrics"], help="v0.1 metrics.csv")
    parser.add_argument("--v11_metrics", default=paths["v11_metrics"], help="v1.1 metrics.csv")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="对比图输出目录")
    parser.add_argument("--top_k", type=int, default=5, help="改善/变差各选多少个样本")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    return parser.parse_args()


def read_metrics(path):
    """读取逐样本指标 CSV。"""
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(float(row["sample_id"]))
            rows[sample_id] = {key: float(value) for key, value in row.items() if key != "sample_id"}
    return rows


def select_samples(v01_rows, v11_rows, top_k):
    """挑选 v1.1 相比 v0.1 目标峰误差改善最大和变差最大的样本。"""
    deltas = []
    for sample_id in sorted(set(v01_rows) & set(v11_rows)):
        delta = v11_rows[sample_id]["label_peak_error_pred"] - v01_rows[sample_id]["label_peak_error_pred"]
        deltas.append((delta, sample_id))

    best = sorted(deltas, key=lambda item: (item[0], item[1]))[:top_k]
    worst = sorted(deltas, key=lambda item: (item[0], item[1]), reverse=True)[:top_k]

    selected = []
    selected.extend(("improved", rank, sample_id) for rank, (_, sample_id) in enumerate(best, start=1))
    selected.extend(("worse", rank, sample_id) for rank, (_, sample_id) in enumerate(worst, start=1))
    return selected


def load_model(checkpoint_path, device, mode):
    """加载单个 U-Net 模型。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = TFConfig(**checkpoint.get("tf_config", {}))
    channels = 2 if mode == "complex" else 1
    model = SmallUNet(in_channels=channels, out_channels=channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def display_tf_image(value, mode):
    """把模型张量转成二维可视化图。"""
    value = np.asarray(value)
    if mode == "magnitude":
        return value
    return np.sqrt(value[0] ** 2 + value[1] ** 2)


def predict_magnitude(model, dataset, sample_id, device, cfg):
    """v0.1 单通道幅度预测，并借用输入相位近似 ISTFT。"""
    x, y = dataset[sample_id]
    raw = dataset.raw_sample(sample_id)
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()

    _, _, noisy_zxx = complex_stft(raw["sb"], cfg)
    pred_signal = istft_from_magnitude_and_phase(pred, noisy_zxx, len(raw["sb"]), cfg)
    return x.squeeze(0).numpy(), y.squeeze(0).numpy(), pred, raw["sb"], raw["sb0"], pred_signal


def predict_complex(model, dataset, sample_id, device, cfg):
    """v1.1 复数两通道预测，并直接用预测实部/虚部做 ISTFT。"""
    x, y = dataset[sample_id]
    raw = dataset.raw_sample(sample_id)
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    pred_signal = istft_from_channels(pred, len(raw["sb"]), cfg)
    return display_tf_image(pred, "complex"), pred_signal


def save_compare_figure(
    path,
    case_type,
    sample_id,
    noisy_tf,
    clean_tf,
    v01_tf,
    v11_tf,
    noisy_signal,
    clean_signal,
    v01_signal,
    v11_signal,
):
    """保存一张同样本 v0.1/v1.1 对比图。"""
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    case_name = "v1.1 改善最大" if case_type == "improved" else "v1.1 变差最大"
    fig.suptitle(f"v0.1 与 v1.1 同样本对比 | {case_name} | sample_id={sample_id}", fontsize=14)

    tf_images = [
        (noisy_tf, "带干扰时频图"),
        (clean_tf, "干净时频图"),
        (v01_tf, "v0.1 输出时频图"),
        (v11_tf, "v1.1 输出时频图"),
    ]
    vmin = float(np.min([np.min(image) for image, _ in tf_images]))
    vmax = float(np.max([np.max(image) for image, _ in tf_images]))
    for col, (image, title) in enumerate(tf_images):
        ax = fig.add_subplot(grid[0, col])
        im = ax.imshow(image, aspect="auto", origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time frame")
        ax.set_ylabel("frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    eps = 1e-12
    ax = fig.add_subplot(grid[1, :])
    ax.plot(20 * np.log10(range_spectrum(noisy_signal) + eps), label="带干扰 noisy", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(clean_signal) + eps), label="干净 clean", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(v01_signal) + eps), label="v0.1 输出", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(v11_signal) + eps), label="v1.1 输出", linewidth=1.0)
    ax.set_title("距离谱对比")
    ax.set_xlabel("range bin")
    ax.set_ylabel("magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    v01_rows = read_metrics(args.v01_metrics)
    v11_rows = read_metrics(args.v11_metrics)
    selected = select_samples(v01_rows, v11_rows, args.top_k)

    v01_model, v01_cfg = load_model(args.v01_checkpoint, device, mode="magnitude")
    v11_model, v11_cfg = load_model(args.v11_checkpoint, device, mode="complex")
    v01_dataset = RadarTFDataset(args.test_path, tf_config=v01_cfg, mode="magnitude")
    v11_dataset = RadarTFDataset(args.test_path, tf_config=v11_cfg, mode="complex")

    summary_rows = []
    image_names = []
    for case_type, rank, sample_id in selected:
        noisy_tf, clean_tf, v01_tf, noisy_signal, clean_signal, v01_signal = predict_magnitude(
            v01_model,
            v01_dataset,
            sample_id,
            device,
            v01_cfg,
        )
        v11_tf, v11_signal = predict_complex(v11_model, v11_dataset, sample_id, device, v11_cfg)

        image_name = f"{case_type}_{rank:02d}_sample_{sample_id:05d}.png"
        save_compare_figure(
            os.path.join(args.output_dir, image_name),
            case_type,
            sample_id,
            noisy_tf,
            clean_tf,
            v01_tf,
            v11_tf,
            noisy_signal,
            clean_signal,
            v01_signal,
            v11_signal,
        )
        image_names.append(image_name)

        v11_minus_v01_peak_error = (
            v11_rows[sample_id]["label_peak_error_pred"] - v01_rows[sample_id]["label_peak_error_pred"]
        )
        summary_rows.append(
            {
                "case_type": case_type,
                "rank": rank,
                "sample_id": sample_id,
                "image_file": image_name,
                "v01_label_peak_keep_ratio": v01_rows[sample_id]["label_pred_peak_keep_ratio"],
                "v11_label_peak_keep_ratio": v11_rows[sample_id]["label_pred_peak_keep_ratio"],
                "v01_label_peak_error_pred": v01_rows[sample_id]["label_peak_error_pred"],
                "v11_label_peak_error_pred": v11_rows[sample_id]["label_peak_error_pred"],
                "v01_pred_noise_floor": v01_rows[sample_id]["pred_noise_floor"],
                "v11_pred_noise_floor": v11_rows[sample_id]["pred_noise_floor"],
                "v11_minus_v01_peak_error": v11_minus_v01_peak_error,
            }
        )

    summary_path = os.path.join(args.output_dir, "compare_case_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    improved_ids = [row["sample_id"] for row in summary_rows if row["case_type"] == "improved"]
    worse_ids = [row["sample_id"] for row in summary_rows if row["case_type"] == "worse"]
    print(f"输出目录: {args.output_dir}")
    print(f"对比汇总表: {summary_path}")
    print(f"v1.1 改善最大 sample_id: {improved_ids}")
    print(f"v1.1 变差最大 sample_id: {worse_ids}")
    print("生成图像:")
    for name in image_names:
        print(name)


if __name__ == "__main__":
    main()
