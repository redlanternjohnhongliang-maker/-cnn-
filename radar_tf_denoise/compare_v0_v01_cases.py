"""生成 v0 和 v0.1 的同样本可视化对比图。

脚本只加载已有模型和测试集，不训练、不修改模型结构。
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
from stft_utils import TFConfig, complex_stft, istft_from_magnitude_and_phase, range_spectrum

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SUMMARY_FIELDNAMES = [
    "sample_id",
    "v0_label_peak_keep_ratio",
    "v01_label_peak_keep_ratio",
    "v0_label_peak_error_pred",
    "v01_label_peak_error_pred",
    "v0_pred_noise_floor",
    "v01_pred_noise_floor",
]


def default_paths():
    """基于脚本位置生成默认路径，避免写死绝对路径。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "v0_checkpoint": os.path.join(script_dir, "checkpoints", "best_model.pth"),
        "v01_checkpoint": os.path.join(script_dir, "checkpoints_v01", "best_model.pth"),
        "v0_metrics": os.path.join(script_dir, "outputs", "metrics.csv"),
        "v01_metrics": os.path.join(script_dir, "outputs_v01", "metrics.csv"),
        "output_dir": os.path.join(script_dir, "outputs_compare_v0_v01"),
    }


def parse_args():
    paths = default_paths()
    parser = argparse.ArgumentParser(description="生成 v0 / v0.1 同样本对比图")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--v0_checkpoint", default=paths["v0_checkpoint"], help="v0 best_model.pth")
    parser.add_argument("--v01_checkpoint", default=paths["v01_checkpoint"], help="v0.1 best_model.pth")
    parser.add_argument("--v0_metrics", default=paths["v0_metrics"], help="v0 metrics.csv")
    parser.add_argument("--v01_metrics", default=paths["v01_metrics"], help="v0.1 metrics.csv")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="对比图输出目录")
    parser.add_argument("--num_samples", type=int, default=10, help="对比样本数量")
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


def select_samples(v0_rows, v01_rows, num_samples):
    """优先选择 v0.1 在标签目标峰上明显改善的样本。

    分数越高表示 v0.1 相比 v0 的标签目标峰误差下降更多，同时峰值保持率更高。
    如果没有可用指标，则回退到前 num_samples 个样本。
    """
    scored = []
    for sample_id in sorted(set(v0_rows) & set(v01_rows)):
        v0 = v0_rows[sample_id]
        v01 = v01_rows[sample_id]
        error_gain = v0["label_peak_error_pred"] - v01["label_peak_error_pred"]
        keep_gain = v01["label_pred_peak_keep_ratio"] - v0["label_pred_peak_keep_ratio"]
        if error_gain > 0 and keep_gain > 0:
            score = error_gain + 100.0 * keep_gain
            scored.append((score, sample_id))

    if scored:
        return [sample_id for _, sample_id in sorted(scored, reverse=True)[:num_samples]]
    return sorted(v0_rows)[:num_samples]


def load_model(checkpoint_path, device):
    """加载单个 U-Net 模型。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = TFConfig(**checkpoint.get("tf_config", {}))
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def predict_one(model, dataset, sample_id, device, cfg):
    """对一个样本预测时频图，并用输入相位近似重建时域信号。"""
    x, y = dataset[sample_id]
    raw = dataset.raw_sample(sample_id)
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()
    _, _, noisy_zxx = complex_stft(raw["sb"], cfg)
    pred_signal = istft_from_magnitude_and_phase(pred, noisy_zxx, len(raw["sb"]), cfg)
    return x.squeeze(0).numpy(), y.squeeze(0).numpy(), pred, raw["sb"], raw["sb0"], pred_signal


def save_compare_figure(path, sample_id, noisy_tf, clean_tf, v0_tf, v01_tf, noisy_signal, clean_signal, v0_signal, v01_signal):
    """保存一张同样本 v0/v0.1 对比图。"""
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    fig.suptitle(f"v0 与 v0.1 同样本对比 | sample_id={sample_id}", fontsize=14)

    tf_images = [
        (noisy_tf, "带干扰时频图"),
        (clean_tf, "干净时频图"),
        (v0_tf, "v0 输出时频图"),
        (v01_tf, "v0.1 输出时频图"),
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
    ax.plot(20 * np.log10(range_spectrum(v0_signal) + eps), label="v0 输出", linewidth=1.0)
    ax.plot(20 * np.log10(range_spectrum(v01_signal) + eps), label="v0.1 输出", linewidth=1.0)
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

    v0_rows = read_metrics(args.v0_metrics)
    v01_rows = read_metrics(args.v01_metrics)
    sample_ids = select_samples(v0_rows, v01_rows, args.num_samples)

    v0_model, v0_cfg = load_model(args.v0_checkpoint, device)
    v01_model, v01_cfg = load_model(args.v01_checkpoint, device)
    dataset = RadarTFDataset(args.test_path, tf_config=v0_cfg)

    summary_rows = []
    image_names = []
    for rank, sample_id in enumerate(sample_ids, start=1):
        noisy_tf, clean_tf, v0_tf, noisy_signal, clean_signal, v0_signal = predict_one(
            v0_model,
            dataset,
            sample_id,
            device,
            v0_cfg,
        )
        _, _, v01_tf, _, _, v01_signal = predict_one(
            v01_model,
            dataset,
            sample_id,
            device,
            v01_cfg,
        )

        image_name = f"compare_{rank:02d}_sample_{sample_id:05d}.png"
        save_compare_figure(
            os.path.join(args.output_dir, image_name),
            sample_id,
            noisy_tf,
            clean_tf,
            v0_tf,
            v01_tf,
            noisy_signal,
            clean_signal,
            v0_signal,
            v01_signal,
        )
        image_names.append(image_name)
        summary_rows.append(
            {
                "sample_id": sample_id,
                "v0_label_peak_keep_ratio": v0_rows[sample_id]["label_pred_peak_keep_ratio"],
                "v01_label_peak_keep_ratio": v01_rows[sample_id]["label_pred_peak_keep_ratio"],
                "v0_label_peak_error_pred": v0_rows[sample_id]["label_peak_error_pred"],
                "v01_label_peak_error_pred": v01_rows[sample_id]["label_peak_error_pred"],
                "v0_pred_noise_floor": v0_rows[sample_id]["pred_noise_floor"],
                "v01_pred_noise_floor": v01_rows[sample_id]["pred_noise_floor"],
            }
        )

    summary_path = os.path.join(args.output_dir, "compare_case_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"输出目录: {args.output_dir}")
    print(f"对比汇总表: {summary_path}")
    print(f"选中 sample_id: {sample_ids}")
    print("生成图像:")
    for name in image_names:
        print(name)


if __name__ == "__main__":
    main()
