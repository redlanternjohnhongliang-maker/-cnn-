"""评估 U-Net 并保存可视化结果。"""

import argparse
import csv
import os

# Windows/Conda 下 torch、scipy、matplotlib 偶尔会重复加载 OpenMP。
# 这里仅作为评估脚本的本机兜底，避免画图流程被运行库冲突中断。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import RadarTFDataset
from metrics import METRIC_FIELDNAMES, SUMMARY_FIELDNAMES, compute_metrics_for_sample, summarize_metrics
from model_unet import SmallUNet
from stft_utils import (
    TFConfig,
    complex_stft,
    istft_from_magnitude_and_phase,
    model_space_to_magnitude,
    range_spectrum,
    stft_magnitude,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="ARIM 时频图去干扰评估")
    parser.add_argument("--test_path", required=True, help="arim_test.npy 路径")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth 路径")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--save_images", action="store_true", help="保存样本图像")
    return parser.parse_args()


def save_tf_image(path, image, title):
    plt.figure(figsize=(7, 4))
    plt.imshow(image, aspect="auto", origin="lower", cmap="turbo")
    plt.colorbar(label="model-space magnitude")
    plt.title(title)
    plt.xlabel("time frame")
    plt.ylabel("frequency bin")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_range_plot(path, noisy_signal, clean_signal, pred_signal):
    noisy_spec = range_spectrum(noisy_signal)
    clean_spec = range_spectrum(clean_signal)
    pred_spec = range_spectrum(pred_signal)

    eps = 1e-12
    plt.figure(figsize=(8, 4))
    plt.plot(20 * np.log10(noisy_spec + eps), label="带干扰 sb")
    plt.plot(20 * np.log10(clean_spec + eps), label="干净 sb0")
    plt.plot(20 * np.log10(pred_spec + eps), label="模型输出近似重建")
    plt.xlabel("range bin")
    plt.ylabel("magnitude (dB)")
    plt.title("距离谱对比")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = TFConfig(**checkpoint.get("tf_config", {}))

    dataset = RadarTFDataset(args.test_path, max_samples=args.max_samples, tf_config=cfg)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if args.num_samples is not None and args.num_samples > 0:
        indices = indices[: args.num_samples]

    metrics_rows = []

    with torch.no_grad():
        for out_idx, data_idx in enumerate(indices):
            x, y = dataset[data_idx]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()
            x_img = x.squeeze(0).numpy()
            y_img = y.squeeze(0).numpy()

            raw = dataset.raw_sample(data_idx)
            _, _, noisy_zxx = complex_stft(raw["sb"], cfg)

            # v0 只预测时频图幅度，不预测相位。为了计算距离谱指标，
            # 这里沿用当前已有的近似方式：预测幅度 + 带干扰输入相位。
            pred_signal = istft_from_magnitude_and_phase(pred, noisy_zxx, len(raw["sb"]), cfg)

            metrics_rows.append(
                compute_metrics_for_sample(
                    sample_id=data_idx,
                    noisy_tf=x_img,
                    clean_tf=y_img,
                    pred_tf=pred,
                    noisy_signal=raw["sb"],
                    clean_signal=raw["sb0"],
                    pred_signal=pred_signal,
                )
            )

            if args.save_images:
                prefix = f"sample_{out_idx:03d}_idx_{data_idx:05d}"
                save_tf_image(os.path.join(output_dir, f"{prefix}_01_noisy_tf.png"), x_img, "带干扰时频图")
                save_tf_image(os.path.join(output_dir, f"{prefix}_02_clean_tf.png"), y_img, "干净时频图")
                save_tf_image(os.path.join(output_dir, f"{prefix}_03_pred_tf.png"), pred, "模型输出时频图")
                save_range_plot(
                    os.path.join(output_dir, f"{prefix}_04_range_compare.png"),
                    raw["sb"],
                    raw["sb0"],
                    pred_signal,
                )

                # 额外保存一张三联图，方便快速扫结果。
                plt.figure(figsize=(12, 3.5))
                for i, (img, title) in enumerate([(x_img, "带干扰"), (y_img, "干净"), (pred, "模型输出")], start=1):
                    plt.subplot(1, 3, i)
                    plt.imshow(img, aspect="auto", origin="lower", cmap="turbo")
                    plt.title(title)
                    plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{prefix}_tf_triplet.png"), dpi=160)
                plt.close()

    metrics_path = os.path.join(output_dir, "metrics.csv")
    fieldnames = METRIC_FIELDNAMES
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = summarize_metrics(metrics_rows)
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary)

    print(f"指标已保存到: {metrics_path}")
    print(f"汇总指标已保存到: {summary_path}")
    if args.save_images:
        print(f"评估图像已保存到: {output_dir}")

    print("整体汇总指标:")
    for key in SUMMARY_FIELDNAMES:
        print(f"{key}: {summary[key]:.6f}")


if __name__ == "__main__":
    main()
