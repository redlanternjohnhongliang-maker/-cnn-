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
from metrics import (
    METRIC_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    compute_metrics_for_sample,
    label_target_indices,
    summarize_metrics,
)
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


def parse_args():
    parser = argparse.ArgumentParser(description="ARIM 时频图去干扰评估")
    parser.add_argument("--test_path", required=True, help="arim_test.npy 路径")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth 路径")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--save_images", action="store_true", help="保存样本图像")
    parser.add_argument("--num_images", type=int, default=None, help="最多保存多少个样本图像；默认保存全部评估样本")
    parser.add_argument("--output_dir", default=None, help="评估输出目录；默认保存到 outputs")
    parser.add_argument(
        "--mode",
        choices=["magnitude", "complex", "complex_mask_residual", "complex_mask_clean"],
        default="magnitude",
        help="输入输出模式",
    )
    return parser.parse_args()


def resolve_output_dir(script_dir, output_dir, mode):
    """解析评估输出目录。

    magnitude 默认保持 v0 的 outputs 不变；complex 默认输出到 outputs_v1。
    """
    if output_dir is None:
        if mode == "complex_mask_clean":
            return os.path.join(script_dir, "outputs_v14_full")
        if mode == "complex_mask_residual":
            return os.path.join(script_dir, "outputs_v12_smoke")
        if mode == "complex":
            return os.path.join(script_dir, "outputs_v1")
        return os.path.join(script_dir, "outputs")
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(script_dir, output_dir)


def display_tf_image(value, mode):
    """把模型张量转成可视化用的二维图。

    magnitude 模式本来就是 [H,W]；complex 模式显示 sqrt(real^2 + imag^2)。
    """
    value = np.asarray(value)
    if mode == "magnitude":
        return value
    return np.sqrt(value[0] ** 2 + value[1] ** 2)


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


def load_label_arrays(test_path, max_samples=None):
    """读取 ARIM 测试集里的目标标签。

    v0 的 Dataset 只需要 sb/sb0；标签峰值评估额外从原始 npy 中读取
    distances 和 amplitudes，不改训练数据读取逻辑。
    """
    data = np.load(test_path, allow_pickle=True)[()]
    labels = {
        "distances": data.get("distances"),
        "amplitudes": data.get("amplitudes"),
    }
    for key, value in list(labels.items()):
        if value is not None and max_samples is not None:
            labels[key] = value[:max_samples]
    return labels


def print_label_format(labels, sample_index):
    """自动检查标签格式，并打印一个样本的有效目标位置数量。"""
    distances = labels.get("distances")
    amplitudes = labels.get("amplitudes")
    print("标签格式检查:")
    if distances is None:
        print("  distances: 未找到")
    else:
        print(f"  distances shape={distances.shape}, dtype={distances.dtype}")
    if amplitudes is None:
        print("  amplitudes: 未找到")
    else:
        print(f"  amplitudes shape={amplitudes.shape}, dtype={amplitudes.dtype}")

    sample_distances = distances[sample_index] if distances is not None else None
    sample_amplitudes = amplitudes[sample_index] if amplitudes is not None else None
    target_indices, _ = label_target_indices(sample_distances, sample_amplitudes)
    preview = target_indices[:10].tolist()
    print(f"  样本 sample_id={sample_index} 的有效目标位置数量: {len(target_indices)}")
    print(f"  有效目标位置前 10 个 bin: {preview}")


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

    dataset = RadarTFDataset(args.test_path, max_samples=args.max_samples, tf_config=cfg, mode=args.mode)
    labels = load_label_arrays(args.test_path, max_samples=args.max_samples)
    if args.mode in {"complex_mask_residual", "complex_mask_clean"}:
        in_channels = int(checkpoint.get("in_channels", 3))
        out_channels = int(checkpoint.get("out_channels", 2))
    elif args.mode == "complex":
        in_channels = 2
        out_channels = 2
    else:
        in_channels = 1
        out_channels = 1
    model = SmallUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = resolve_output_dir(script_dir, args.output_dir, args.mode)
    os.makedirs(output_dir, exist_ok=True)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if args.num_samples is not None and args.num_samples > 0:
        indices = indices[: args.num_samples]
    if indices:
        print_label_format(labels, indices[0])

    metrics_rows = []

    with torch.no_grad():
        for out_idx, data_idx in enumerate(indices):
            if args.mode in {"complex_mask_residual", "complex_mask_clean"}:
                x, y, noisy_channels, mask = dataset[data_idx]
                model_out = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
                if args.mode == "complex_mask_residual":
                    pred_tensor = noisy_channels + mask * model_out
                else:
                    # v1.4: mask 内使用 clean 预测，非 mask 区域保留 noisy。
                    pred_tensor = mask * model_out + (1.0 - mask) * noisy_channels
                pred = pred_tensor.numpy()
                x_np = noisy_channels.numpy()
                y_np = y.numpy()
                mask_np = mask.numpy()
            else:
                x, y = dataset[data_idx]
                pred_tensor = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
                pred = pred_tensor.numpy()
                x_np = x.numpy()
                y_np = y.numpy()
                mask_np = None

            if args.mode == "magnitude":
                pred_metric = pred.squeeze(0)
                x_metric = x_np.squeeze(0)
                y_metric = y_np.squeeze(0)
            else:
                pred_metric = pred
                x_metric = x_np
                y_metric = y_np

            raw = dataset.raw_sample(data_idx)
            distances = labels["distances"][data_idx] if labels.get("distances") is not None else None
            amplitudes = labels["amplitudes"][data_idx] if labels.get("amplitudes") is not None else None

            if args.mode == "magnitude":
                _, _, noisy_zxx = complex_stft(raw["sb"], cfg)
                # v0/v0.1 只预测时频图幅度，不预测相位。为了计算距离谱指标，
                # 这里沿用当前已有的近似方式：预测幅度 + 带干扰输入相位。
                pred_signal = istft_from_magnitude_and_phase(pred_metric, noisy_zxx, len(raw["sb"]), cfg)
            else:
                # v1/v1.2 直接得到复数 STFT，因此用预测实部/虚部做 ISTFT。
                pred_signal = istft_from_channels(pred_metric, len(raw["sb"]), cfg)

            metrics_rows.append(
                compute_metrics_for_sample(
                    sample_id=data_idx,
                    noisy_tf=x_metric,
                    clean_tf=y_metric,
                    pred_tf=pred_metric,
                    noisy_signal=raw["sb"],
                    clean_signal=raw["sb0"],
                    pred_signal=pred_signal,
                    distances=distances,
                    amplitudes=amplitudes,
                )
            )

            if args.save_images and (args.num_images is None or out_idx < args.num_images):
                x_img = display_tf_image(x_metric, args.mode)
                y_img = display_tf_image(y_metric, args.mode)
                pred_img = display_tf_image(pred_metric, args.mode)
                prefix = f"sample_{out_idx:03d}_idx_{data_idx:05d}"
                save_tf_image(os.path.join(output_dir, f"{prefix}_01_noisy_tf.png"), x_img, "带干扰时频图")
                save_tf_image(os.path.join(output_dir, f"{prefix}_02_clean_tf.png"), y_img, "干净时频图")
                save_tf_image(os.path.join(output_dir, f"{prefix}_03_pred_tf.png"), pred_img, "模型输出时频图")
                if mask_np is not None:
                    save_tf_image(os.path.join(output_dir, f"{prefix}_04_cfar_mask.png"), mask_np.squeeze(0), "CFAR mask")
                save_range_plot(
                    os.path.join(output_dir, f"{prefix}_05_range_compare.png"),
                    raw["sb"],
                    raw["sb0"],
                    pred_signal,
                )

                # 额外保存一张三联图，方便快速扫结果。
                plt.figure(figsize=(12, 3.5))
                triplet_items = [(x_img, "带干扰"), (y_img, "干净"), (pred_img, "模型输出")]
                if mask_np is not None:
                    triplet_items = [(x_img, "带干扰"), (mask_np.squeeze(0), "CFAR mask"), (pred_img, "模型输出")]
                for i, (img, title) in enumerate(triplet_items, start=1):
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
