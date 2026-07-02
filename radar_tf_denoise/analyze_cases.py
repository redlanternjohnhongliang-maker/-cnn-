"""分析 v0 的典型成功样本和失败样本。

脚本只读取已有 metrics.csv、测试集和 best_model.pth，不训练、不改模型。
输出:
    outputs/case_analysis/*.png
    outputs/case_analysis/case_summary.csv
"""

import argparse
import csv
import os

# Windows/Conda 下多个科学计算库可能重复加载 OpenMP，这里只做本机兜底。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import RadarTFDataset
from metrics import METRIC_FIELDNAMES
from model_unet import SmallUNet
from stft_utils import TFConfig, complex_stft, istft_from_magnitude_and_phase, range_spectrum

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


CASE_DEFINITIONS = [
    ("spectrum_best", "距离谱改善最好", "spectrum_mae_improvement", True),
    ("spectrum_worst", "距离谱改善最差", "spectrum_mae_improvement", False),
    ("peak_best", "目标峰改善最好", "peak_error_improvement", True),
    ("peak_worst", "目标峰改善最差", "peak_error_improvement", False),
    ("lowest_peak_keep", "目标峰保持率最低", "pred_peak_keep_ratio", False),
]


def default_paths():
    """基于脚本位置生成默认路径，避免写死绝对路径。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "metrics_path": os.path.join(script_dir, "outputs", "metrics.csv"),
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "checkpoint": os.path.join(script_dir, "checkpoints", "best_model.pth"),
        "output_dir": os.path.join(script_dir, "outputs", "case_analysis"),
    }


def parse_args():
    paths = default_paths()
    parser = argparse.ArgumentParser(description="分析 v0 典型成功和失败样本")
    parser.add_argument("--metrics_path", default=paths["metrics_path"], help="evaluate.py 输出的 metrics.csv")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--checkpoint", default=paths["checkpoint"], help="best_model.pth 路径")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="案例分析输出目录")
    parser.add_argument("--top_k", type=int, default=5, help="每类挑选样本数")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    return parser.parse_args()


def read_metrics(path):
    """读取逐样本指标，并把数值列转成 float。"""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "sample_id":
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"metrics.csv 为空: {path}")
    return rows


def select_top_rows(rows, key, descending, top_k):
    """按某个指标挑选 top_k 个样本。"""
    return sorted(rows, key=lambda row: (row[key], row["sample_id"]), reverse=descending)[:top_k]


def select_cases(rows, top_k):
    """根据指标自动挑选所有案例类别。"""
    selected = []
    for case_key, case_name, metric_key, descending in CASE_DEFINITIONS:
        for rank, row in enumerate(select_top_rows(rows, metric_key, descending, top_k), start=1):
            selected.append((case_key, case_name, rank, row))

    # 底噪明显低于干净信号：只选 pred_noise_floor < clean_noise_floor 的样本，
    # 并按 pred-clean 的负差值排序，越负说明模型把底噪压得越低。
    low_floor_rows = [row for row in rows if row["pred_noise_floor"] < row["clean_noise_floor"]]
    low_floor_rows = sorted(
        low_floor_rows,
        key=lambda row: (row["pred_noise_floor"] - row["clean_noise_floor"], row["sample_id"]),
    )[:top_k]
    for rank, row in enumerate(low_floor_rows, start=1):
        selected.append(("too_low_noise_floor", "预测底噪明显低于干净底噪", rank, row))

    return selected


def make_prediction(dataset, model, device, sample_id, cfg):
    """重新计算单个样本的模型输出和近似重建信号。"""
    x, y = dataset[sample_id]
    raw = dataset.raw_sample(sample_id)
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()

    _, _, noisy_zxx = complex_stft(raw["sb"], cfg)
    # v0 只预测幅度图，不预测相位；距离谱仍使用“预测幅度 + 输入相位”的近似重建。
    pred_signal = istft_from_magnitude_and_phase(pred, noisy_zxx, len(raw["sb"]), cfg)
    return x.squeeze(0).numpy(), y.squeeze(0).numpy(), pred, raw["sb"], raw["sb0"], pred_signal


def save_case_figure(path, case_name, rank, row, x_img, y_img, pred_img, noisy_signal, clean_signal, pred_signal):
    """保存单个案例图：三张时频图 + 一张距离谱对比。"""
    noisy_spec = range_spectrum(noisy_signal)
    clean_spec = range_spectrum(clean_signal)
    pred_spec = range_spectrum(pred_signal)

    # 三张时频图使用同一色标，方便肉眼比较能量是否被过度压低。
    vmax = float(np.max([np.max(x_img), np.max(y_img), np.max(pred_img)]))
    vmin = float(np.min([np.min(x_img), np.min(y_img), np.min(pred_img)]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"{case_name} #{rank} | sample_id={row['sample_id']} | "
        f"spec_improve={row['spectrum_mae_improvement']:.2f} | "
        f"peak_improve={row['peak_error_improvement']:.2f}",
        fontsize=12,
    )

    images = [
        (axes[0, 0], x_img, "带干扰时频图"),
        (axes[0, 1], y_img, "干净时频图"),
        (axes[1, 0], pred_img, "模型输出时频图"),
    ]
    for ax, image, title in images:
        im = ax.imshow(image, aspect="auto", origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time frame")
        ax.set_ylabel("frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    eps = 1e-12
    ax = axes[1, 1]
    ax.plot(20 * np.log10(noisy_spec + eps), label="带干扰 sb", linewidth=1.0)
    ax.plot(20 * np.log10(clean_spec + eps), label="干净 sb0", linewidth=1.0)
    ax.plot(20 * np.log10(pred_spec + eps), label="模型输出近似重建", linewidth=1.0)
    ax.set_title("距离谱对比")
    ax.set_xlabel("range bin")
    ax.set_ylabel("magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

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

    rows = read_metrics(args.metrics_path)
    selected = select_cases(rows, args.top_k)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = TFConfig(**checkpoint.get("tf_config", {}))
    dataset = RadarTFDataset(args.test_path, tf_config=cfg)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    summary_rows = []
    printed_cases = {}
    for case_key, case_name, rank, row in selected:
        sample_id = row["sample_id"]
        x_img, y_img, pred_img, noisy_signal, clean_signal, pred_signal = make_prediction(
            dataset,
            model,
            device,
            sample_id,
            cfg,
        )

        image_name = f"{case_key}_{rank:02d}_sample_{sample_id:05d}.png"
        image_path = os.path.join(args.output_dir, image_name)
        save_case_figure(
            image_path,
            case_name,
            rank,
            row,
            x_img,
            y_img,
            pred_img,
            noisy_signal,
            clean_signal,
            pred_signal,
        )

        summary = {
            "case_key": case_key,
            "case_name": case_name,
            "rank": rank,
            "image_file": image_name,
        }
        summary.update({key: row[key] for key in METRIC_FIELDNAMES if key in row})
        summary_rows.append(summary)
        printed_cases.setdefault(case_name, []).append(sample_id)

    summary_path = os.path.join(args.output_dir, "case_summary.csv")
    fieldnames = ["case_key", "case_name", "rank", "image_file"] + METRIC_FIELDNAMES
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"案例图输出目录: {args.output_dir}")
    print(f"案例汇总表: {summary_path}")
    print("每类样本 sample_id:")
    for case_name, sample_ids in printed_cases.items():
        print(f"{case_name}: {sample_ids}")
    print("生成图像:")
    for row in summary_rows:
        print(row["image_file"])


if __name__ == "__main__":
    main()
