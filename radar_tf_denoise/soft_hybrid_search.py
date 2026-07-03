"""soft hybrid 参数搜索，不训练新模型。

soft hybrid 定义:
    Z_soft = W * Z_pred + (1 - W) * Z_noisy
    W = M + rho * (1 - M)

其中:
- M 是 CFAR 检测得到的干扰 mask。
- mask 区域 W=1，完全使用 v1.1 输出。
- 非 mask 区域 W=rho，部分使用 v1.1，部分保留 noisy。
- rho=0 等价于 hard hybrid。
- rho=1 等价于纯 v1.1。
"""

import argparse
import csv
import os
import random
from itertools import product
from typing import Dict, List, Optional, Sequence

# Windows/Conda 下 torch、scipy、matplotlib 可能重复加载 OpenMP。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines_cfar import cfar_mask_time, dilate_mask
from metrics import SUMMARY_FIELDNAMES, compute_metrics_for_sample, summarize_metrics
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

GRID_FIELDNAMES = [
    "rho",
    "pfa",
    "dilation_iter",
    "mean_mask_ratio",
    *SUMMARY_FIELDNAMES,
    "score",
]


def default_paths() -> Dict[str, str]:
    """生成默认输入输出路径，便于从脚本目录直接运行。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return {
        "test_path": os.path.join(repo_root, "training", "arim_test_random.npy"),
        "checkpoint": os.path.join(script_dir, "checkpoints_v11", "best_model.pth"),
        "output_dir": os.path.join(script_dir, "outputs_soft_hybrid_search"),
        "v11_summary": os.path.join(script_dir, "outputs_v11", "summary_metrics.csv"),
        "cfar_ac_summary": os.path.join(script_dir, "outputs_baseline_cfar", "summary_cfar_ac.csv"),
        "hard_hybrid_summary": os.path.join(script_dir, "outputs_hybrid_v11_cfar", "summary_metrics.csv"),
    }


def parse_float_list(text: str) -> List[float]:
    """解析逗号分隔的浮点数列表。"""
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    """解析逗号分隔的整数列表。"""
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="v1.1 soft hybrid 参数搜索")
    parser.add_argument("--test_path", default=paths["test_path"], help="arim_test_random.npy 路径")
    parser.add_argument("--checkpoint", default=paths["checkpoint"], help="v1.1 best_model.pth 路径")
    parser.add_argument("--output_dir", default=paths["output_dir"], help="输出目录")
    parser.add_argument("--num_samples", type=int, default=200, help="搜索样本数")
    parser.add_argument("--seed", type=int, default=707, help="随机种子，默认和 evaluate.py 一致")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--num_images", type=int, default=6, help="最佳参数下保存的可视化图数量")

    parser.add_argument("--rho_list", default="0.0,0.1,0.25,0.5,0.75,1.0", help="rho 搜索列表")
    parser.add_argument("--pfa_list", default="1e-5,1e-4,1e-3", help="Pfa 搜索列表")
    parser.add_argument("--dilation_iter_list", default="0,1,2", help="mask 膨胀迭代次数列表")
    parser.add_argument("--train_cells", type=int, default=4, help="CFAR 每侧训练单元数")
    parser.add_argument("--guard_cells", type=int, default=1, help="CFAR 每侧保护单元数")
    parser.add_argument("--threshold_scale", type=float, default=None, help="手动阈值倍数；默认使用 Pfa 公式")

    parser.add_argument("--v11_summary", default=paths["v11_summary"], help="v1.1 summary_metrics.csv 路径")
    parser.add_argument("--cfar_ac_summary", default=paths["cfar_ac_summary"], help="CFAR-AC summary CSV 路径")
    parser.add_argument("--hard_hybrid_summary", default=paths["hard_hybrid_summary"], help="hard hybrid summary CSV 路径")
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
    """使用固定随机种子抽样，保证和已有 v1.1/CFAR/hard hybrid 结果可比。"""
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
    """对单条样本预测 v1.1 复数 STFT。"""
    x = torch.from_numpy(complex_to_channels(noisy_zxx)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_channels = model(x).squeeze(0).cpu().numpy()
    return channels_to_complex(pred_channels)


def apply_dilation_iterations(mask: np.ndarray, dilation_iter: int) -> np.ndarray:
    """按迭代次数膨胀 mask；0 表示不膨胀。"""
    out = np.asarray(mask, dtype=bool)
    for _ in range(int(dilation_iter)):
        out = dilate_mask(out, freq_radius=1, time_radius=1)
    return out


def read_summary(path: str) -> Optional[Dict[str, float]]:
    """读取已有 summary CSV；不存在时返回 None。"""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return {key: float(value) for key, value in row.items()}


def score_summary(summary: Dict[str, float]) -> float:
    """计算本次搜索使用的综合得分。"""
    return float(
        summary["aggregate_spectrum_mae_improvement_ratio"]
        + summary["label_peak_error_improved_rate"]
        + summary["mean_label_pred_peak_keep_ratio"]
        - abs(summary["pred_to_clean_noise_floor_ratio"] - 1.0)
    )


def cache_samples(
    data: Dict[str, np.ndarray],
    indices: Sequence[int],
    model: SmallUNet,
    cfg: TFConfig,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    """缓存 STFT 和 v1.1 输出，避免网格搜索时重复跑网络。"""
    samples = []
    for sample_id in indices:
        noisy_signal = data["sb"][sample_id]
        clean_signal = data["sb0"][sample_id]
        _, _, noisy_zxx = complex_stft(noisy_signal, cfg)
        _, _, clean_zxx = complex_stft(clean_signal, cfg)
        pred_zxx = predict_v11_zxx(model, noisy_zxx, device)

        samples.append(
            {
                "sample_id": int(sample_id),
                "noisy_signal": noisy_signal,
                "clean_signal": clean_signal,
                "distances": data["distances"][sample_id],
                "amplitudes": data["amplitudes"][sample_id],
                "noisy_zxx": noisy_zxx,
                "clean_zxx": clean_zxx,
                "pred_zxx": pred_zxx,
            }
        )
    return samples


def cache_masks(
    samples: Sequence[Dict[str, np.ndarray]],
    pfa_list: Sequence[float],
    dilation_iter_list: Sequence[int],
    train_cells: int,
    guard_cells: int,
    threshold_scale: Optional[float],
) -> Dict[tuple[float, int, int], np.ndarray]:
    """缓存每个样本、每个 pfa、每个 dilation_iter 对应的 CFAR mask。"""
    masks = {}
    for sample_idx, sample in enumerate(samples):
        power = np.abs(sample["noisy_zxx"]) ** 2
        for pfa in pfa_list:
            base_mask = cfar_mask_time(
                power,
                train_cells=train_cells,
                guard_cells=guard_cells,
                pfa=pfa,
                threshold_scale=threshold_scale,
            )
            for dilation_iter in dilation_iter_list:
                masks[(sample_idx, float(pfa), int(dilation_iter))] = apply_dilation_iterations(base_mask, dilation_iter)
    return masks


def evaluate_params(
    samples: Sequence[Dict[str, np.ndarray]],
    masks: Dict[tuple[float, int, int], np.ndarray],
    rho: float,
    pfa: float,
    dilation_iter: int,
    cfg: TFConfig,
) -> Dict[str, float]:
    """评估一组 soft hybrid 参数。"""
    rows = []
    mask_ratios = []
    rho = float(rho)

    for sample_idx, sample in enumerate(samples):
        mask = masks[(sample_idx, float(pfa), int(dilation_iter))]
        mask_ratios.append(float(np.mean(mask)))

        # W = M + rho * (1 - M)。mask 区域完全用 v1.1，非 mask 区域按 rho 软混合。
        weight = mask.astype(np.float64) + rho * (~mask).astype(np.float64)
        z_soft = weight * sample["pred_zxx"] + (1.0 - weight) * sample["noisy_zxx"]
        soft_signal = istft_from_channels(complex_to_channels(z_soft), len(sample["noisy_signal"]), cfg)

        rows.append(
            compute_metrics_for_sample(
                sample_id=sample["sample_id"],
                noisy_tf=complex_to_channels(sample["noisy_zxx"]),
                clean_tf=complex_to_channels(sample["clean_zxx"]),
                pred_tf=complex_to_channels(z_soft),
                noisy_signal=sample["noisy_signal"],
                clean_signal=sample["clean_signal"],
                pred_signal=soft_signal,
                distances=sample["distances"],
                amplitudes=sample["amplitudes"],
            )
        )

    summary = summarize_metrics(rows)
    grid_row = {
        "rho": rho,
        "pfa": float(pfa),
        "dilation_iter": int(dilation_iter),
        "mean_mask_ratio": float(np.mean(mask_ratios)),
        **summary,
    }
    grid_row["score"] = score_summary(summary)
    return grid_row


def write_rows(path: str, rows: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    """保存 CSV。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_csv(path: str, summaries: Dict[str, Dict[str, float]]) -> None:
    """保存最佳 soft hybrid 和已有方法的关键指标对比。"""
    fieldnames = ["method"] + KEY_METRICS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, summary in summaries.items():
            row = {"method": method}
            row.update({key: summary[key] for key in KEY_METRICS})
            writer.writerow(row)


def save_best_figures(
    output_dir: str,
    samples: Sequence[Dict[str, np.ndarray]],
    masks: Dict[tuple[float, int, int], np.ndarray],
    best_row: Dict[str, float],
    cfg: TFConfig,
    num_images: int,
) -> None:
    """保存最佳参数下的可视化图。"""
    rho = float(best_row["rho"])
    pfa = float(best_row["pfa"])
    dilation_iter = int(best_row["dilation_iter"])

    for out_idx, sample in enumerate(samples[:num_images]):
        mask = masks[(out_idx, pfa, dilation_iter)]
        weight = mask.astype(np.float64) + rho * (~mask).astype(np.float64)
        z_soft = weight * sample["pred_zxx"] + (1.0 - weight) * sample["noisy_zxx"]

        pred_signal = istft_from_channels(complex_to_channels(sample["pred_zxx"]), len(sample["noisy_signal"]), cfg)
        soft_signal = istft_from_channels(complex_to_channels(z_soft), len(sample["noisy_signal"]), cfg)

        fig = plt.figure(figsize=(16, 8))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
        fig.suptitle(
            f"soft hybrid 最佳参数 | sample_id={sample['sample_id']} | "
            f"rho={rho}, pfa={pfa:g}, dilation_iter={dilation_iter}",
            fontsize=14,
        )

        images = [
            (np.abs(sample["noisy_zxx"]), "带干扰时频图"),
            (mask.astype(float), "CFAR mask"),
            (np.abs(sample["pred_zxx"]), "v1.1 输出时频图"),
            (np.abs(z_soft), "soft hybrid 输出时频图"),
            (np.abs(sample["clean_zxx"]), "干净时频图"),
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
        ax.plot(20 * np.log10(range_spectrum(sample["noisy_signal"]) + eps), label="noisy", linewidth=1.0)
        ax.plot(20 * np.log10(range_spectrum(sample["clean_signal"]) + eps), label="clean", linewidth=1.0)
        ax.plot(20 * np.log10(range_spectrum(pred_signal) + eps), label="v1.1", linewidth=1.0)
        ax.plot(20 * np.log10(range_spectrum(soft_signal) + eps), label="soft hybrid", linewidth=1.0)
        ax.set_title("距离谱对比")
        ax.set_xlabel("range bin")
        ax.set_ylabel("magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"best_soft_case_{out_idx:03d}_sample_{sample['sample_id']:05d}.png")
        plt.savefig(fig_path, dpi=160)
        plt.close(fig)


def print_key_summary(title: str, summary: Dict[str, float]) -> None:
    """打印关键指标。"""
    print(title)
    for key in KEY_METRICS:
        print(f"{key}: {summary[key]:.6f}")


def print_comparison_table(summaries: Dict[str, Dict[str, float]]) -> None:
    """打印方法对比表。"""
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

    rho_list = parse_float_list(args.rho_list)
    pfa_list = parse_float_list(args.pfa_list)
    dilation_iter_list = parse_int_list(args.dilation_iter_list)

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_dataset(args.test_path)
    model, cfg = load_model(args.checkpoint, device)
    indices = select_indices(len(data["sb"]), args.num_samples, args.seed)

    print(f"搜索样本数: {len(indices)}")
    print(f"rho_list: {rho_list}")
    print(f"pfa_list: {pfa_list}")
    print(f"dilation_iter_list: {dilation_iter_list}")
    print("正在缓存 STFT 和 v1.1 输出...")
    samples = cache_samples(data, indices, model, cfg, device)
    print("正在缓存 CFAR mask...")
    masks = cache_masks(
        samples,
        pfa_list,
        dilation_iter_list,
        train_cells=args.train_cells,
        guard_cells=args.guard_cells,
        threshold_scale=args.threshold_scale,
    )

    grid_rows = []
    total = len(rho_list) * len(pfa_list) * len(dilation_iter_list)
    for combo_idx, (rho, pfa, dilation_iter) in enumerate(product(rho_list, pfa_list, dilation_iter_list), start=1):
        print(f"[{combo_idx}/{total}] rho={rho}, pfa={pfa:g}, dilation_iter={dilation_iter}")
        row = evaluate_params(samples, masks, rho, pfa, dilation_iter, cfg)
        grid_rows.append(row)
        print(
            "  score={score:.6f}, spectrum={spec:.6f}, peak_rate={peak_rate:.6f}, "
            "peak_keep={peak_keep:.6f}, noise_ratio={noise:.6f}".format(
                score=row["score"],
                spec=row["aggregate_spectrum_mae_improvement_ratio"],
                peak_rate=row["label_peak_error_improved_rate"],
                peak_keep=row["mean_label_pred_peak_keep_ratio"],
                noise=row["pred_to_clean_noise_floor_ratio"],
            )
        )

    grid_rows = sorted(grid_rows, key=lambda item: item["score"], reverse=True)
    best_row = grid_rows[0]

    grid_path = os.path.join(args.output_dir, "grid_summary.csv")
    best_path = os.path.join(args.output_dir, "best_summary.csv")
    write_rows(grid_path, grid_rows, GRID_FIELDNAMES)
    write_rows(best_path, [best_row], GRID_FIELDNAMES)
    save_best_figures(args.output_dir, samples, masks, best_row, cfg, args.num_images)

    comparisons = {"best_soft_hybrid": best_row}
    v11_summary = read_summary(args.v11_summary)
    cfar_ac_summary = read_summary(args.cfar_ac_summary)
    hard_hybrid_summary = read_summary(args.hard_hybrid_summary)
    if v11_summary is not None:
        comparisons["v1.1"] = v11_summary
    else:
        print(f"未找到 v1.1 summary: {args.v11_summary}")
    if cfar_ac_summary is not None:
        comparisons["CFAR-AC"] = cfar_ac_summary
    else:
        print(f"未找到 CFAR-AC summary: {args.cfar_ac_summary}")
    if hard_hybrid_summary is not None:
        comparisons["hard_hybrid"] = hard_hybrid_summary
    else:
        print(f"未找到 hard hybrid summary: {args.hard_hybrid_summary}")

    comparison_path = os.path.join(args.output_dir, "comparison_summary.csv")
    save_comparison_csv(comparison_path, comparisons)

    print(f"输出目录: {args.output_dir}")
    print(f"网格搜索汇总: {grid_path}")
    print(f"最佳参数汇总: {best_path}")
    print(f"对比汇总: {comparison_path}")
    print(
        "最优参数: "
        f"rho={best_row['rho']}, pfa={best_row['pfa']:.0e}, "
        f"dilation_iter={int(best_row['dilation_iter'])}, score={best_row['score']:.6f}, "
        f"mean_mask_ratio={best_row['mean_mask_ratio']:.6f}"
    )
    print_key_summary("best_soft_hybrid 关键汇总指标", best_row)
    print_comparison_table(comparisons)


if __name__ == "__main__":
    main()
