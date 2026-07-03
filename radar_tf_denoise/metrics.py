"""v0 评估指标。

v0 模型只预测单通道 STFT 幅度图，不预测相位。因此距离谱相关指标使用
"预测幅度 + 带干扰输入相位" 近似反变换得到的时域信号，仅用于阶段性量化。
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from stft_utils import range_spectrum


METRIC_FIELDNAMES = [
    "sample_id",
    "tf_mae",
    "spectrum_mae",
    "noisy_spectrum_mae",
    "pred_spectrum_mae",
    "spectrum_mae_improvement",
    "spectrum_mae_improvement_ratio",
    "noisy_noise_floor",
    "clean_noise_floor",
    "pred_noise_floor",
    "noise_floor_reduction",
    "peak_error",
    "noisy_peak_error",
    "pred_peak_error",
    "peak_error_improvement",
    "peak_error_improvement_ratio",
    "clean_peak_mean",
    "pred_peak_keep_ratio",
    "noisy_peak_keep_ratio",
    "label_peak_clean_mean",
    "label_peak_noisy_mean",
    "label_peak_pred_mean",
    "label_pred_peak_keep_ratio",
    "label_noisy_peak_keep_ratio",
    "label_peak_error_noisy",
    "label_peak_error_pred",
    "label_peak_error_improvement",
]


SUMMARY_FIELDNAMES = [
    "mean_noisy_spectrum_mae",
    "mean_pred_spectrum_mae",
    "aggregate_spectrum_mae_improvement_ratio",
    "median_spectrum_mae_improvement_ratio",
    "spectrum_improved_rate",
    "mean_noisy_peak_error",
    "mean_pred_peak_error",
    "aggregate_peak_error_improvement_ratio",
    "median_peak_error_improvement_ratio",
    "peak_improved_rate",
    "mean_clean_noise_floor",
    "mean_pred_noise_floor",
    "pred_to_clean_noise_floor_ratio",
    "mean_pred_peak_keep_ratio",
    "median_pred_peak_keep_ratio",
    "valid_label_peak_sample_count",
    "valid_label_peak_sample_rate",
    "mean_label_peak_clean_mean",
    "mean_label_peak_noisy_mean",
    "mean_label_peak_pred_mean",
    "aggregate_label_pred_peak_keep_ratio",
    "mean_label_pred_peak_keep_ratio",
    "median_label_pred_peak_keep_ratio",
    "aggregate_label_noisy_peak_keep_ratio",
    "mean_label_noisy_peak_keep_ratio",
    "mean_label_peak_error_noisy",
    "mean_label_peak_error_pred",
    "mean_label_peak_error_improvement",
    "label_peak_error_improved_rate",
]


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """平均绝对误差。"""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def estimate_noise_floor(spectrum: np.ndarray, low_ratio: float = 0.7) -> float:
    """估计距离谱底噪。

    简单做法：对幅度排序，取较低 low_ratio 比例的幅度均值作为底噪。
    """
    values = np.sort(np.asarray(spectrum).reshape(-1))
    count = max(1, int(len(values) * low_ratio))
    return float(np.mean(values[:count]))


def safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    """安全计算比例，避免分母接近 0 时产生 inf。"""
    if abs(float(denominator)) < eps:
        return 0.0
    return float(numerator) / float(denominator)


def safe_nanmean(values: np.ndarray) -> float:
    """忽略 NaN 求均值；如果全是 NaN，则返回 NaN。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def safe_nanmedian(values: np.ndarray) -> float:
    """忽略 NaN 求中位数；如果全是 NaN，则返回 NaN。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmedian(values))


def full_range_spectrum(x: np.ndarray, nfft: int = 2048) -> np.ndarray:
    """计算完整 2048 点距离谱幅度，用于和 ARIM 标签 bin 对齐。"""
    return np.abs(np.fft.fft(np.asarray(x).reshape(-1), nfft))


def label_target_indices(
    distances: Optional[np.ndarray],
    amplitudes: Optional[np.ndarray],
    eps: float = 1e-12,
) -> Tuple[np.ndarray, int]:
    """根据 ARIM 标签找真实目标位置。

    distances 通常是 2048 维距离标签，非零位置表示目标 bin；
    amplitudes 通常是 2048 维复数幅度标签，因此用 abs(amplitudes) > 0 判断有效位置。
    两者都存在时取并集，避免某一列格式变化导致漏检。
    """
    masks = []
    label_length = 0

    if distances is not None:
        distance_values = np.asarray(distances).reshape(-1)
        label_length = max(label_length, len(distance_values))
        distance_mask = np.isfinite(distance_values) & (np.abs(distance_values) > eps)
        masks.append(distance_mask)

    if amplitudes is not None:
        amplitude_values = np.asarray(amplitudes).reshape(-1)
        label_length = max(label_length, len(amplitude_values))
        amplitude_mask = np.isfinite(amplitude_values) & (np.abs(amplitude_values) > eps)
        masks.append(amplitude_mask)

    if not masks:
        return np.asarray([], dtype=np.int64), 2048

    valid = np.zeros(label_length, dtype=bool)
    for mask in masks:
        valid[: len(mask)] |= mask
    return np.flatnonzero(valid).astype(np.int64), label_length


def clean_peak_indices(clean_spectrum: np.ndarray, top_k: int = 3) -> np.ndarray:
    """在干净距离谱中寻找 top_k 个目标峰位置。"""
    clean_spectrum = np.asarray(clean_spectrum).reshape(-1)
    top_k = min(top_k, len(clean_spectrum))
    return np.argpartition(clean_spectrum, -top_k)[-top_k:]


def peak_error(clean_spectrum: np.ndarray, pred_spectrum: np.ndarray, top_k: int = 3) -> float:
    """目标峰值误差。

    在干净距离谱中找 top_k 个最大峰值位置，比较模型输出在这些位置的幅度差异。
    """
    clean_spectrum = np.asarray(clean_spectrum).reshape(-1)
    pred_spectrum = np.asarray(pred_spectrum).reshape(-1)
    peak_indices = clean_peak_indices(clean_spectrum, top_k=top_k)
    return mean_absolute_error(pred_spectrum[peak_indices], clean_spectrum[peak_indices])


def compute_label_peak_metrics(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    pred_signal: np.ndarray,
    distances: Optional[np.ndarray],
    amplitudes: Optional[np.ndarray],
) -> Dict[str, float]:
    """计算基于标签目标位置的峰值指标。

    如果该样本没有有效目标标签，则所有标签峰值指标写 NaN。
    """
    target_indices, label_length = label_target_indices(distances, amplitudes)
    if len(target_indices) == 0:
        return {
            "label_peak_clean_mean": float("nan"),
            "label_peak_noisy_mean": float("nan"),
            "label_peak_pred_mean": float("nan"),
            "label_pred_peak_keep_ratio": float("nan"),
            "label_noisy_peak_keep_ratio": float("nan"),
            "label_peak_error_noisy": float("nan"),
            "label_peak_error_pred": float("nan"),
            "label_peak_error_improvement": float("nan"),
        }

    nfft = max(2048, label_length)
    noisy_spectrum = full_range_spectrum(noisy_signal, nfft=nfft)
    clean_spectrum = full_range_spectrum(clean_signal, nfft=nfft)
    pred_spectrum = full_range_spectrum(pred_signal, nfft=nfft)

    clean_values = clean_spectrum[target_indices]
    noisy_values = noisy_spectrum[target_indices]
    pred_values = pred_spectrum[target_indices]

    clean_mean = float(np.mean(clean_values))
    noisy_mean = float(np.mean(noisy_values))
    pred_mean = float(np.mean(pred_values))
    noisy_error = mean_absolute_error(noisy_values, clean_values)
    pred_error = mean_absolute_error(pred_values, clean_values)

    return {
        "label_peak_clean_mean": clean_mean,
        "label_peak_noisy_mean": noisy_mean,
        "label_peak_pred_mean": pred_mean,
        "label_pred_peak_keep_ratio": safe_ratio(pred_mean, clean_mean),
        "label_noisy_peak_keep_ratio": safe_ratio(noisy_mean, clean_mean),
        "label_peak_error_noisy": noisy_error,
        "label_peak_error_pred": pred_error,
        "label_peak_error_improvement": noisy_error - pred_error,
    }


def compute_metrics_for_sample(
    sample_id: int,
    noisy_tf: np.ndarray,
    clean_tf: np.ndarray,
    pred_tf: np.ndarray,
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    pred_signal: np.ndarray,
    distances: Optional[np.ndarray] = None,
    amplitudes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """计算单个样本的全部指标。"""
    noisy_spectrum = range_spectrum(noisy_signal)
    clean_spectrum = range_spectrum(clean_signal)
    pred_spectrum = range_spectrum(pred_signal)

    noisy_floor = estimate_noise_floor(noisy_spectrum)
    clean_floor = estimate_noise_floor(clean_spectrum)
    pred_floor = estimate_noise_floor(pred_spectrum)

    noisy_spectrum_mae = mean_absolute_error(noisy_spectrum, clean_spectrum)
    pred_spectrum_mae = mean_absolute_error(pred_spectrum, clean_spectrum)
    spectrum_mae_improvement = noisy_spectrum_mae - pred_spectrum_mae

    peak_indices = clean_peak_indices(clean_spectrum, top_k=3)
    clean_peak_values = clean_spectrum[peak_indices]
    noisy_peak_values = noisy_spectrum[peak_indices]
    pred_peak_values = pred_spectrum[peak_indices]

    clean_peak_mean = float(np.mean(clean_peak_values))
    noisy_peak_mean = float(np.mean(noisy_peak_values))
    pred_peak_mean = float(np.mean(pred_peak_values))
    noisy_peak_error = mean_absolute_error(noisy_peak_values, clean_peak_values)
    pred_peak_error = mean_absolute_error(pred_peak_values, clean_peak_values)
    peak_error_improvement = noisy_peak_error - pred_peak_error

    metrics = {
        "sample_id": int(sample_id),
        "tf_mae": mean_absolute_error(pred_tf, clean_tf),
        # 保留旧列名，等价于 pred_spectrum_mae，方便和前一次结果对照。
        "spectrum_mae": pred_spectrum_mae,
        "noisy_spectrum_mae": noisy_spectrum_mae,
        "pred_spectrum_mae": pred_spectrum_mae,
        "spectrum_mae_improvement": spectrum_mae_improvement,
        "spectrum_mae_improvement_ratio": safe_ratio(spectrum_mae_improvement, noisy_spectrum_mae),
        "noisy_noise_floor": noisy_floor,
        "clean_noise_floor": clean_floor,
        "pred_noise_floor": pred_floor,
        "noise_floor_reduction": noisy_floor - pred_floor,
        # 保留旧列名，等价于 pred_peak_error。
        "peak_error": pred_peak_error,
        "noisy_peak_error": noisy_peak_error,
        "pred_peak_error": pred_peak_error,
        "peak_error_improvement": peak_error_improvement,
        "peak_error_improvement_ratio": safe_ratio(peak_error_improvement, noisy_peak_error),
        "clean_peak_mean": clean_peak_mean,
        "pred_peak_keep_ratio": safe_ratio(pred_peak_mean, clean_peak_mean),
        "noisy_peak_keep_ratio": safe_ratio(noisy_peak_mean, clean_peak_mean),
    }
    metrics.update(
        compute_label_peak_metrics(
            noisy_signal=noisy_signal,
            clean_signal=clean_signal,
            pred_signal=pred_signal,
            distances=distances,
            amplitudes=amplitudes,
        )
    )
    return metrics


def metric_values(rows: Sequence[Dict[str, float]], key: str) -> np.ndarray:
    """从逐样本指标中取出某一列，统一转成浮点数组。"""
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def summarize_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """计算整体汇总指标。

    注意：aggregate ratio 先对误差求均值，再计算改善比例，
    避免逐样本 ratio 被少数异常样本过度拉偏。
    """
    if not rows:
        raise ValueError("没有可汇总的逐样本指标")

    noisy_spectrum_mae = metric_values(rows, "noisy_spectrum_mae")
    pred_spectrum_mae = metric_values(rows, "pred_spectrum_mae")
    spectrum_ratio = metric_values(rows, "spectrum_mae_improvement_ratio")

    noisy_peak_error = metric_values(rows, "noisy_peak_error")
    pred_peak_error = metric_values(rows, "pred_peak_error")
    peak_ratio = metric_values(rows, "peak_error_improvement_ratio")

    clean_noise_floor = metric_values(rows, "clean_noise_floor")
    pred_noise_floor = metric_values(rows, "pred_noise_floor")
    pred_peak_keep_ratio = metric_values(rows, "pred_peak_keep_ratio")
    label_peak_clean_mean = metric_values(rows, "label_peak_clean_mean")
    label_peak_noisy_mean = metric_values(rows, "label_peak_noisy_mean")
    label_peak_pred_mean = metric_values(rows, "label_peak_pred_mean")
    label_pred_peak_keep_ratio = metric_values(rows, "label_pred_peak_keep_ratio")
    label_noisy_peak_keep_ratio = metric_values(rows, "label_noisy_peak_keep_ratio")
    label_peak_error_noisy = metric_values(rows, "label_peak_error_noisy")
    label_peak_error_pred = metric_values(rows, "label_peak_error_pred")
    label_peak_error_improvement = metric_values(rows, "label_peak_error_improvement")

    mean_noisy_spectrum_mae = float(np.mean(noisy_spectrum_mae))
    mean_pred_spectrum_mae = float(np.mean(pred_spectrum_mae))
    mean_noisy_peak_error = float(np.mean(noisy_peak_error))
    mean_pred_peak_error = float(np.mean(pred_peak_error))
    mean_clean_noise_floor = float(np.mean(clean_noise_floor))
    mean_pred_noise_floor = float(np.mean(pred_noise_floor))
    mean_label_peak_clean_mean = safe_nanmean(label_peak_clean_mean)
    mean_label_peak_noisy_mean = safe_nanmean(label_peak_noisy_mean)
    mean_label_peak_pred_mean = safe_nanmean(label_peak_pred_mean)
    mean_label_peak_error_noisy = safe_nanmean(label_peak_error_noisy)
    mean_label_peak_error_pred = safe_nanmean(label_peak_error_pred)
    valid_label_mask = ~np.isnan(label_peak_clean_mean)

    return {
        "mean_noisy_spectrum_mae": mean_noisy_spectrum_mae,
        "mean_pred_spectrum_mae": mean_pred_spectrum_mae,
        "aggregate_spectrum_mae_improvement_ratio": safe_ratio(
            mean_noisy_spectrum_mae - mean_pred_spectrum_mae,
            mean_noisy_spectrum_mae,
        ),
        "median_spectrum_mae_improvement_ratio": float(np.median(spectrum_ratio)),
        "spectrum_improved_rate": float(np.mean(pred_spectrum_mae < noisy_spectrum_mae)),
        "mean_noisy_peak_error": mean_noisy_peak_error,
        "mean_pred_peak_error": mean_pred_peak_error,
        "aggregate_peak_error_improvement_ratio": safe_ratio(
            mean_noisy_peak_error - mean_pred_peak_error,
            mean_noisy_peak_error,
        ),
        "median_peak_error_improvement_ratio": float(np.median(peak_ratio)),
        "peak_improved_rate": float(np.mean(pred_peak_error < noisy_peak_error)),
        "mean_clean_noise_floor": mean_clean_noise_floor,
        "mean_pred_noise_floor": mean_pred_noise_floor,
        "pred_to_clean_noise_floor_ratio": safe_ratio(mean_pred_noise_floor, mean_clean_noise_floor),
        "mean_pred_peak_keep_ratio": float(np.mean(pred_peak_keep_ratio)),
        "median_pred_peak_keep_ratio": float(np.median(pred_peak_keep_ratio)),
        "valid_label_peak_sample_count": int(np.sum(valid_label_mask)),
        "valid_label_peak_sample_rate": float(np.mean(valid_label_mask)),
        "mean_label_peak_clean_mean": mean_label_peak_clean_mean,
        "mean_label_peak_noisy_mean": mean_label_peak_noisy_mean,
        "mean_label_peak_pred_mean": mean_label_peak_pred_mean,
        "aggregate_label_pred_peak_keep_ratio": safe_ratio(
            mean_label_peak_pred_mean,
            mean_label_peak_clean_mean,
        ),
        "mean_label_pred_peak_keep_ratio": safe_nanmean(label_pred_peak_keep_ratio),
        "median_label_pred_peak_keep_ratio": safe_nanmedian(label_pred_peak_keep_ratio),
        "aggregate_label_noisy_peak_keep_ratio": safe_ratio(
            mean_label_peak_noisy_mean,
            mean_label_peak_clean_mean,
        ),
        "mean_label_noisy_peak_keep_ratio": safe_nanmean(label_noisy_peak_keep_ratio),
        "mean_label_peak_error_noisy": mean_label_peak_error_noisy,
        "mean_label_peak_error_pred": mean_label_peak_error_pred,
        "mean_label_peak_error_improvement": safe_nanmean(label_peak_error_improvement),
        "label_peak_error_improved_rate": float(
            np.mean(label_peak_error_pred[valid_label_mask] < label_peak_error_noisy[valid_label_mask])
        )
        if np.any(valid_label_mask)
        else float("nan"),
    }
