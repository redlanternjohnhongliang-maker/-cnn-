"""ARIM .mat 随机切分脚本。

v0 数据入口：从 arim.mat / arim_smoke.mat 读取原始复数信号，
随机打乱后按比例保存为和原始 process.py 一致的 .npy 字段格式。
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import scipy.io


REQUIRED_MAT_KEYS = ["sb_mat", "sb0_mat", "amplitude_mat", "distance_mat", "info_mat"]


def parse_args():
    parser = argparse.ArgumentParser(description="ARIM .mat 随机切分为 train/test .npy")
    parser.add_argument("--mat_path", required=True, help="arim.mat 或 arim_smoke.mat 文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录，通常是原仓库 training 目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--seed", type=int, default=707, help="随机种子，默认 707")
    return parser.parse_args()


def _check_inputs(mat_data: Dict[str, np.ndarray], train_ratio: float) -> int:
    missing = [key for key in REQUIRED_MAT_KEYS if key not in mat_data]
    if missing:
        raise KeyError(f".mat 文件缺少字段: {missing}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train_ratio 必须在 0 和 1 之间")

    sample_count = mat_data["sb_mat"].shape[0]
    for key in REQUIRED_MAT_KEYS:
        if mat_data[key].shape[0] != sample_count:
            raise ValueError(f"字段 {key} 样本数不一致: {mat_data[key].shape[0]} != {sample_count}")
    return sample_count


def _build_dataset(mat_data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """保持和原始 process.py 一致的字段命名。"""
    return {
        "sb0": mat_data["sb0_mat"][indices],
        "sb": mat_data["sb_mat"][indices],
        "amplitudes": mat_data["amplitude_mat"][indices],
        "distances": mat_data["distance_mat"][indices],
        "info_mat": mat_data["info_mat"][indices],
    }


def random_split_arim_mat(
    mat_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 707,
) -> Tuple[str, str]:
    """读取 ARIM .mat，随机切分并保存为 arim_train_random.npy / arim_test_random.npy。"""
    mat_data = scipy.io.loadmat(mat_path)
    sample_count = _check_inputs(mat_data, train_ratio)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    train_count = int(sample_count * train_ratio)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    train_dataset = _build_dataset(mat_data, train_indices)
    test_dataset = _build_dataset(mat_data, test_indices)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "arim_train_random.npy")
    test_path = os.path.join(output_dir, "arim_test_random.npy")
    np.save(train_path, train_dataset)
    np.save(test_path, test_dataset)

    print(f"总样本数: {sample_count}")
    print(f"训练集: {len(train_indices)} -> {train_path}")
    print(f"测试集: {len(test_indices)} -> {test_path}")
    print(f"随机种子: {seed}, train_ratio: {train_ratio}")
    return train_path, test_path


def main():
    args = parse_args()
    random_split_arim_mat(
        mat_path=args.mat_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
