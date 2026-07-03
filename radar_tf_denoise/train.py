"""训练带干扰时频图 -> 干净时频图的小型 U-Net。"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_loader import RadarTFDataset
from losses import build_loss
from model_unet import SmallUNet
from stft_utils import TFConfig, expected_tf_shape


def parse_args():
    parser = argparse.ArgumentParser(description="ARIM 时频图去干扰训练")
    parser.add_argument("--train_path", required=True, help="arim_train.npy 路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--loss_type",
        choices=["l1", "weighted_l1", "complex_l1_mag", "complex_weighted_mag", "complex_weighted_mag_mask"],
        default="l1",
        help="损失函数类型",
    )
    parser.add_argument("--peak_weight_alpha", type=float, default=3.0, help="weighted_l1 的目标峰权重系数")
    parser.add_argument("--mag_loss_beta", type=float, default=0.5, help="复数幅度损失权重")
    parser.add_argument("--mask_loss_weight", type=float, default=5.0, help="v1.3 mask 内复数 L1 损失权重")
    parser.add_argument("--save_dir", default=None, help="模型保存目录；不传时按 loss_type 自动选择")
    parser.add_argument("--init_from_v11", default=None, help="用 v1.1 checkpoint 初始化 v1.4 的可兼容权重")
    parser.add_argument(
        "--mode",
        choices=["magnitude", "complex", "complex_mask_residual", "complex_mask_clean"],
        default="magnitude",
        help="输入输出模式",
    )
    return parser.parse_args()


def resolve_save_dir(script_dir, mode, loss_type, save_dir):
    """解析模型保存目录。

    普通 l1 默认仍保存到 checkpoints；weighted_l1 默认保存到 checkpoints_v01，
    complex 默认保存到 checkpoints_v1，v1.1 默认保存到 checkpoints_v11，避免覆盖 v0/v0.1/v1。
    """
    if save_dir is None:
        if mode == "complex_mask_clean":
            dirname = "checkpoints_v14_full"
        elif mode == "complex_mask_residual":
            dirname = "checkpoints_v13_full" if loss_type == "complex_weighted_mag_mask" else "checkpoints_v12_smoke"
        elif mode == "complex":
            dirname = "checkpoints_v11" if loss_type == "complex_weighted_mag" else "checkpoints_v1"
        elif loss_type == "weighted_l1":
            dirname = "checkpoints_v01"
        else:
            dirname = "checkpoints"
        return os.path.join(script_dir, dirname)
    if os.path.isabs(save_dir):
        return save_dir
    return os.path.join(script_dir, save_dir)


def init_model_from_v11(model, checkpoint_path, device):
    """用 v1.1 的 2 通道复数模型初始化 v1.4 的 3 通道输入模型。

    第一层卷积多出的 mask 通道权重置 0；其余 shape 完全一致的层直接复制。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    copied = []
    skipped = []

    for key, target_value in target_state.items():
        if key not in source_state:
            skipped.append((key, "v1.1 checkpoint 中不存在"))
            continue

        source_value = source_state[key]
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            copied.append(key)
            continue

        if (
            key == "inc.net.0.weight"
            and len(source_value.shape) == 4
            and len(target_value.shape) == 4
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] == 2
            and target_value.shape[1] == 3
            and source_value.shape[2:] == target_value.shape[2:]
        ):
            new_weight = target_value.clone()
            new_weight[:, :2] = source_value
            new_weight[:, 2:3] = 0.0
            target_state[key] = new_weight
            copied.append(f"{key} 前两个通道复制，mask 通道置 0")
            continue

        skipped.append((key, f"shape 不一致: v1.1 {tuple(source_value.shape)} -> 当前 {tuple(target_value.shape)}"))

    model.load_state_dict(target_state)
    print(f"v1.1 初始化完成: 复制 {len(copied)} 项，跳过 {len(skipped)} 项")
    for item in skipped[:20]:
        print(f"  跳过 {item[0]}: {item[1]}")
    if len(skipped) > 20:
        print(f"  其余跳过项数量: {len(skipped) - 20}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, mode: str, loss_type: str):
    model.train(train)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        if mode in {"complex_mask_residual", "complex_mask_clean"}:
            x, y, noisy_channels, mask = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            noisy_channels = noisy_channels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if mode == "complex_mask_residual":
                # v1.2: 网络只预测 residual，最终输出只在 CFAR mask 区域修复。
                residual = model(x)
                pred = noisy_channels + mask * residual
            elif mode == "complex_mask_clean":
                # v1.4: 网络直接预测 clean STFT；非 mask 区域保留原始 noisy STFT。
                clean_pred = model(x)
                pred = mask * clean_pred + (1.0 - mask) * noisy_channels
            else:
                pred = model(x)
            if mode == "complex_mask_residual" and loss_type == "complex_weighted_mag_mask":
                # v1.3: 在全局 complex_weighted_mag 外，额外强化 mask 内复数重构。
                loss = criterion(pred, y, mask)
            else:
                loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def main():
    args = parse_args()
    random.seed(707)
    np.random.seed(707)
    torch.manual_seed(707)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    cfg = TFConfig()
    print(f"STFT 输出尺寸 H,W = {expected_tf_shape(1024, cfg)}")
    dataset = RadarTFDataset(args.train_path, max_samples=args.max_samples, tf_config=cfg, mode=args.mode)

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(707),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.mode in {"complex_mask_residual", "complex_mask_clean"}:
        in_channels = 3
        out_channels = 2
    elif args.mode == "complex":
        in_channels = 2
        out_channels = 2
    else:
        in_channels = 1
        out_channels = 1
    model = SmallUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    if args.mode in {"complex", "complex_mask_residual", "complex_mask_clean"}:
        effective_loss_type = args.loss_type if args.loss_type.startswith("complex_") else "complex_l1_mag"
    else:
        if args.loss_type.startswith("complex_"):
            raise ValueError("magnitude 模式不能使用 complex_* 损失")
        effective_loss_type = args.loss_type
    if effective_loss_type == "complex_weighted_mag_mask" and args.mode != "complex_mask_residual":
        raise ValueError("complex_weighted_mag_mask 只能用于 complex_mask_residual 模式")
    criterion = build_loss(
        effective_loss_type,
        peak_weight_alpha=args.peak_weight_alpha,
        mag_loss_beta=args.mag_loss_beta,
        mask_loss_weight=args.mask_loss_weight,
    )
    if args.init_from_v11:
        init_model_from_v11(model, args.init_from_v11, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = resolve_save_dir(script_dir, args.mode, args.loss_type, args.save_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    best_val = float("inf")

    print(f"训练样本数: {train_size}, 验证样本数: {val_size}, device: {device}")
    print(f"mode: {args.mode}, in_channels: {in_channels}, out_channels: {out_channels}")
    print(
        f"loss_type: {effective_loss_type}, peak_weight_alpha: {args.peak_weight_alpha}, "
        f"mag_loss_beta: {args.mag_loss_beta}, mask_loss_weight: {args.mask_loss_weight}"
    )
    print(f"模型保存目录: {checkpoint_dir}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            mode=args.mode,
            loss_type=effective_loss_type,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            mode=args.mode,
            loss_type=effective_loss_type,
        )
        print(f"Epoch {epoch}/{args.epochs} | train {effective_loss_type}: {train_loss:.6f} | val {effective_loss_type}: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tf_config": cfg.__dict__,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "mode": args.mode,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "loss_type": effective_loss_type,
                    "peak_weight_alpha": args.peak_weight_alpha,
                    "mag_loss_beta": args.mag_loss_beta,
                    "mask_loss_weight": args.mask_loss_weight,
                    "init_from_v11": args.init_from_v11,
                },
                best_path,
            )
            print(f"保存最优模型: {best_path}")


if __name__ == "__main__":
    main()
