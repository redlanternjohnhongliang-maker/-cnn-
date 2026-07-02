"""训练带干扰时频图 -> 干净时频图的小型 U-Net。"""

import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data_loader import RadarTFDataset
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
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(x)
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
    dataset = RadarTFDataset(args.train_path, max_samples=args.max_samples, tf_config=cfg)

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

    model = SmallUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    best_val = float("inf")

    print(f"训练样本数: {train_size}, 验证样本数: {val_size}, device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch}/{args.epochs} | train L1: {train_loss:.6f} | val L1: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tf_config": cfg.__dict__,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"保存最优模型: {best_path}")


if __name__ == "__main__":
    main()
