"""v0/v0.1 训练损失函数。"""

import torch
from torch import nn


class WeightedL1Loss(nn.Module):
    """加权 L1 损失。

    v0.1 用干净时频图 clean 的高能量区域生成权重，让模型更重视目标峰附近：
        clean_norm = clean / (clean.max() + eps)
        weight = 1 + alpha * clean_norm
        loss = mean(weight * abs(pred - clean))
    """

    def __init__(self, alpha: float = 3.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        clean_norm = clean / (torch.amax(clean).detach() + self.eps)
        weight = 1.0 + self.alpha * clean_norm
        return torch.mean(weight * torch.abs(pred - clean))


class ComplexSTFTLoss(nn.Module):
    """v1 复数两通道损失。

    第一项约束实部/虚部，第二项约束复数幅度，避免只拟合相位或符号而忽略能量结构。
    """

    def __init__(self, magnitude_weight: float = 0.5) -> None:
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        ri_loss = self.l1(pred, clean)
        pred_mag = torch.sqrt(torch.clamp(pred[:, 0] ** 2 + pred[:, 1] ** 2, min=0.0))
        clean_mag = torch.sqrt(torch.clamp(clean[:, 0] ** 2 + clean[:, 1] ** 2, min=0.0))
        mag_loss = self.l1(pred_mag, clean_mag)
        return ri_loss + self.magnitude_weight * mag_loss


class ComplexWeightedMagnitudeLoss(nn.Module):
    """v1.1 复数两通道加权幅度损失。

    保留实部/虚部 L1，同时用干净 STFT 幅度生成权重，让高能量目标区域在幅度损失中占更大比重。
    """

    def __init__(self, alpha: float = 3.0, beta: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        ri_loss = self.l1(pred, clean)

        pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + self.eps)
        clean_mag = torch.sqrt(clean[:, 0] ** 2 + clean[:, 1] ** 2 + self.eps)

        clean_mag_norm = clean_mag / (clean_mag.amax(dim=(1, 2), keepdim=True) + self.eps)
        weight = 1.0 + self.alpha * clean_mag_norm
        weighted_mag_loss = torch.mean(weight * torch.abs(pred_mag - clean_mag))

        return ri_loss + self.beta * weighted_mag_loss


class ComplexWeightedMagnitudeMaskLoss(nn.Module):
    """v1.3 复数加权幅度 + mask 内修复损失。

    global_loss 沿用 v1.2/v1.1 的 complex_weighted_mag；
    mask_complex_l1 只在 CFAR mask 区域计算实部/虚部 L1，让网络更重视干扰区域的重构质量。
    """

    def __init__(
        self,
        alpha: float = 3.0,
        beta: float = 0.5,
        mask_loss_weight: float = 5.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.global_loss = ComplexWeightedMagnitudeLoss(alpha=alpha, beta=beta, eps=eps)
        self.mask_loss_weight = mask_loss_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        global_loss = self.global_loss(pred, clean)

        # mask: [B,1,H,W]，扩展到复数两通道 [B,2,H,W] 后只统计 mask 内误差。
        mask_2ch = mask.expand_as(pred)
        mask_complex_l1 = torch.sum(mask_2ch * torch.abs(pred - clean)) / (torch.sum(mask_2ch) + self.eps)
        return global_loss + self.mask_loss_weight * mask_complex_l1


def build_loss(
    loss_type: str,
    peak_weight_alpha: float = 3.0,
    mag_loss_beta: float = 0.5,
    mask_loss_weight: float = 5.0,
) -> nn.Module:
    """根据命令行参数创建损失函数。"""
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "weighted_l1":
        return WeightedL1Loss(alpha=peak_weight_alpha)
    if loss_type == "complex_l1_mag":
        return ComplexSTFTLoss(magnitude_weight=mag_loss_beta)
    if loss_type == "complex_weighted_mag":
        return ComplexWeightedMagnitudeLoss(alpha=peak_weight_alpha, beta=mag_loss_beta)
    if loss_type == "complex_weighted_mag_mask":
        return ComplexWeightedMagnitudeMaskLoss(
            alpha=peak_weight_alpha,
            beta=mag_loss_beta,
            mask_loss_weight=mask_loss_weight,
        )
    raise ValueError(f"不支持的 loss_type: {loss_type}")
