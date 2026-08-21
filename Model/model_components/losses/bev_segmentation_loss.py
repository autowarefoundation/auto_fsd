"""Masked multi-label BEV segmentation auxiliary loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVSegmentationAuxiliaryLoss(nn.Module):
    """FP32 mixture of class-balanced BCE and per-sample Soft Dice."""

    pos_weight: torch.Tensor

    def __init__(
        self,
        pos_weight: Sequence[float] | torch.Tensor,
        *,
        dice_epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(pos_weight, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("pos_weight must be a non-empty 1D sequence")
        if not torch.isfinite(weights).all() or bool((weights < 1.0).any()):
            raise ValueError("pos_weight entries must be finite and >= 1")
        if dice_epsilon <= 0.0:
            raise ValueError("dice_epsilon must be positive")
        self.register_buffer("pos_weight", weights)
        self.dice_epsilon = float(dice_epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [B,C,H,W]")
        if target.shape != logits.shape or valid_mask.shape != logits.shape:
            raise ValueError("target and valid_mask must match logits")
        if logits.shape[1] != self.pos_weight.numel():
            raise ValueError("logit channels differ from pos_weight")
        logits_fp32 = logits.to(dtype=torch.float32)
        target_fp32 = target.to(device=logits.device, dtype=torch.float32)
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)
        active = valid.any(dim=(2, 3))
        if not bool(active.any()):
            return logits_fp32.sum() * 0.0

        with torch.autocast(
            device_type=logits.device.type,
            enabled=False,
        ):
            mask = valid.to(torch.float32)
            bce = F.binary_cross_entropy_with_logits(
                logits_fp32,
                target_fp32,
                pos_weight=self.pos_weight.view(1, -1, 1, 1),
                reduction="none",
            )
            valid_counts = mask.sum(dim=(2, 3)).clamp_min(1.0)
            sample_bce = (bce * mask).sum(dim=(2, 3)) / valid_counts

            probabilities = logits_fp32.sigmoid()
            intersection = (
                probabilities * target_fp32 * mask
            ).sum(dim=(2, 3))
            denominator = (
                (probabilities + target_fp32) * mask
            ).sum(dim=(2, 3))
            sample_dice = 1.0 - (
                2.0 * intersection + self.dice_epsilon
            ) / (denominator + self.dice_epsilon)
            sample_loss = 0.5 * sample_bce + 0.5 * sample_dice
            return sample_loss[active].mean()
