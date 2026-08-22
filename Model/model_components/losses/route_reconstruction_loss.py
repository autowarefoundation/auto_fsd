"""Route representation-retention loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    logits_fp32 = logits.float()
    target_fp32 = target.float()
    probabilities = logits_fp32.sigmoid()
    intersection = (probabilities * target_fp32).sum(dim=(1, 2))
    denominator = (probabilities + target_fp32).sum(dim=(1, 2))
    return 1.0 - (2.0 * intersection + epsilon) / (
        denominator + epsilon
    )


def _destination_heatmap_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    logits_fp32 = logits.float()
    target_fp32 = target.float()
    probabilities = logits_fp32.sigmoid()
    positive = target >= 1.0 - 1e-4
    negative = ~positive
    negative_weights = (1.0 - target_fp32).pow(4)
    positive_loss = (
        -F.logsigmoid(logits_fp32)
        * (1.0 - probabilities).pow(2)
        * positive
    ).sum(dim=(1, 2))
    negative_loss = (
        -F.logsigmoid(-logits_fp32)
        * probabilities.pow(2)
        * negative_weights
        * negative
    ).sum(dim=(1, 2))
    positive_count = positive.sum(dim=(1, 2)).clamp_min(1).float()
    negative_count = negative.sum(dim=(1, 2)).clamp_min(1).float()
    return (
        positive_loss / positive_count
        + negative_loss / negative_count
    )


class RouteReconstructionLoss(nn.Module):
    """Corridor BCE/Dice plus destination heatmap focal loss."""

    corridor_pos_weight: torch.Tensor

    def __init__(
        self,
        *,
        corridor_pos_weight: float = 1.0,
        destination_weight: float = 0.25,
        dice_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if corridor_pos_weight < 1.0:
            raise ValueError("corridor_pos_weight must be >= 1")
        if destination_weight < 0.0:
            raise ValueError("destination_weight must be non-negative")
        if dice_epsilon <= 0.0:
            raise ValueError("dice_epsilon must be positive")
        self.register_buffer(
            "corridor_pos_weight",
            torch.tensor(float(corridor_pos_weight)),
        )
        self.destination_weight = float(destination_weight)
        self.dice_epsilon = float(dice_epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        channel_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(logits, target, channel_valid)["total"]

    def components(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        channel_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return total and per-channel loss terms in FP32."""
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError("route logits must have shape [B,2,H,W]")
        if target.shape != logits.shape:
            raise ValueError("route target must match logits")
        if channel_valid.shape != logits.shape[:2]:
            raise ValueError("channel_valid must have shape [B,2]")
        logits_fp32 = logits.to(dtype=torch.float32)
        target_fp32 = target.to(device=logits.device, dtype=torch.float32)
        valid = channel_valid.to(device=logits.device, dtype=torch.bool)
        zero = logits_fp32.sum() * 0.0
        if not bool(valid.any()):
            return {
                "total": zero,
                "corridor_bce": zero,
                "corridor_dice": zero,
                "destination_focal": zero,
            }

        sample_losses = logits_fp32.new_zeros(
            logits.shape[0],
        )
        sample_terms = logits_fp32.new_zeros(
            logits.shape[0],
        )
        corridor_bce_loss = zero
        corridor_dice_loss = zero
        destination_focal_loss = zero

        corridor_valid = valid[:, 0]
        if bool(corridor_valid.any()):
            corridor_logits = logits_fp32[corridor_valid, 0]
            corridor_target = target_fp32[corridor_valid, 0]
            corridor_bce = F.binary_cross_entropy_with_logits(
                corridor_logits,
                corridor_target,
                pos_weight=self.corridor_pos_weight.float(),
                reduction="none",
            ).mean(dim=(1, 2))
            corridor_dice = _soft_dice(
                corridor_logits,
                corridor_target,
                epsilon=self.dice_epsilon,
            )
            sample_losses[corridor_valid] += (
                0.5 * corridor_bce + 0.5 * corridor_dice
            )
            sample_terms[corridor_valid] += 1.0
            corridor_bce_loss = corridor_bce.mean()
            corridor_dice_loss = corridor_dice.mean()

        destination_valid = valid[:, 1]
        if bool(destination_valid.any()):
            destination_loss = _destination_heatmap_focal(
                logits_fp32[destination_valid, 1],
                target_fp32[destination_valid, 1],
            )
            sample_losses[destination_valid] += (
                self.destination_weight * destination_loss
            )
            sample_terms[destination_valid] += self.destination_weight
            destination_focal_loss = destination_loss.mean()

        active = sample_terms > 0
        total = sample_losses[active].mean() if bool(active.any()) else zero
        return {
            "total": total,
            "corridor_bce": corridor_bce_loss,
            "corridor_dice": corridor_dice_loss,
            "destination_focal": destination_focal_loss,
        }
