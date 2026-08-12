"""Fidelity-aware reward prototype for stage-3 closed-loop RL (issue #123).

Design rule from the issue: every reward term must add information the
reactive policy cannot already infer from its own inputs. A reward built
only from the reasoning band (a re-encoding of planner inputs) is redundant
for the same DPI reason that drove ``reasoning_coupling.alpha → 0``.

This module therefore shapes rewards with **world-model fidelity** — how
well the WM predicts consequences of the rolled-out action — and optionally
scores those predicted consequences against an external preference signal.
Both are information the camera-only reactive policy does not observe at
decision time.

This is deliberately a pure function / small helper, not a training-loop
integration. Wire it behind ``compute_planner_loss`` (#115) when that hook
lands; until then it is safe to unit-test and prototype against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


FIDELITY_AWARE_REWARD_VERSION = "wm_fidelity_gate_v1"


@dataclass(frozen=True)
class FidelityAwareRewardResult:
    """Per-sample reward and diagnostic scalars."""

    reward: torch.Tensor
    fidelity: torch.Tensor
    base_reward: torch.Tensor
    consequence_reward: torch.Tensor | None
    metadata: dict[str, float]


def world_model_fidelity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduce_dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Map WM prediction error to a ``(0, 1]`` fidelity weight.

    ``fidelity = exp(-mse / temperature)``. High fidelity means the world
    model is a trustworthy source of consequence information for this sample;
    low fidelity down-weights any reward that depends on those predictions.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    err = (prediction.float() - target.float()).pow(2)
    if reduce_dims is None:
        # Reduce all but batch dim when present.
        if err.ndim == 0:
            mse = err
        else:
            mse = err.flatten(1).mean(dim=1)
    else:
        mse = err.mean(dim=reduce_dims)

    return torch.exp(-mse / temperature)


def consequence_alignment_reward(
    predicted_future: torch.Tensor,
    preferred_future: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward from WM-predicted futures vs an external preference target.

    ``preferred_future`` is intended to carry information outside the
    reactive policy's inputs (e.g. HD-map / LiDAR conflict geometry, or a
    human preference embedding) — not a re-encoding of BEV/ego already fed
    to the planner.
    """
    if predicted_future.shape != preferred_future.shape:
        raise ValueError(
            "predicted_future/preferred_future shape mismatch: "
            f"{tuple(predicted_future.shape)} vs {tuple(preferred_future.shape)}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    # Negative MSE so better alignment → higher reward. Keep per-batch.
    if predicted_future.ndim == 0:
        mse = (predicted_future.float() - preferred_future.float()).pow(2)
    else:
        mse = (
            (predicted_future.float() - preferred_future.float())
            .pow(2)
            .flatten(1)
            .mean(dim=1)
        )
    return -scale * mse


def fidelity_aware_reward(
    base_reward: torch.Tensor,
    *,
    wm_prediction: torch.Tensor,
    wm_target: torch.Tensor,
    preferred_future: torch.Tensor | None = None,
    predicted_future: torch.Tensor | None = None,
    fidelity_temperature: float = 1.0,
    consequence_scale: float = 1.0,
    consequence_weight: float = 1.0,
    min_fidelity: float = 0.0,
) -> FidelityAwareRewardResult:
    """Gate ``base_reward`` (and optional consequence term) by WM fidelity.

    Final reward::

        r = fidelity * (base_reward + consequence_weight * consequence)

    where ``fidelity`` is clipped below by ``min_fidelity`` so a broken WM
    cannot zero the entire objective if a floor is desired.
    """
    if base_reward.ndim > 1:
        raise ValueError(
            f"base_reward must be scalar or (B,), got shape {tuple(base_reward.shape)}"
        )

    fidelity = world_model_fidelity(
        wm_prediction,
        wm_target,
        temperature=fidelity_temperature,
    )
    if fidelity.shape != base_reward.shape and not (
        fidelity.ndim == 1 and base_reward.ndim == 0
    ):
        # Allow broadcasting scalar base over batch fidelity.
        if base_reward.ndim == 0 and fidelity.ndim == 1:
            base_reward = base_reward.expand_as(fidelity)
        elif fidelity.ndim == 0 and base_reward.ndim == 1:
            fidelity = fidelity.expand_as(base_reward)
        elif fidelity.shape != base_reward.shape:
            raise ValueError(
                "fidelity/base_reward shape mismatch after reduction: "
                f"{tuple(fidelity.shape)} vs {tuple(base_reward.shape)}"
            )

    fidelity = fidelity.clamp(min=float(min_fidelity), max=1.0)

    consequence: torch.Tensor | None = None
    total = base_reward.float()
    if preferred_future is not None:
        if predicted_future is None:
            raise ValueError(
                "predicted_future is required when preferred_future is provided"
            )
        consequence = consequence_alignment_reward(
            predicted_future,
            preferred_future,
            scale=consequence_scale,
        )
        if consequence.shape != total.shape:
            raise ValueError(
                "consequence/base_reward shape mismatch: "
                f"{tuple(consequence.shape)} vs {tuple(total.shape)}"
            )
        total = total + float(consequence_weight) * consequence

    reward = fidelity * total
    metadata = {
        "fidelity_mean": float(fidelity.detach().mean().cpu()),
        "base_reward_mean": float(base_reward.detach().float().mean().cpu()),
        "reward_mean": float(reward.detach().mean().cpu()),
        "fidelity_temperature": float(fidelity_temperature),
        "consequence_weight": float(consequence_weight),
        "min_fidelity": float(min_fidelity),
    }
    if consequence is not None:
        metadata["consequence_reward_mean"] = float(
            consequence.detach().mean().cpu()
        )

    return FidelityAwareRewardResult(
        reward=reward,
        fidelity=fidelity,
        base_reward=base_reward.float(),
        consequence_reward=consequence,
        metadata=metadata,
    )


def soft_advantage_from_reward(
    reward: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Optional REINFORCE-style advantage: ``(r - baseline) / temperature``.

    Kept separate from :func:`fidelity_aware_reward` so callers can plug the
    gated reward into whatever policy-gradient estimator they choose once
    ``compute_planner_loss`` (#115) is restored.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if baseline is None:
        baseline = reward.detach().mean()
    return (reward - baseline) / temperature
