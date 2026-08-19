"""A/B: confidence-scaled coupling vs unscaled (always-1) (#110 / PR #193).

Trains two Combined-style mock models on the same batch:
  * ``scaled`` — current PR (reasoning confidence multiplies the residual)
  * ``unscaled`` — pre-#110 (confidence forced to 1, residual always full)

Reports open-loop ADE/FDE vs the imitation target after the same step count,
plus a post-hoc sweep of confidence 0 vs 1 on the scaled model (the safety
loop: low confidence should sit closer to the unmodulated plan).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from evaluation.metrics import compute_open_loop_metrics
from training.losses.horizon_reasoning_loss import HorizonReasoningLoss
from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.targets import (
    collate_reasoning_targets,
    record_to_target_tensors,
)
from data_processing.reasoning_label_generation.teacher_client import TeacherRequest


def _split_controls(traj: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    t = traj.detach().float().cpu().numpy()
    if t.ndim == 3 and t.shape[-1] == 2:
        return t[..., 0], t[..., 1]
    if t.ndim == 2 and t.shape[1] % 2 == 0:
        paired = t.reshape(t.shape[0], -1, 2)
        return paired[..., 0], paired[..., 1]
    raise ValueError(f"unrecognised trajectory shape {t.shape}")


def ade_fde_vs_target(pred: torch.Tensor, target: torch.Tensor, speed: float = 5.0) -> dict[str, float]:
    pa, pc = _split_controls(pred)
    ta, tc = _split_controls(target)
    v0 = np.full(pa.shape[0], speed)
    return compute_open_loop_metrics(pa, pc, ta, tc, v0)


def _patch_forced_confidence(model: torch.nn.Module, value: torch.Tensor | None) -> Callable[[], None]:
    planner = model.Reactive_E2E.TrajectoryPlanner
    orig = planner.forward

    def wrapped(*args: Any, **kwargs: Any):
        kwargs["reasoning_confidence"] = value
        return orig(*args, **kwargs)

    planner.forward = wrapped  # type: ignore[method-assign]

    def restore() -> None:
        planner.forward = orig  # type: ignore[method-assign]

    return restore


def _batch(device: torch.device, b: int = 2, v: int = 6) -> dict[str, torch.Tensor]:
    return {
        "visual": torch.randn(b, v, 3, 256, 256, device=device),
        "map_input": torch.randn(b, 3, 256, 256, device=device),
        "vis_hist": torch.zeros(b, 896, device=device),
        "ego": torch.randn(b, 256, device=device),
        "target": torch.randn(b, 128, device=device),
    }


def _reasoning_targets(b: int):
    teacher = MockTeacher()
    per = [record_to_target_tensors(teacher.label(TeacherRequest(f"s{i}", "l2d"))) for i in range(b)]
    return collate_reasoning_targets(per)


def _train(model: torch.nn.Module, batch: dict[str, torch.Tensor], steps: int, lr: float) -> list[float]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    traj_loss_fn = torch.nn.SmoothL1Loss()
    reason_loss_fn = HorizonReasoningLoss()
    tb = _reasoning_targets(batch["visual"].shape[0])
    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        traj, aux_or_pred = model(
            batch["visual"], batch["map_input"], batch["vis_hist"], batch["ego"],
            mode="train", trajectory_target=batch["target"],
        )
        # ReactiveE2E returns (traj, reasoning_pred) when reasoning is on.
        if isinstance(aux_or_pred, dict):
            pred = aux_or_pred["reasoning_pred"]
        else:
            pred = aux_or_pred
        loss = traj_loss_fn(traj, batch["target"])
        terms = reason_loss_fn(pred, tb.targets, source_weights=tb.source_weights,
                               confidence_targets=tb.confidence_targets)
        total = loss + 0.5 * terms["total"]
        total.backward()
        opt.step()
        history.append(float(total.detach()))
    return history


@torch.no_grad()
def _eval_ade(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    out = model(
        batch["visual"], batch["map_input"], batch["vis_hist"], batch["ego"],
        mode="infer",
    )
    traj = out[0] if isinstance(out, tuple) else out
    return ade_fde_vs_target(traj, batch["target"])


def run_confidence_coupling_ab(
    build_model,
    *,
    device: torch.device | None = None,
    steps: int = 10,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Train scaled vs unscaled coupling; return ADE/FDE and confidence sweep."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    batch = _batch(device)
    b = batch["visual"].shape[0]

    scaled = build_model(
        num_views=6, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    hist_s = _train(scaled, batch, steps, lr)
    metrics_s = _eval_ade(scaled, batch)

    torch.manual_seed(seed)
    batch_u = _batch(device)
    unscaled = build_model(
        num_views=6, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    ones = torch.ones(b, device=device)
    restore = _patch_forced_confidence(unscaled, ones)
    try:
        hist_u = _train(unscaled, batch_u, steps, lr)
        metrics_u = _eval_ade(unscaled, batch_u)
    finally:
        restore()

    # Safety-loop sweep on the scaled model: conf=0 vs conf=1.
    zeros = torch.zeros(b, device=device)
    r0 = _patch_forced_confidence(scaled, zeros)
    try:
        ade_c0 = _eval_ade(scaled, batch)
    finally:
        r0()
    r1 = _patch_forced_confidence(scaled, ones)
    try:
        ade_c1 = _eval_ade(scaled, batch)
    finally:
        r1()

    return {
        "train_steps": steps,
        "scaled": {**metrics_s, "loss_first": hist_s[0], "loss_last": hist_s[-1]},
        "unscaled": {**metrics_u, "loss_first": hist_u[0], "loss_last": hist_u[-1]},
        "scaled_conf0": ade_c0,
        "scaled_conf1": ade_c1,
        "ade3s_delta_scaled_minus_unscaled": (
            metrics_s["ADE@3s"] - metrics_u["ADE@3s"]
        ),
        "ade3s_delta_conf1_minus_conf0": (
            ade_c1["ADE@3s"] - ade_c0["ADE@3s"]
        ),
    }
