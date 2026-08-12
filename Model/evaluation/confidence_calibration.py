"""Calibration metrics for reasoning-band confidence (#110)."""

from __future__ import annotations

from typing import Dict

import torch


def expected_calibration_error(
    confidence: torch.Tensor,
    correctness: torch.Tensor,
    *,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute ECE between predicted confidence and binary correctness.

    Args:
        confidence: probabilities in ``[0, 1]``, any shape (flattened).
        correctness: same shape, values in ``{0, 1}`` (or soft ``[0, 1]``).
        n_bins: number of equal-width confidence bins.

    Returns:
        Dict with ``ece``, ``n``, and per-bin ``bin_confidence`` / ``bin_accuracy``
        lists (empty bins omitted from the lists but counted in normalization).
    """
    conf = confidence.detach().float().reshape(-1).clamp(0.0, 1.0)
    corr = correctness.detach().float().reshape(-1)
    if conf.numel() != corr.numel():
        raise ValueError("confidence and correctness must have the same number of elements")
    if conf.numel() == 0:
        return {"ece": 0.0, "n": 0, "bin_confidence": [], "bin_accuracy": []}

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    ece = conf.new_zeros(())
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    n = conf.numel()
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        avg_conf = conf[mask].mean()
        avg_acc = corr[mask].mean()
        ece = ece + (count / n) * (avg_conf - avg_acc).abs()
        bin_conf.append(float(avg_conf))
        bin_acc.append(float(avg_acc))
    return {
        "ece": float(ece),
        "n": n,
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
    }
