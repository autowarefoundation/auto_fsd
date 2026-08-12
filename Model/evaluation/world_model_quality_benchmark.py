#!/usr/bin/env python3
"""CLI: world-model quality benchmark (JEPA recon + Reactive vs Combined impact).

Sibling to ``Model/speed_benchmark/speed_benchmark.py``, but reports *quality*
metrics instead of FPS.

Examples::

    # Synthetic JEPA recon smoke (no checkpoint)
    python evaluation/world_model_quality_benchmark.py --synthetic

    # Model-level Reactive vs Combined trajectory impact (random init)
    python evaluation/world_model_quality_benchmark.py --impact --backbone swin_v2_tiny

Writes a JSON blob suitable for pasting into QUALITY_BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Allow running from Model/ or Model/evaluation/
_HERE = Path(__file__).resolve().parent
_MODEL_ROOT = _HERE.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from evaluation.world_model_quality import (  # noqa: E402
    jepa_reconstruction_metrics,
    null_predictor_metrics,
    relative_jepa_improvement,
    summarize_world_model_quality,
    world_model_trajectory_impact,
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_MODEL_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_synthetic_jepa(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    target = tuple(torch.randn(4, 16, 8, 8, generator=g) for _ in range(4))
    # Partially aligned prediction: target + noise
    g2 = torch.Generator().manual_seed(seed + 1)
    pred = tuple(t + 0.25 * torch.randn(t.shape, generator=g2) for t in target)
    jepa = jepa_reconstruction_metrics(pred, target)
    null = null_predictor_metrics(target)
    rel = relative_jepa_improvement(jepa, null)
    return summarize_world_model_quality(jepa={**jepa, **null, **rel})


def run_impact(backbone: str, device: torch.device, batch: int = 2, views: int = 7) -> dict:
    from model_components.auto_e2e import AutoE2E
    from model_components.view_fusion import PinholeProjection

    model = AutoE2E(
        backbone=backbone,
        num_views=views,
        view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
        enable_world_model=True,
    ).to(device)
    model.eval()

    camera = torch.randn(batch, views, 3, 256, 256, device=device)
    map_input = torch.randn(batch, 3, 256, 256, device=device)
    visual_history = torch.randn(batch, 896, device=device)
    egomotion = torch.randn(batch, 256, device=device)
    projection = PinholeProjection(torch.randn(batch, views, 3, 4, device=device))

    impact = world_model_trajectory_impact(
        model, camera, map_input, visual_history, egomotion,
        projection=projection, geometry_type="pinhole",
    )
    return summarize_world_model_quality(impact=impact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic JEPA recon smoke")
    parser.add_argument("--impact", action="store_true", help="Run Reactive vs Combined impact")
    parser.add_argument("--backbone", default="swin_v2_tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if not args.synthetic and not args.impact:
        args.synthetic = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)

    payload = {
        "schema": "auto_e2e_world_model_quality_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "metrics": {},
    }

    if args.synthetic:
        payload["metrics"].update(run_synthetic_jepa(args.seed))
    if args.impact:
        payload["metrics"].update(run_impact(args.backbone, device, args.batch))

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
