"""CLI: navigation-input robustness matrix (#157) on synthetic or saved tensors.

Example (synthetic smoke)::

    python -m evaluation.navigation_robustness_cli --synthetic
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from evaluation.navigation_robustness import (
    DEFAULT_MODES,
    run_navigation_robustness,
)


def _synthetic_predict(map_context: np.ndarray, route_mask: np.ndarray) -> np.ndarray:
    b = map_context.shape[0]
    t = 16
    # Use route mass as a crude "knows where to go" signal.
    route_mass = route_mask.reshape(b, -1).sum(axis=-1, keepdims=True)  # [B,1]
    map_mass = map_context.reshape(b, -1).sum(axis=-1, keepdims=True)
    speed = 0.5 + 0.5 * (route_mass > 0).astype(np.float64) + 0.25 * (map_mass > 0)
    xs = np.linspace(0, 1, t)[None, :] * speed  # [B,T]
    ys = np.zeros_like(xs)
    return np.stack([xs, ys], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic tensors")
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Pass --synthetic (KITScenes wiring comes in a follow-up).")

    rng = np.random.default_rng(args.seed)
    b, t = 8, 16
    map_context = rng.random((b, 3, 32, 32), dtype=np.float32)
    route_mask = rng.random((b, 2, 32, 32), dtype=np.float32)
    gt = np.stack(
        [np.stack([np.linspace(0, 8, t), np.zeros(t)], axis=-1) for _ in range(b)],
        axis=0,
    )
    report = run_navigation_robustness(
        map_context, route_mask, gt, _synthetic_predict, modes=tuple(args.modes), seed=args.seed
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
