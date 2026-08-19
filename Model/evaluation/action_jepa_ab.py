"""A/B: history-only JEPA vs action-conditioned JEPA in the train loop.

Trains two Combined mock models on the same batch:
  * ``history`` — ``action_dim=None`` (current Combined default)
  * ``action``  — ``action_dim=128``, AutoE2E teacher-forces ``trajectory_target``

Reports JEPA loss after the same step count, whether ``action_proj`` left zero,
and matched-vs-shuffled action JEPA on the action-conditioned arm (the wiring
check: at init the residual is a no-op so the two JEPA values match; after
training they should split if the projection used the plan).
"""

from __future__ import annotations

from typing import Any

import torch


ACTION_DIM = 128  # 64 timesteps × 2 signals (a, κ)


def _batch(device: torch.device, b: int = 2, v: int = 6, hw: int = 256) -> dict[str, torch.Tensor]:
    return {
        "visual": torch.randn(b, v, 3, hw, hw, device=device),
        "map_input": torch.randn(b, 3, hw, hw, device=device),
        "vis_hist": torch.zeros(b, 896, device=device),
        "ego": torch.randn(b, 256, device=device),
        "target": torch.randn(b, ACTION_DIM, device=device),
        "history_frames": torch.randn(b, 4, v, 3, hw, hw, device=device),
        "future_frames": torch.randn(b, 4, v, 3, hw, hw, device=device),
    }


def _batch_from_shard(shard_dir: str, device: torch.device) -> dict[str, torch.Tensor]:
    from data_parsing.pre_extracted import make_pre_extracted_loader

    loader = make_pre_extracted_loader(shard_dir, batch_size=2, num_workers=0, shuffle=0)
    raw = next(iter(loader))
    if "history_frames" not in raw or "future_frames" not in raw:
        raise ValueError(
            f"{shard_dir} has no World-Model windows; re-pack with world_model=True"
        )
    return {
        "visual": raw["visual_tiles"].to(device),
        "map_input": raw["map_context"].to(device),
        "vis_hist": raw["visual_history"].to(device),
        "ego": raw["egomotion_history"].to(device),
        "target": raw["trajectory_target"].to(device),
        "history_frames": raw["history_frames"].to(device),
        "future_frames": raw["future_frames"].to(device),
    }


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor], actions=None):
    kw: dict[str, Any] = dict(
        mode="train",
        trajectory_target=batch["target"],
        history_frames=batch["history_frames"],
        future_frames=batch["future_frames"],
    )
    if actions is not None:
        kw["actions"] = actions
    return model(
        batch["visual"], batch["map_input"], batch["vis_hist"], batch["ego"], **kw,
    )


def _shuffled_actions(target: torch.Tensor) -> torch.Tensor:
    """A permutation that is never the identity, so B=2 cannot accidentally match."""
    b = target.shape[0]
    if b <= 1:
        return target.roll(1, dims=-1)
    return target.flip(0)


def _jepa(model: torch.nn.Module, traj_and_aux) -> torch.Tensor:
    aux = traj_and_aux[1]
    return model.World_Action_Model_E2E.jepa_loss(
        aux["future_state_pred"], aux["future_frames"],
    )


def _train(model: torch.nn.Module, batch: dict[str, torch.Tensor], steps: int, lr: float) -> list[float]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    traj_loss_fn = torch.nn.SmoothL1Loss()
    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = _forward(model, batch)
        traj = out[0]
        loss = traj_loss_fn(traj, batch["target"]) + _jepa(model, out)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def _eval_jepa(model: torch.nn.Module, batch: dict[str, torch.Tensor], actions=None) -> float:
    model.eval()
    return float(_jepa(model, _forward(model, batch, actions=actions)))


def _action_proj_norm(model: torch.nn.Module) -> float:
    proj = getattr(model.World_Action_Model_E2E.future_predictor, "action_proj", None)
    if proj is None:
        return 0.0
    return float(proj.weight.detach().pow(2).sum().sqrt())


def _pred_l1(model: torch.nn.Module, batch: dict[str, torch.Tensor], a, b) -> float:
    model.eval()
    with torch.no_grad():
        pa = _forward(model, batch, actions=a)[1]["future_state_pred"]
        pb = _forward(model, batch, actions=b)[1]["future_state_pred"]
    diffs = [((x - y).abs().mean()) for x, y in zip(pa, pb)]
    return float(torch.stack(diffs).mean())


def run_action_jepa_ab(
    build_model,
    *,
    device: torch.device | None = None,
    steps: int = 12,
    lr: float = 1e-3,
    seed: int = 0,
    shard_dir: str | None = None,
) -> dict[str, Any]:
    """Train history-only vs action-conditioned Combined; return JEPA numbers."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    batch = _batch_from_shard(shard_dir, device) if shard_dir else _batch(device)
    source = "shard" if shard_dir else "mock_batch"

    wmk_hist = {"feature_channels": 768}
    wmk_act = {"feature_channels": 768, "action_dim": ACTION_DIM}

    hist_model = build_model(
        num_views=batch["visual"].shape[1], device=device,
        enable_world_model=True, world_model_kwargs=wmk_hist,
    )
    loss_h = _train(hist_model, batch, steps, lr)
    jepa_h = _eval_jepa(hist_model, batch)

    torch.manual_seed(seed)
    batch_a = _batch_from_shard(shard_dir, device) if shard_dir else _batch(device)
    act_model = build_model(
        num_views=batch_a["visual"].shape[1], device=device,
        enable_world_model=True, world_model_kwargs=wmk_act,
    )
    jepa_init_matched = _eval_jepa(act_model, batch_a)
    shuffled0 = _shuffled_actions(batch_a["target"])
    jepa_init_shuffled = _eval_jepa(act_model, batch_a, actions=shuffled0)
    init_action_l1 = _pred_l1(act_model, batch_a, batch_a["target"], shuffled0)

    loss_a = _train(act_model, batch_a, steps, lr)
    jepa_a = _eval_jepa(act_model, batch_a)
    shuffled = _shuffled_actions(batch_a["target"])
    jepa_shuffled = _eval_jepa(act_model, batch_a, actions=shuffled)
    trained_action_l1 = _pred_l1(act_model, batch_a, batch_a["target"], shuffled)

    return {
        "train_steps": steps,
        "source": source,
        "history": {
            "jepa": jepa_h,
            "loss_first": loss_h[0],
            "loss_last": loss_h[-1],
        },
        "action": {
            "jepa": jepa_a,
            "jepa_shuffled_actions": jepa_shuffled,
            "jepa_init_matched": jepa_init_matched,
            "jepa_init_shuffled": jepa_init_shuffled,
            "loss_first": loss_a[0],
            "loss_last": loss_a[-1],
            "action_proj_l2": _action_proj_norm(act_model),
            "pred_l1_init_matched_vs_shuffled": init_action_l1,
            "pred_l1_trained_matched_vs_shuffled": trained_action_l1,
        },
        "jepa_delta_action_minus_history": jepa_a - jepa_h,
        "jepa_delta_shuffled_minus_matched": jepa_shuffled - jepa_a,
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--shard-dir", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from tests.conftest import _build_model_with_mock_backbone

    report = run_action_jepa_ab(
        _build_model_with_mock_backbone,
        steps=args.steps, lr=args.lr, seed=args.seed, shard_dir=args.shard_dir,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()

