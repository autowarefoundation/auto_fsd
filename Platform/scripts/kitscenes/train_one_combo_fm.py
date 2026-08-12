"""Training script using compute_planner_loss() — correct for flow_matching
and also better for bezier (temporal decay + signal scaling).

Requires: fix/115-compute-planner-loss branch (return_planner_loss flag).
DO NOT run against main — it will silently fall back to the inference path.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse, json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from timm.data import resolve_model_data_config

# Dynamically find the repo root (3 folders up) for internal imports
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
model_dir = str(Path(repo_root) / "Model")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)    

from Model.model_components.auto_e2e import AutoE2E
from Model.model_components.view_fusion.projection import PinholeProjection
from rotating_dataset import RotatingSceneKitScenes

BACKBONE_DATASET_TAG = {
    "swin_v2_tiny":      "swinv2_tiny_window8_256",
    "conv_next_v2_tiny": "convnextv2_tiny",
    "res_net_50":        "resnet50",
}

# Training policy for bezier — matches intisar1020's physical-unit-aligned scales.
# training_policy=None is correct for flow_matching (per PR #124 docstring).
BEZIER_TRAINING_POLICY = SimpleNamespace(
    temporal_decay=0.95,
    signal_scales=(0.778, 0.035),   # accel (m/s²), curvature (1/m)
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone",       required=True, choices=BACKBONE_DATASET_TAG)
    ap.add_argument("--map_fusion_mode",required=True, choices=["residual", "cross_attn"])
    ap.add_argument("--planner_mode",   required=True, choices=["bezier", "flow_matching"])
    ap.add_argument("--epochs",    type=int, default=1)
    ap.add_argument("--batch_size",type=int, default=1)
    ap.add_argument("--bev_h",     type=int, default=90)
    ap.add_argument("--bev_w",     type=int, default=60)
    ap.add_argument("--ckpt_dir",  type=str, default="checkpoints")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    # cross_attn guard — 90x60 = 5400 tokens > 4096 limit
    if args.map_fusion_mode == "cross_attn" and args.bev_h * args.bev_w > 4096:
        raise SystemExit(
            f"cross_attn requires bev_h*bev_w <= 4096 "
            f"(got {args.bev_h}x{args.bev_w}={args.bev_h*args.bev_w}). "
            f"Use --bev_h 60 --bev_w 60."
        )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = json.load(open("kitscenes_split.json"))
    train_ds = RotatingSceneKitScenes(
        scene_ids=split["train_ids"],
        manifest_split_by_id=split["train_ids_manifest_split"],
        backbone_name=BACKBONE_DATASET_TAG[args.backbone],
        seed=args.seed,
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)

    model = AutoE2E(
        backbone=args.backbone,
        map_fusion_mode=args.map_fusion_mode,
        planner_mode=args.planner_mode,
        map_context_channels=14,
        view_fusion_kwargs={"bev_h": args.bev_h, "bev_w": args.bev_w},
    ).to(device)

    # gradient checkpointing on both backbones — confirmed working from M1 runs
    try:
        model.Reactive_E2E.Backbone.backbone.set_grad_checkpointing(enable=True)
        print("grad checkpointing: camera backbone ON")
    except AttributeError:
        pass
    try:
        model.Reactive_E2E.NavigationEncoder._backbone.set_grad_checkpointing(enable=True)
        print("grad checkpointing: navigation backbone ON")
    except AttributeError:
        pass

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    data_cfg = resolve_model_data_config(model=BACKBONE_DATASET_TAG[args.backbone])
    img_mean = torch.tensor(data_cfg["mean"], device=device).view(1, 1, 3, 1, 1)
    img_std  = torch.tensor(data_cfg["std"],  device=device).view(1, 1, 3, 1, 1)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.backbone}_{args.map_fusion_mode}_{args.planner_mode}_seed{args.seed}"

    # training_policy: None for flow_matching (velocity-MSE, scaling doesn't apply)
    # SimpleNamespace for bezier (temporal decay + physical-unit signal scales)
    training_policy = (
        None if args.planner_mode == "flow_matching"
        else BEZIER_TRAINING_POLICY
    )

    print(f"Training: {run_tag} | device={device} | "
          f"bev={args.bev_h}x{args.bev_w} | bf16=True")
    print(f"training_policy: {training_policy}")

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(loader):
            camera_tiles = batch["visual_tiles"].to(device).float() / 255.0
            camera_tiles = (camera_tiles - img_mean) / img_std

            map_context       = batch["map_context"].to(device).float()
            route_mask        = batch.get("route_mask")
            if route_mask is not None:
                route_mask    = route_mask.to(device).float()
            map_valid         = batch.get("map_valid")
            if map_valid is not None:
                map_valid     = map_valid.to(device)
            route_valid       = batch.get("route_valid")
            if route_valid is not None:
                route_valid   = route_valid.to(device)

            visual_history    = batch["visual_history"].to(device).float()
            egomotion_history = batch["egomotion_history"].to(device).float()
            trajectory_target = batch["trajectory_target"].to(device).float()
            camera_params     = batch["camera_params"].to(device).float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                 enabled=(device == "cuda")):
                # KEY DIFFERENCE from train_one_combo.py:
                # return_planner_loss=True → calls compute_planner_loss() instead
                # of forward(). Returns dict, not tensor.
                out = model(
                    camera_tiles, map_context, visual_history, egomotion_history,
                    route_mask=route_mask,
                    map_valid=map_valid,
                    route_valid=route_valid,
                    projection=PinholeProjection(camera_params),
                    geometry_type="pinhole",
                    trajectory_target=trajectory_target,
                    training_policy=training_policy,
                    return_planner_loss=True,   # <-- the whole point of this script
                    mode="train",
                )

            # out is a dict: {"loss": scalar, "velocity_mse": scalar}
            # or {"loss": scalar, "imitation_loss": scalar} for bezier
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if step % 20 == 0:
                extra = {k: f"{v.item():.4f}" for k, v in out.items() if k != "loss"}
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} {extra}")

            # periodic checkpoint every 100 steps
            if step % 100 == 0 and step > 0:
                torch.save(
                    {"epoch": epoch, "step": step,
                     "model_state": model.state_dict(),
                     "opt_state": opt.state_dict(),
                     "args": vars(args)},
                    ckpt_dir / f"{run_tag}_epoch{epoch}_step{step}.pt",
                )

        # end-of-epoch checkpoint
        torch.save(
            {"epoch": epoch, "step": "end",
             "model_state": model.state_dict(),
             "opt_state": opt.state_dict(),
             "args": vars(args)},
            ckpt_dir / f"{run_tag}_epoch{epoch}.pt",
        )
        print(f"saved: {ckpt_dir}/{run_tag}_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
