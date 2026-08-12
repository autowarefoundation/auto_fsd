"""Standalone eval — run against any saved checkpoint from train_one_combo.py.
Usage:
  python eval_checkpoint.py --ckpt checkpoints/swin_v2_tiny_residual_bezier_epoch0_step100.pt
"""

import argparse, json, sys, os
from pathlib import Path
import numpy as np
import torch

# Dynamically find the repo root (3 folders up) for internal imports
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
model_dir = str(Path(repo_root) / "Model")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)    

from Model.model_components.auto_e2e import AutoE2E
from Model.model_components.view_fusion.projection import PinholeProjection
from Model.evaluation.metrics import integrate_trajectory, compute_comfort_metrics
from rotating_dataset import RotatingSceneKitScenes
from torch.utils.data import DataLoader
from timm.data import resolve_model_data_config

BACKBONE_DATASET_TAG = {
    "swin_v2_tiny": "swinv2_tiny_window8_256",
    "conv_next_v2_tiny": "convnextv2_tiny",
    "res_net_50": "resnet50",
}

def compute_ade_fde_extended(pred_accel, pred_curv, gt_accel, gt_curv, initial_speed):
    """Extends metrics.py with 6.4s horizon required by m-zain-khawaja."""
    B = pred_accel.shape[0]
    results = {k: [] for k in ["ADE@3s","FDE@3s","ADE@6.4s","FDE@6.4s"]}
    
    for i in range(B):
        v0 = float(initial_speed[i])
        pred_xy = integrate_trajectory(pred_accel[i], pred_curv[i], v0)
        gt_xy   = integrate_trajectory(gt_accel[i],   gt_curv[i],   v0)
        errors = np.linalg.norm(pred_xy - gt_xy, axis=1)  # (64,)
        
        results["ADE@3s"].append(errors[:30].mean())   # 30 steps = 3.0s
        results["FDE@3s"].append(errors[29])
        results["ADE@6.4s"].append(errors.mean())      # all 64 steps = 6.4s
        results["FDE@6.4s"].append(errors[-1])
    
    return {k: float(np.mean(v)) for k, v in results.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_scenes", type=int, default=None,
                    help="Limit val scenes for a quick sanity check (None = all 65)")
    ap.add_argument("--bev_h", type=int, default=90)
    ap.add_argument("--bev_w", type=int, default=60)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    run_args = ckpt["args"]
    print(f"Evaluating: {args.ckpt}")
    print(f"  Config: {run_args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = run_args["backbone"]
    
    model = AutoE2E(
        backbone=backbone,
        map_fusion_mode=run_args["map_fusion_mode"],
        planner_mode=run_args["planner_mode"],
        map_context_channels=14,
        view_fusion_kwargs={"bev_h": run_args.get("bev_h", 90),
                            "bev_w": run_args.get("bev_w", 60)},
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    split = json.load(open("kitscenes_split.json"))
    val_ids = split["val_ids"]
    if args.max_scenes:
        val_ids = val_ids[:args.max_scenes]
    print(f"Val scenes: {len(val_ids)}")

    val_manifest = {scene_id: "val" for scene_id in val_ids}

    val_ds = RotatingSceneKitScenes(
        scene_ids=val_ids,
        manifest_split_by_id=val_manifest,
        backbone_name=BACKBONE_DATASET_TAG[backbone],
        shuffle=False,
    )
    loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    data_cfg = resolve_model_data_config(model=BACKBONE_DATASET_TAG[backbone])
    img_mean = torch.tensor(data_cfg["mean"], device=device).view(1, 1, 3, 1, 1)
    img_std  = torch.tensor(data_cfg["std"],  device=device).view(1, 1, 3, 1, 1)

    all_pred_accel, all_pred_curv = [], []
    all_gt_accel,   all_gt_curv   = [], []
    all_v0 = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            camera_tiles = batch["visual_tiles"].to(device).float() / 255.0
            camera_tiles = (camera_tiles - img_mean) / img_std
            map_context  = batch["map_context"].to(device).float()
            route_mask   = batch.get("route_mask")
            if route_mask is not None:
                route_mask = route_mask.to(device).float()
            visual_history    = batch["visual_history"].to(device).float()
            egomotion_history = batch["egomotion_history"].to(device).float()
            trajectory_target = batch["trajectory_target"].to(device).float()
            camera_params     = batch["camera_params"].to(device).float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device=="cuda")):
                out = model(
                    camera_tiles, map_context, visual_history, egomotion_history,
                    route_mask=route_mask,
                    projection=PinholeProjection(camera_params),
                    geometry_type="pinhole",
                    mode="infer",  # <-- infer mode: returns trajectory directly, no loss
                )

            traj_pred = (out if torch.is_tensor(out) else out[0]).float().cpu().numpy()
            traj_gt   = trajectory_target.float().cpu().numpy()

            # (B, 128) → (B, 64, 2) → split accel/curv
            pred_ac = traj_pred.reshape(-1, 64, 2)
            gt_ac   = traj_gt.reshape(-1, 64, 2)

            all_pred_accel.append(pred_ac[:, :, 0])
            all_pred_curv.append(pred_ac[:, :, 1])
            all_gt_accel.append(gt_ac[:, :, 0])
            all_gt_curv.append(gt_ac[:, :, 1])

            # initial speed = last history frame's speed
            # egomotion_history (B, 256) = 64 steps x 4 signals, speed is signal 0
            eh = egomotion_history.float().cpu().numpy().reshape(-1, 64, 4)
            all_v0.append(eh[:, -1, 0])  # speed at last history step

            if step % 100 == 0:
                print(f"  eval step {step}")

    pred_accel = np.concatenate(all_pred_accel)
    pred_curv  = np.concatenate(all_pred_curv)
    gt_accel   = np.concatenate(all_gt_accel)
    gt_curv    = np.concatenate(all_gt_curv)
    v0         = np.concatenate(all_v0)

    metrics = compute_ade_fde_extended(pred_accel, pred_curv, gt_accel, gt_curv, v0)
    comfort  = compute_comfort_metrics(pred_accel, pred_curv, v0)

    print("\n=== RESULTS ===")
    print(f"Samples evaluated: {len(v0)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f} m")
    print(f"  comfort_violation_rate: {comfort['comfort_violation_rate']:.3f}")
    print(f"\nConfig: backbone={backbone} map_fusion={run_args['map_fusion_mode']} "
          f"planner={run_args['planner_mode']}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"\nNOTE: val split is 65-scene custom (not the frozen train_il digest). "
          f"Numbers not directly comparable with contributors using train_il --validation_scope full.")

if __name__ == "__main__":
    main()
