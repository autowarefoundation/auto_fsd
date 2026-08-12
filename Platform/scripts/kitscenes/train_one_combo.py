import os
# Must be set before any CUDA call. Reduces the allocator-fragmentation
# stalls PyTorch itself suggested in the OOM message.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
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
    "swin_v2_tiny": "swinv2_tiny_window8_256",
    "conv_next_v2_tiny": "convnextv2_tiny",
    "res_net_50": "resnet50",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=BACKBONE_DATASET_TAG.keys())
    ap.add_argument("--map_fusion_mode", required=True, choices=["residual", "cross_attn"])
    ap.add_argument("--planner_mode", required=True, choices=["bezier", "flow_matching"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    # Reduced BEV grid to save memory on 8GB VRAM
    ap.add_argument("--bev_h", type=int, default=90)
    ap.add_argument("--bev_w", type=int, default=60)
    ap.add_argument("--bf16", action="store_true", default=True,
                    help="Use bfloat16 autocast (safe on Ampere+; fp16 is explicitly "
                         "off by default in train_il due to GradScaler overflow)")
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    if args.planner_mode == "flow_matching":
        raise SystemExit(
            "flow_matching deferred per your plan — use your wired-in "
            "compute_planner_loss version for this arm, not this script."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.bf16 and device == "cuda"

    split = json.load(open("kitscenes_split.json"))
    train_ds = RotatingSceneKitScenes(
        scene_ids=split["train_ids"],
        manifest_split_by_id=split["train_ids_manifest_split"],
        backbone_name=BACKBONE_DATASET_TAG[args.backbone],
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)

    model = AutoE2E(
        backbone=args.backbone,
        map_fusion_mode=args.map_fusion_mode,
        planner_mode=args.planner_mode,
        map_context_channels=14,
        view_fusion_kwargs={"bev_h": args.bev_h, "bev_w": args.bev_w},
    ).to(device)

    try:
        model.Reactive_E2E.Backbone.backbone.set_grad_checkpointing(enable=True)
        print("Enabled grad checkpointing on Camera Backbone")
    except AttributeError:
        pass
    
    try:
        model.Reactive_E2E.NavigationEncoder._backbone.set_grad_checkpointing(enable=True)
        print("Enabled grad checkpointing on Navigation Backbone")
    except AttributeError:
        pass

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data_cfg = resolve_model_data_config(model=BACKBONE_DATASET_TAG[args.backbone])
    img_mean = torch.tensor(data_cfg["mean"], device=device).view(1, 1, 3, 1, 1)
    img_std  = torch.tensor(data_cfg["std"],  device=device).view(1, 1, 3, 1, 1)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.backbone}_{args.map_fusion_mode}_{args.planner_mode}"

    print(f"Starting training on device: {device} "
          f"(bev_h={args.bev_h}, bev_w={args.bev_w}, bf16={use_amp})...")
          
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        for step, batch in enumerate(loader):

            camera_tiles = batch["visual_tiles"].to(device).float() / 255.0
            camera_tiles = (camera_tiles - img_mean) / img_std

            map_context = batch["map_context"].to(device).float()

            route_mask = batch.get("route_mask")
            if route_mask is not None:
                route_mask = route_mask.to(device).float()

            map_valid = batch.get("map_valid")
            if map_valid is not None:
                map_valid = map_valid.to(device)

            route_valid = batch.get("route_valid")
            if route_valid is not None:
                route_valid = route_valid.to(device)

            visual_history = batch["visual_history"].to(device).float()
        
            egomotion_history = batch["egomotion_history"].to(device).float()
            trajectory_target = batch["trajectory_target"].to(device).float()
            camera_params = batch["camera_params"].to(device).float()

            if step == 0 and epoch == 0:
                print("camera_tiles:", camera_tiles.shape)
                print("visual_history:", visual_history.shape)
                print("egomotion_history:", egomotion_history.shape)
                print("map_context:", map_context.shape)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(
                  camera_tiles,
                  map_context,
                  visual_history,
                  egomotion_history,
                  route_mask=route_mask,
                  map_valid=map_valid,
                  route_valid=route_valid,
                  projection=PinholeProjection(camera_params),
                  geometry_type="pinhole",
                  trajectory_target=trajectory_target,
                  mode="train",
                )

                trajectory = out if torch.is_tensor(out) else out[0]
                loss = F.smooth_l1_loss(trajectory, trajectory_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        ckpt_path = ckpt_dir / f"{run_tag}_epoch{epoch}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
