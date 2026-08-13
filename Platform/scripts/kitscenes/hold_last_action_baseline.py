"""Computes the no-perception baseline for YOUR val split.
Hold last observed (accel, curvature) from egomotion_history and integrate.
Run this BEFORE reporting any model ADE — needed to contextualise results.
No GPU, no model, no re-download needed if eval scenes are still on disk.
"""
import json
import shutil
import sys
import os
import time
from pathlib import Path
import numpy as np

# Dynamically find the repo root (3 folders up) for internal imports
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
model_dir = str(Path(repo_root) / "Model")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)    

from Model.data_parsing.kit_scenes import KitScenesDataset
from Model.data_parsing.kit_scenes.source import PinnedKITScenesDownloader, KITSCENES_DATA_REVISION
from Model.evaluation.metrics import integrate_trajectory

# Sanitized for public repository
OUTPUT_DIR = Path(os.environ.get("KITSCENES_ROOT", "./kitscenes_root"))
SPLIT_FILE = "kitscenes_split.json"
BACKBONE   = "swinv2_tiny_window8_256"   # any backbone — only egomotion matters here

split    = json.load(open(SPLIT_FILE))
val_ids  = split["val_ids"]
manifest = split["train_ids_manifest_split"]

downloader = PinnedKITScenesDownloader(OUTPUT_DIR, revision=KITSCENES_DATA_REVISION)

ade3, fde3 = [], []
skipped = 0
network_failures = 0

for i, scene_id in enumerate(val_ids):
    print(f"scene {i+1}/{len(val_ids)}: {scene_id[:8]}...", end=" ", flush=True)
    
    download_success = False
    for attempt in range(3):
        try:
            downloader.download([scene_id], expected_split="val")
            download_success = True
            break
        except Exception as e:
            print(f"  download attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5)
                
    if not download_success:
        print(f"Scene {scene_id}: skipped due to persistent network failure (not a short scene)")
        network_failures += 1
        continue

    try:
        ds = KitScenesDataset(
            data_root=str(OUTPUT_DIR),
            backbone_name=BACKBONE,
            split="val",
            scene_ids=[scene_id],
            include_navigation=True,
        )
    except ValueError:
        print("skipped (too short)")
        skipped += 1
        shutil.rmtree(OUTPUT_DIR / "data" / "val" / scene_id, ignore_errors=True)
        continue

    for sample in ds:
        # egomotion_history: (256,) = 64 steps × 4 signals
        # signals: 0=speed, 1=accel, 2=yaw_rate, 3=curvature
        eh = sample["egomotion_history"].numpy().reshape(64, 4)
        v0          = float(eh[-1, 0])
        last_accel  = float(eh[-1, 1])
        last_curv   = float(eh[-1, 3])

        # ground truth future
        tgt = sample["trajectory_target"].numpy().reshape(64, 2)
        gt_accel = tgt[:, 0]
        gt_curv  = tgt[:, 1]

        # hold-last-action: repeat last observed action
        hold_accel = np.full(64, last_accel)
        hold_curv  = np.full(64, last_curv)

        gt_xy   = integrate_trajectory(gt_accel,   gt_curv,   v0)
        hold_xy = integrate_trajectory(hold_accel, hold_curv, v0)
        err = np.linalg.norm(hold_xy - gt_xy, axis=1)

        ade3.append(err[:30].mean())   # 30 steps = 3s (post-#176 contract)
        fde3.append(err[29])

    print(f"{len(ade3)} samples so far")
    shutil.rmtree(OUTPUT_DIR / "data" / "val" / scene_id, ignore_errors=True)

print(f"\n{'='*50}")
print("Hold-last-action baseline — your 65-scene val split")
print("(repeats last observed accel+curvature; harder than zero-accel constant-velocity)")
print(f"Samples: {len(ade3)}  Skipped (too short): {skipped}  Network failures: {network_failures}")
print(f"ADE@3s: {np.mean(ade3):.3f} m")
print(f"FDE@3s: {np.mean(fde3):.3f} m")
print(f"(gate thresholds per #176: ADE@3s 2.0, FDE@3s 4.0)")