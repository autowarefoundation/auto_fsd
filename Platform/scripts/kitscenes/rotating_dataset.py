import os
import json
import shutil
import random
import torch
from pathlib import Path
from torch.utils.data import IterableDataset
import time
import sys

# Dynamically find the repo root (3 folders up) for internal imports
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
model_dir = str(Path(repo_root) / "Model")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)    

from Model.data_parsing.kit_scenes.source import PinnedKITScenesDownloader, KITSCENES_DATA_REVISION
from Model.data_parsing.kit_scenes import KitScenesDataset
from Model.navigation.artifacts import raster_from_sample_members

# Sanitized for public repository
OUTPUT_DIR = Path(os.environ.get("KITSCENES_ROOT", "./kitscenes_root"))

# Global downloader instance leveraging HF token from environment if present
downloader = PinnedKITScenesDownloader(
    OUTPUT_DIR, 
    revision=KITSCENES_DATA_REVISION,
    token=None
)

class RotatingSceneKitScenes(IterableDataset):
    """One scene materialized on disk at a time; deleted before the next."""
    
    def __init__(self, scene_ids, manifest_split_by_id, backbone_name, shuffle=True, seed=0):
        self.scene_ids = list(scene_ids)
        self.manifest_split = manifest_split_by_id
        self.backbone_name = backbone_name
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        order = list(self.scene_ids)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)

        for i, scene_id in enumerate(order):
            real_split = self.manifest_split[scene_id]
            
            # --- PROGRESS TRACKER ---
            print(f"\n>>> [Epoch {self.epoch}] Processing Scene {i+1}/{len(order)}: {scene_id} <<<")
            
            # Fetch scene on demand
            max_retries = 10
            download_success = False
            
            for attempt in range(max_retries):
                try:
                    downloader.download([scene_id], expected_split=real_split)
                    download_success = True
                    break
                except Exception as e:
                    print(f"\n[Network Error] HuggingFace download failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in 10 seconds (Attempt {attempt+2}/{max_retries})...")
                        time.sleep(10)
            
            if not download_success:
                print(f"Skipping scene {scene_id} due to persistent network failures.")
                continue
            
            try:
                ds = KitScenesDataset(
                    data_root=str(OUTPUT_DIR),
                    backbone_name=self.backbone_name,
                    split=real_split,
                    scene_ids=[scene_id],
                    include_navigation=True,
                )
            except ValueError as e:
                if "No valid samples" not in str(e):
                    raise  # don't swallow a genuinely different error
                print(f"Skipping scene {scene_id}: too few poses ({e})")
                shutil.rmtree(OUTPUT_DIR / "data" / real_split / scene_id, ignore_errors=True)
                continue

            try:
                for sample in ds:
                    raster = raster_from_sample_members(sample["navigation_members"])
                    
                    sample["map_context"] = torch.from_numpy(raster.map_context.copy())
                    sample["route_mask"] = torch.from_numpy(raster.route_mask.copy())
                    sample["map_valid"] = torch.tensor(raster.map_valid)
                    sample["route_valid"] = torch.tensor(raster.route_valid)
                    
                    yield sample
                    
            finally:
                scene_path = OUTPUT_DIR / "data" / real_split / scene_id
                shutil.rmtree(scene_path, ignore_errors=True)
