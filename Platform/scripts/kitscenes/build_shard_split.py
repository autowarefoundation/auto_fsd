import sys
import os
import json
from pathlib import Path

# Dynamically find the repo root (3 folders up) for internal imports
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

model_dir = str(Path(repo_root) / "Model")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
    
from Model.data_parsing.kit_scenes.source import (
    parse_archive_manifest, fetch_archive_manifest, KITSCENES_DATA_REVISION
)

# Sanitized for public repository
OUTPUT_DIR = Path(os.environ.get("KITSCENES_ROOT", "./kitscenes_root"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

archives = fetch_archive_manifest(OUTPUT_DIR, revision=KITSCENES_DATA_REVISION)

# Sort strictly by scene_id for reproducibility
train = sorted([a for a in archives.values() if a.split == "train"], key=lambda a: a.scene_id)
val = sorted([a for a in archives.values() if a.split == "val"], key=lambda a: a.scene_id)

assert len(train) == 533 and len(val) == 117, f"Manifest length mismatch: train={len(train)}, val={len(val)}"

# Extract the 52 validation scenes to append to train
val_to_train = val[:52]
val_remaining = val[52:]

train_ids = [a.scene_id for a in train] + [a.scene_id for a in val_to_train]
val_ids = [a.scene_id for a in val_remaining]

assert len(train_ids) == 585 and len(val_ids) == 65

# Disk math
total_train_bytes = sum(a.size_bytes for a in train) + sum(a.size_bytes for a in val_to_train)
total_val_bytes = sum(a.size_bytes for a in val_remaining)

print(f"train: {len(train_ids)} scenes, {total_train_bytes/1e12:.2f} TB")
print(f"val:   {len(val_ids)} scenes, {total_val_bytes/1e12:.2f} TB")

# Export split logic
out_data = {
    "train_ids": train_ids,
    "val_ids": val_ids,
    "train_ids_manifest_split": {a.scene_id: a.split for a in train + val_to_train}
}

with open("kitscenes_split.json", "w") as f:
    json.dump(out_data, f, indent=2)

print("Wrote kitscenes_split.json")
