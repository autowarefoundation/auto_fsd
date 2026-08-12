# KITScenes local training scripts

Exploratory training of the Reactive branch of AutoE2E on KITScenes without
the full cluster pipeline (`train_il`). Designed for single-GPU contributor
machines where the full 4 TB dataset cannot be stored locally. Implements the
grid search requested in #168.

## Requirements

```bash
# Python 3.12 venv from repo root
make setup TORCH_CHANNEL=cu121
make setup-map

# KITScenes SDK
cd Model/data_parsing/kit_scenes
git clone https://github.com/KIT-MRT/kitscenes.git
cd kitscenes && pip install -e . --no-deps && cd ../../..

# Lanelet2 (required by KITScenes navigation path on x86_64)
pip install lanelet2

# Native rasterizer
python Model/navigation/native/build.py

# HuggingFace gated access (account must have accepted KIT-MRT/KITScenes-Multimodal terms)
huggingface-cli login
```

## Environment

```bash
export KITSCENES_ROOT=/path/to/local/scratch  # needs ~20 GB free (one scene at a time)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Run all scripts from the **repo root**, not from this directory.

## Workflow

### 1. Build the 585/65 scene split

Matches the split specified by @m-zain-khawaja in #168
(all 533 `data/train/` archives + first 52 `data/val/` archives sorted by
`sequence_id`, leaving 65 val scenes).

```bash
python Platform/scripts/kitscenes/build_shard_split.py
# writes kitscenes_split.json to repo root
```

### 2. Train — bezier planner (main-compatible)

```bash
python Platform/scripts/kitscenes/train_one_combo.py \
  --backbone swin_v2_tiny \
  --map_fusion_mode residual \
  --planner_mode bezier \
  --epochs 1 --batch_size 1 \
  --bev_h 90 --bev_w 60
```

### 3. Train — flow_matching planner (requires PR #172)

Verify the branch has `return_planner_loss` before running:

```bash
python -c "
import inspect, sys; sys.path.insert(0,'.')
from Model.model_components.auto_e2e import AutoE2E
assert 'return_planner_loss' in inspect.signature(AutoE2E.forward).parameters
print('OK')
"
```

```bash
python Platform/scripts/kitscenes/train_one_combo_fm.py \
  --backbone swin_v2_tiny \
  --map_fusion_mode residual \
  --planner_mode flow_matching \
  --epochs 1 --batch_size 1 \
  --bev_h 90 --bev_w 60 \
  --seed 42
```

`cross_attn` fusion requires `--bev_h 60 --bev_w 60` (≤ 4096 tokens).

### 4. Compute the constant-velocity baseline

Run this **before** reporting any model ADE. Holds the last observed
(accel, curvature) from `egomotion_history` and integrates — no GPU,
no model, no re-download if val scenes are still on disk.

```bash
python Platform/scripts/kitscenes/constant_velocity_baseline.py
```

### 5. Evaluate a checkpoint

```bash
python Platform/scripts/kitscenes/eval_checkpoint.py \
  --ckpt checkpoints/swin_v2_tiny_residual_bezier_seed42_epoch0.pt
```

Reports ADE/FDE @ 3 s (post-#176 contract) and @ 6.4 s.

## Important caveats

- **Split**: 585/65 custom split, not the frozen `kitscenes_train_dev_v2.json`
  manifest enforced by `--validation_scope full`. Numbers from these scripts are
  not directly comparable with `train_il` results. Report alongside your
  constant-velocity baseline so the gap is interpretable.
- **Short scenes**: scenes with fewer than 129 ego poses yield no valid samples
  under the default 6.4 s window and are skipped. Log `skipped_scenes.log` to
  track them.
- **Memory**: tested on RTX 4060 8 GB with `--bev_h 90 --bev_w 60` and
  gradient checkpointing enabled on both backbones.
