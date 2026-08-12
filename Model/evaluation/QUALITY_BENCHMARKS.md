# World-Model Quality Benchmarks

Speed benchmarks live in [`../speed_benchmark/`](../speed_benchmark/). This folder
documents **quality** metrics for the policy / World Model stack:

| Metric family | What it measures |
|---------------|------------------|
| JEPA reconstruction | Per-horizon L1 / L2 / cosine of predicted vs frozen-target feature maps |
| Null-relative improvement | How much better than predicting zeros |
| Reactive vs Combined impact | Trajectory L1/L2 delta when the World Model is enabled |
| Open-loop ADE/FDE pair | Reactive and Combined ADE@3s / FDE@3s on the same GT (when labels exist) |

## Quick start

```bash
cd Model

# Synthetic JEPA recon smoke (CPU, no checkpoint)
python evaluation/world_model_quality_benchmark.py --synthetic

# Reactive vs Combined trajectory impact (builds AutoE2E with WM on)
python evaluation/world_model_quality_benchmark.py --impact --device cpu
```

## Library API

```python
from evaluation.world_model_quality import (
    jepa_reconstruction_metrics,
    world_model_trajectory_impact,
    open_loop_pair_metrics,
)
```
