"""Ray Train DDP runner for nuPlan and L2D Reactive stages."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import socket
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from distributed_training.reactive_data import (
    RestartingIterator,
    assign_reactive_shards,
    build_reactive_dataset_plan,
    optimizer_steps_per_epoch,
    reactive_assignment_sha256,
    stage_rank_reactive_shards,
)
from training.reactive_multitask import ReactiveTrainingStage


SUPPORTED_WORLD_SIZES = frozenset({2, 4, 8})
SUPPORTED_PRECISIONS = frozenset({"fp32", "bf16"})
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def validate_reactive_stage_config(config: Mapping[str, Any]) -> None:
    """Validate the JSON-safe contract before starting a Ray cluster."""
    try:
        stage = ReactiveTrainingStage(str(config["stage"]))
    except (KeyError, ValueError) as error:
        raise ValueError("unsupported Reactive distributed stage") from error
    world_size = int(config.get("num_workers", 0))
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"num_workers must be one of {sorted(SUPPORTED_WORLD_SIZES)}"
        )
    if int(config.get("worker_cpus", 0)) <= 0:
        raise ValueError("worker_cpus must be positive")
    if int(config.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    if int(config.get("per_rank_batch_size", 0)) != 1:
        raise ValueError(
            "Reactive DDP v1 requires per_rank_batch_size=1"
        )
    if int(config.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config.get("num_loader_workers", -1)) < 0:
        raise ValueError("num_loader_workers must be non-negative")
    if not 0.0 < float(config.get("val_fraction", 0.0)) < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if float(config.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(config.get("weight_decay", -1.0)) < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if float(config.get("grad_clip", 0.0)) <= 0.0:
        raise ValueError("grad_clip must be positive")
    precision = str(config.get("precision", ""))
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}"
        )
    if precision == "bf16" and not bool(config.get("use_gpu", True)):
        raise ValueError("bf16 Reactive DDP requires GPU workers")
    sources = config.get("source_uris")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_uris must be a non-empty list")
    if len(set(str(item) for item in sources)) != len(sources):
        raise ValueError("source_uris contains duplicates")
    run_name = str(config.get("run_name", ""))
    if _RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError("run_name contains unsupported characters")
    storage_path = str(config.get("storage_path", ""))
    if not storage_path.startswith("s3://"):
        raise ValueError("storage_path must be an S3 URI")
    parent_uri = str(config.get("parent_checkpoint_uri") or "")
    if stage is ReactiveTrainingStage.NUPLAN_FULL and parent_uri:
        raise ValueError("Stage A cannot load a parent checkpoint")
    if stage is ReactiveTrainingStage.L2D_CONTINUATION and not parent_uri:
        raise ValueError("Stage B requires the exact Stage A checkpoint")
    override = int(config.get("steps_per_epoch", 0))
    if override < 0:
        raise ValueError("steps_per_epoch cannot be negative")
    overfit_sample_count = int(config.get("overfit_sample_count", 0))
    if overfit_sample_count not in (0, *range(64, 129)):
        raise ValueError(
            "overfit_sample_count must be zero or between 64 and 128"
        )
    if (
        overfit_sample_count
        and stage is not ReactiveTrainingStage.NUPLAN_FULL
    ):
        raise ValueError("BEV overfit mode is valid only for Stage A")
    if overfit_sample_count and overfit_sample_count % world_size:
        raise ValueError(
            "overfit_sample_count must be divisible by num_workers"
        )
    overfit_shard_limit = int(config.get("overfit_shard_limit", 0))
    if overfit_shard_limit and overfit_shard_limit < world_size:
        raise ValueError(
            "overfit_shard_limit must be zero or at least num_workers"
        )
    for name in ("overfit_min_ap", "overfit_min_recall"):
        threshold = float(config.get(name, -1.0))
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"{name} must be in (0,1]")
    if "bev_pos_weights" in config:
        raise ValueError(
            "bev_pos_weights is derived from train statistics and cannot "
            "be configured"
        )
    if float(config.get("bev_pos_weight_cap", 0.0)) < 1.0:
        raise ValueError("bev_pos_weight_cap must be at least one")
    repeat_threshold = float(
        config.get("bev_repeat_frequency_threshold", 0.0)
    )
    if not 0.0 < repeat_threshold <= 1.0:
        raise ValueError(
            "bev_repeat_frequency_threshold must be in (0,1]"
        )
    if int(config.get("bev_max_repeat", 0)) < 1:
        raise ValueError("bev_max_repeat must be positive")
    if int(config.get("bev_min_positive_samples", 0)) < 1:
        raise ValueError("bev_min_positive_samples must be positive")
    if int(config.get("bev_min_positive_cells", 0)) < 1:
        raise ValueError("bev_min_positive_cells must be positive")
    if int(config.get("bev_ap_bins", 0)) < 256:
        raise ValueError("bev_ap_bins must be at least 256")
    if float(config.get("selection_ade_scale_m", 0.0)) <= 0.0:
        raise ValueError("selection_ade_scale_m must be positive")
    if int(config.get("validation_sample_limit", 1024)) < world_size:
        raise ValueError(
            "validation_sample_limit must be at least num_workers"
        )
    if float(
        config.get("selection_ade_regression_margin_m", -1.0)
    ) < 0.0:
        raise ValueError(
            "selection_ade_regression_margin_m must be non-negative"
        )
    gate_dataset_digest = str(
        config.get("required_gate_dataset_manifest_sha256", "")
    )
    if gate_dataset_digest and re.fullmatch(
        r"[0-9a-f]{64}",
        gate_dataset_digest,
    ) is None:
        raise ValueError(
            "required_gate_dataset_manifest_sha256 must be empty or SHA-256"
        )


def _base_model(model):
    from torch.nn.parallel import DistributedDataParallel

    return (
        model.module
        if isinstance(model, DistributedDataParallel)
        else model
    )


def _batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _loader_item(item: Any) -> tuple[Mapping[str, Any], Any, str]:
    if isinstance(item, tuple):
        if len(item) != 3:
            raise ValueError(
                "Reactive loader items must be "
                "(batch, projection, geometry_type)"
            )
        batch, projection, geometry_type = item
        return batch, projection, str(geometry_type)
    return item, None, "pseudo"


def _collective_true(value, device) -> bool:
    import torch
    import torch.distributed as dist

    flag = torch.as_tensor(value, device=device)
    if flag.numel() != 1:
        raise ValueError("collective boolean flag must be scalar")
    flag = flag.to(dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _all_reduce_bev_statistics(local_statistics, device):
    """Combine exact rank-local BEV counts into one global contract."""
    import torch
    import torch.distributed as dist

    from data_parsing.pre_extracted import BEVTrainingStatistics

    packed = torch.tensor(
        [
            float(local_statistics.sample_count),
            float(local_statistics.effective_exposure_count),
            *local_statistics.positive_sample_count,
            *local_statistics.positive_cell_count,
            *local_statistics.positive_mass,
            *local_statistics.valid_cell_count,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    class_count = 8
    offset = 2

    def take(count: int):
        nonlocal offset
        values = packed[offset:offset + count]
        offset += count
        return values

    positive_samples = take(class_count)
    positive_cells = take(class_count)
    positive_mass = take(class_count)
    valid_cells = take(class_count)
    rank_digests: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        rank_digests,
        local_statistics.exposure_digest,
    )
    exposure_digest = hashlib.sha256(
        "\n".join(
            f"{rank}:{digest}"
            for rank, digest in enumerate(rank_digests)
        ).encode("ascii")
    ).hexdigest()
    return BEVTrainingStatistics(
        sample_count=int(round(float(packed[0].item()))),
        effective_exposure_count=int(round(float(packed[1].item()))),
        positive_sample_count=tuple(
            int(round(float(value)))
            for value in positive_samples.tolist()
        ),
        positive_cell_count=tuple(
            int(round(float(value)))
            for value in positive_cells.tolist()
        ),
        positive_mass=tuple(
            float(value) for value in positive_mass.tolist()
        ),
        valid_cell_count=tuple(
            int(round(float(value)))
            for value in valid_cells.tolist()
        ),
        exposure_digest=exposure_digest,
    )


def _select_bev_overfit_subset(
    rank_summaries,
    *,
    sample_count: int,
) -> tuple[str, ...]:
    """Choose a deterministic class-complete subset balanced across ranks."""
    if not 64 <= sample_count <= 128:
        raise ValueError("BEV overfit subset must contain 64 to 128 samples")
    rank_count = len(rank_summaries)
    if rank_count <= 0 or sample_count % rank_count:
        raise ValueError(
            "BEV overfit subset must divide evenly across ranks"
        )
    per_rank_count = sample_count // rank_count
    candidates: list[tuple[str, int, frozenset[int]]] = []
    for rank, summaries in enumerate(rank_summaries):
        if len(summaries) < per_rank_count:
            raise ValueError(
                f"BEV overfit rank {rank} has fewer than "
                f"{per_rank_count} train samples"
            )
        for sample_uid, positive_classes in summaries:
            classes = frozenset(int(value) for value in positive_classes)
            if any(value < 0 or value >= 8 for value in classes):
                raise ValueError("BEV overfit summary has invalid class index")
            candidates.append((str(sample_uid), rank, classes))
    if len(candidates) < sample_count:
        raise ValueError(
            "BEV overfit request exceeds available train samples"
        )
    identities = [candidate[0] for candidate in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("BEV overfit candidates contain duplicate samples")

    selected: dict[str, tuple[str, int, frozenset[int]]] = {}
    selected_per_rank = [0] * rank_count
    covered: set[int] = set()
    while covered != set(range(8)):
        remaining_classes = set(range(8)) - covered
        eligible = [
            candidate
            for candidate in candidates
            if candidate[0] not in selected
            and candidate[2].intersection(remaining_classes)
            and selected_per_rank[candidate[1]] < per_rank_count
        ]
        if not eligible:
            raise ValueError(
                "BEV overfit candidates do not cover every class"
            )
        chosen = min(
            eligible,
            key=lambda candidate: (
                -len(candidate[2].intersection(remaining_classes)),
                selected_per_rank[candidate[1]],
                candidate[0],
            ),
        )
        selected[chosen[0]] = chosen
        selected_per_rank[chosen[1]] += 1
        covered.update(chosen[2])

    for rank in range(rank_count):
        for candidate in sorted(
            (
                candidate
                for candidate in candidates
                if candidate[1] == rank
            ),
            key=lambda value: value[0],
        ):
            if selected_per_rank[rank] >= per_rank_count:
                break
            if candidate[0] in selected:
                continue
            selected[candidate[0]] = candidate
            selected_per_rank[rank] += 1
    if len(selected) != sample_count:
        raise ValueError("BEV overfit subset construction is incomplete")
    if selected_per_rank != [per_rank_count] * rank_count:
        raise ValueError("BEV overfit subset is not rank balanced")
    return tuple(sorted(selected))


def _parameter_sample(model, *, sample_size: int = 2048):
    import torch

    samples = []
    remaining = sample_size
    for parameter in _base_model(model).parameters():
        if not parameter.requires_grad:
            continue
        flattened = parameter.detach().reshape(-1)
        take = min(remaining, flattened.numel())
        if take:
            samples.append(flattened[:take].float())
            remaining -= take
        if remaining == 0:
            break
    if not samples:
        raise RuntimeError("Reactive model has no trainable parameters")
    return torch.cat(samples)


def _maximum_parameter_delta(model, *, world_size: int) -> float:
    import torch
    import torch.distributed as dist

    local = _parameter_sample(model)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return max(
        float((candidate - gathered[0]).abs().max().item())
        for candidate in gathered
    )


def _download_checkpoint(checkpoint_uri: str, destination: Path) -> None:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme == "":
        source = Path(checkpoint_uri)
        destination.write_bytes(source.read_bytes())
        return
    if parsed.scheme == "file":
        source = Path(unquote(parsed.path))
        destination.write_bytes(source.read_bytes())
        return
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("checkpoint must be a local path or S3 URI")
    import boto3

    boto3.client("s3").download_file(
        parsed.netloc,
        parsed.path.lstrip("/"),
        str(destination),
    )


def normalize_ray_checkpoint_uri(
    checkpoint_path: str,
    storage_path: str,
) -> str:
    """Restore the S3 scheme Ray omits from filesystem-backed paths."""
    raw_path = checkpoint_path.rstrip("/")
    storage_uri = storage_path.rstrip("/")
    storage = urlparse(storage_uri)
    if storage.scheme != "s3" or not storage.netloc:
        raise ValueError("Reactive Ray checkpoint storage must be an S3 URI")
    storage_without_scheme = (
        f"{storage.netloc}/{storage.path.lstrip('/')}".rstrip("/")
    )
    parsed = urlparse(raw_path)
    if parsed.scheme == "s3":
        checkpoint_uri = raw_path
    elif parsed.scheme == "" and (
        raw_path == storage_without_scheme
        or raw_path.startswith(f"{storage_without_scheme}/")
    ):
        checkpoint_uri = f"s3://{raw_path}"
    else:
        raise ValueError(
            "Ray checkpoint path is outside the configured S3 storage"
        )
    if not (
        checkpoint_uri == storage_uri
        or checkpoint_uri.startswith(f"{storage_uri}/")
    ):
        raise ValueError(
            "Ray checkpoint URI is outside the configured S3 storage"
        )
    return checkpoint_uri


def _seed_epoch(seed: int, rank: int, epoch: int) -> None:
    import torch

    epoch_seed = seed + rank * 10_007 + epoch * 1_000_003
    random.seed(epoch_seed)
    np.random.seed(epoch_seed % (2**32))
    torch.manual_seed(epoch_seed)
    torch.cuda.manual_seed_all(epoch_seed)


def clip_finite_gradients_float64(
    parameters,
    max_norm: float,
):
    """Clip finite gradients without overflowing the global norm."""
    import torch

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return (
            torch.zeros((), dtype=torch.float64),
            torch.ones((), dtype=torch.bool),
        )
    device = gradients[0].device
    for gradient in gradients:
        if gradient.device != device:
            raise ValueError("all gradients must be on the same device")
    per_tensor_norms = [
        torch.linalg.vector_norm(
            gradient.detach(),
            ord=2,
            dtype=torch.float64,
        )
        for gradient in gradients
    ]
    gradient_norm = torch.linalg.vector_norm(torch.stack([
        value.detach()
        for value in per_tensor_norms
    ]))
    finite = torch.isfinite(gradient_norm)
    scale = torch.clamp(
        torch.as_tensor(
            max_norm,
            dtype=torch.float64,
            device=device,
        )
        / gradient_norm.clamp_min(torch.finfo(torch.float64).tiny),
        max=1.0,
    )
    scale = torch.where(finite, scale, torch.ones_like(scale))
    for gradient in gradients:
        gradient.mul_(scale.to(dtype=gradient.dtype))
    return gradient_norm, finite


def _train_fixed_steps(
    model,
    loader,
    objective,
    optimizer,
    *,
    device,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
    grad_clip: float,
    precision: str,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
    )

    model.train()
    iterator = RestartingIterator(loader)
    term_names = (
        "total",
        "trajectory",
        "bev_segmentation",
        "bev_segmentation_bce",
        "bev_segmentation_dice",
        "route_reconstruction",
    )
    totals = torch.zeros(
        len(term_names),
        dtype=torch.float64,
        device=device,
    )
    consumed_samples = 0
    micro_steps = optimizer_steps * gradient_accumulation_steps
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        finite_step = torch.ones((), dtype=torch.bool, device=device)
        for accumulation_index in range(gradient_accumulation_steps):
            raw_batch, fallback_projection, fallback_geometry_type = (
                _loader_item(next(iterator))
            )
            batch = _batch_to_device(raw_batch, device)
            consumed_samples += int(batch["visual_tiles"].shape[0])
            projection, geometry_type = (
                resolve_reactive_batch_projection(
                    batch,
                    fallback_projection,
                    fallback_geometry_type,
                    device=device,
                )
            )
            synchronize = (
                accumulation_index
                == gradient_accumulation_steps - 1
            )
            sync_context = (
                contextlib.nullcontext()
                if synchronize
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    output = model(
                        batch["visual_tiles"],
                        batch["map_context"],
                        batch["visual_history"],
                        batch["egomotion_history"],
                        route_mask=batch["route_mask"],
                        map_valid=batch["map_valid"],
                        route_valid=batch["route_valid"],
                        projection=projection,
                        geometry_type=geometry_type,
                        mode="train",
                        compute_bev_segmentation=(
                            objective.compute_bev_segmentation
                        ),
                        compute_route_reconstruction=True,
                    )
                    if not isinstance(output, tuple):
                        raise RuntimeError(
                            "Reactive DDP model omitted auxiliary outputs"
                        )
                    predicted_controls, auxiliary = output
                    terms = objective(
                        predicted_controls,
                        auxiliary,
                        batch,
                    )
                    finite_step.logical_and_(
                        torch.isfinite(terms["total"].detach())
                    )
                    scaled_loss = (
                        terms["total"]
                        / gradient_accumulation_steps
                    )
                scaled_loss.backward()
            totals += torch.stack([
                terms[name].detach().to(torch.float64)
                for name in term_names
            ])

        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        gradient_norm, finite_gradient = clip_finite_gradients_float64(
            trainable,
            grad_clip,
        )
        finite_step.logical_and_(finite_gradient)
        if not _collective_true(finite_step, device):
            raise FloatingPointError(
                "a Reactive DDP rank produced non-finite loss or gradients"
            )
        optimizer.step()

    packed = torch.cat([
        totals,
        torch.tensor(
            [float(consumed_samples), float(iterator.restarts)],
            dtype=torch.float64,
            device=device,
        ),
    ])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = dist.get_world_size() * micro_steps
    return {
        "total": float(packed[0].item() / denominator),
        "trajectory": float(packed[1].item() / denominator),
        "bev_segmentation": float(packed[2].item() / denominator),
        "bev_segmentation_bce": float(
            packed[3].item() / denominator
        ),
        "bev_segmentation_dice": float(
            packed[4].item() / denominator
        ),
        "route_reconstruction": float(
            packed[5].item() / denominator
        ),
        "consumed_samples": float(packed[6].item()),
        "loader_restarts": float(packed[7].item()),
        "local_consumed_samples": float(consumed_samples),
        "local_loader_restarts": float(iterator.restarts),
    }


def _histogram_average_precision(
    positive_histogram,
    negative_histogram,
) -> float:
    import torch

    positive_total = positive_histogram.sum()
    if float(positive_total.item()) <= 0.0:
        raise ValueError("BEV validation class has no positive cells")
    cumulative_positive = torch.cumsum(
        positive_histogram.flip(0),
        dim=0,
    )
    cumulative_negative = torch.cumsum(
        negative_histogram.flip(0),
        dim=0,
    )
    precision = cumulative_positive / (
        cumulative_positive + cumulative_negative
    ).clamp_min(1.0)
    recall = cumulative_positive / positive_total
    recall_delta = torch.diff(
        torch.cat([recall.new_zeros(1), recall])
    )
    return float((recall_delta * precision).sum().item())


def _metric_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _evaluate_global_reactive(
    model,
    loader,
    objective,
    *,
    stage: ReactiveTrainingStage,
    device,
    probability_bins: int,
    ade_scale_m: float,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )
    from training.losses.control_rollout import integrate_controls_torch
    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
    )

    base = _base_model(model)
    was_training = base.training
    base.eval()
    ade_sum = 0.0
    fde_sum = 0.0
    sample_count = 0
    class_count = len(BEV_SEGMENTATION_CLASSES)
    bev_counts = torch.zeros(
        (class_count, 5),
        dtype=torch.float64,
        device=device,
    )
    positive_histogram = torch.zeros(
        (class_count, probability_bins),
        dtype=torch.float64,
        device=device,
    )
    negative_histogram = torch.zeros_like(positive_histogram)
    bev_loss_sum = 0.0
    bev_bce_sum = 0.0
    bev_dice_sum = 0.0
    bev_loss_batches = 0
    try:
        with torch.no_grad():
            for item in loader:
                raw_batch, fallback_projection, fallback_geometry_type = (
                    _loader_item(item)
                )
                batch = _batch_to_device(raw_batch, device)
                projection, geometry_type = (
                    resolve_reactive_batch_projection(
                        batch,
                        fallback_projection,
                        fallback_geometry_type,
                        device=device,
                    )
                )
                output = base(
                    batch["visual_tiles"],
                    batch["map_context"],
                    batch["visual_history"],
                    batch["egomotion_history"],
                    route_mask=batch["route_mask"],
                    map_valid=batch["map_valid"],
                    route_valid=batch["route_valid"],
                    projection=projection,
                    geometry_type=geometry_type,
                    mode="infer",
                    return_auxiliary=(
                        stage is ReactiveTrainingStage.NUPLAN_FULL
                    ),
                    compute_bev_segmentation=(
                        stage is ReactiveTrainingStage.NUPLAN_FULL
                    ),
                    compute_route_reconstruction=False,
                )
                if isinstance(output, tuple):
                    controls, auxiliary = output
                else:
                    controls = output
                    auxiliary = {}
                batch_size = int(controls.shape[0])
                controls_3d = controls.reshape(batch_size, -1, 2)
                finite_controls = torch.isfinite(controls_3d).all(
                    dim=(1, 2)
                )
                safe_controls = torch.where(
                    finite_controls[:, None, None],
                    controls_3d,
                    torch.zeros_like(controls_3d),
                )
                predicted_xy, _, _ = integrate_controls_torch(
                    safe_controls,
                    batch["initial_speed_mps"],
                )
                target_xy = batch["trajectory_xy_m"].to(torch.float32)
                valid = (
                    batch["trajectory_valid"].to(dtype=torch.bool)
                    & finite_controls[:, None]
                    & torch.isfinite(target_xy).all(dim=-1)
                )
                if predicted_xy.shape != target_xy.shape:
                    raise ValueError(
                        "trajectory target shape differs from rollout"
                    )
                complete = valid.all(dim=1)
                if not bool(complete.any()):
                    continue
                errors = torch.linalg.vector_norm(
                    predicted_xy - target_xy,
                    dim=-1,
                )
                ade_sum += float(
                    errors[complete].mean(dim=1).sum().item()
                )
                fde_sum += float(errors[complete, -1].sum().item())
                sample_count += int(complete.sum().item())

                if stage is ReactiveTrainingStage.NUPLAN_FULL:
                    bev_logits = auxiliary.get(
                        "bev_segmentation_logits"
                    )
                    if not torch.is_tensor(bev_logits):
                        raise RuntimeError(
                            "Stage A validation omitted BEV logits"
                        )
                    target = batch["bev_segmentation_target"].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    valid_mask = batch[
                        "bev_segmentation_valid"
                    ].to(device=device, dtype=torch.bool)
                    if (
                        target.shape != bev_logits.shape
                        or valid_mask.shape != bev_logits.shape
                    ):
                        raise ValueError(
                            "BEV validation target shape differs"
                        )
                    bev_components = objective.bev_loss.components(
                        bev_logits,
                        target,
                        valid_mask,
                    )
                    bev_loss_sum += float(
                        bev_components["total"].item()
                    )
                    bev_bce_sum += float(
                        bev_components["bce"].item()
                    )
                    bev_dice_sum += float(
                        bev_components["dice"].item()
                    )
                    bev_loss_batches += 1
                    probability = bev_logits.float().sigmoid()
                    binary_target = target >= 0.5
                    binary_prediction = probability >= 0.5
                    for class_index in range(class_count):
                        class_valid = valid_mask[:, class_index]
                        if not bool(class_valid.any()):
                            continue
                        class_target = binary_target[
                            :, class_index
                        ][class_valid]
                        class_prediction = binary_prediction[
                            :, class_index
                        ][class_valid]
                        class_probability = probability[
                            :, class_index
                        ][class_valid]
                        bev_counts[class_index, 0] += (
                            class_prediction & class_target
                        ).sum()
                        bev_counts[class_index, 1] += (
                            class_prediction & ~class_target
                        ).sum()
                        bev_counts[class_index, 2] += (
                            ~class_prediction & class_target
                        ).sum()
                        bev_counts[class_index, 3] += class_target.sum()
                        bev_counts[class_index, 4] += class_valid.sum()
                        bins = torch.clamp(
                            (
                                class_probability * probability_bins
                            ).to(torch.int64),
                            max=probability_bins - 1,
                        )
                        positive_histogram[class_index] += (
                            torch.bincount(
                                bins[class_target],
                                minlength=probability_bins,
                            )
                        )
                        negative_histogram[class_index] += (
                            torch.bincount(
                                bins[~class_target],
                                minlength=probability_bins,
                            )
                        )
    finally:
        base.train(was_training)

    values = torch.tensor(
        [ade_sum, fde_sum, float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    dist.all_reduce(bev_counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(positive_histogram, op=dist.ReduceOp.SUM)
    dist.all_reduce(negative_histogram, op=dist.ReduceOp.SUM)
    bev_loss_values = torch.tensor(
        [
            bev_loss_sum,
            bev_bce_sum,
            bev_dice_sum,
            float(bev_loss_batches),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(bev_loss_values, op=dist.ReduceOp.SUM)
    global_count = int(values[2].item())
    if global_count <= 0:
        raise ValueError(
            "Reactive distributed validation has no complete trajectories"
        )
    metrics = {
        "ade_6p4s_m": float(values[0].item() / global_count),
        "fde_6p4s_m": float(values[1].item() / global_count),
        "complete_samples": float(global_count),
    }
    trajectory_quality = math.exp(
        -metrics["ade_6p4s_m"] / ade_scale_m
    )
    metrics["trajectory_quality"] = trajectory_quality
    if stage is not ReactiveTrainingStage.NUPLAN_FULL:
        metrics["selection_score"] = trajectory_quality
        return metrics

    if float(bev_loss_values[3].item()) <= 0.0:
        raise ValueError("Stage A validation has no BEV batches")
    metrics["bev_loss"] = float(
        bev_loss_values[0].item() / bev_loss_values[3].item()
    )
    metrics["bev_bce"] = float(
        bev_loss_values[1].item() / bev_loss_values[3].item()
    )
    metrics["bev_dice"] = float(
        bev_loss_values[2].item() / bev_loss_values[3].item()
    )
    average_precisions = [0.0] * class_count
    ap_lifts = [0.0] * class_count
    class_support = [False] * class_count
    for class_index, class_name in enumerate(BEV_SEGMENTATION_CLASSES):
        true_positive, false_positive, false_negative, positive, valid = (
            float(value)
            for value in bev_counts[class_index].tolist()
        )
        supported = positive > 0.0 and valid > 0.0
        prevalence = _metric_ratio(positive, valid)
        average_precision = 0.0
        ap_lift = 0.0
        if supported:
            average_precision = _histogram_average_precision(
                positive_histogram[class_index],
                negative_histogram[class_index],
            )
            if prevalence < 1.0:
                ap_lift = max(
                    0.0,
                    min(
                        1.0,
                        (average_precision - prevalence)
                        / (1.0 - prevalence),
                    ),
                )
        class_support[class_index] = supported
        ap_lifts[class_index] = ap_lift
        average_precisions[class_index] = average_precision
        prefix = f"bev_{class_name}"
        metrics[f"{prefix}_iou"] = _metric_ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        )
        metrics[f"{prefix}_precision"] = _metric_ratio(
            true_positive,
            true_positive + false_positive,
        )
        metrics[f"{prefix}_recall"] = _metric_ratio(
            true_positive,
            true_positive + false_negative,
        )
        metrics[f"{prefix}_average_precision"] = average_precision
        metrics[f"{prefix}_ap_lift"] = ap_lift
        metrics[f"{prefix}_positive_prevalence"] = prevalence
        metrics[f"{prefix}_positive_cells"] = positive
        metrics[f"{prefix}_supported"] = float(supported)
        metrics[f"{prefix}_valid_cells"] = valid

    def supported_mean(values, indices) -> float:
        selected = [
            values[index]
            for index in indices
            if class_support[index]
        ]
        return float(np.mean(selected)) if selected else 0.0

    static_indices = range(5)
    dynamic_indices = range(5, class_count)
    static_macro_ap_lift = supported_mean(ap_lifts, static_indices)
    dynamic_macro_ap_lift = supported_mean(ap_lifts, dynamic_indices)
    metrics["bev_static_macro_average_precision"] = supported_mean(
        average_precisions,
        static_indices,
    )
    metrics["bev_dynamic_macro_average_precision"] = supported_mean(
        average_precisions,
        dynamic_indices,
    )
    metrics["bev_static_macro_ap_lift"] = static_macro_ap_lift
    metrics["bev_dynamic_macro_ap_lift"] = dynamic_macro_ap_lift
    metrics["bev_supported_class_count"] = float(sum(class_support))
    metrics["bev_all_classes_supported"] = float(all(class_support))
    metrics["selection_score"] = (
        0.25 * trajectory_quality
        + 0.25 * static_macro_ap_lift
        + 0.50 * dynamic_macro_ap_lift
    )
    return metrics


def _load_resume_checkpoint(
    checkpoint_directory: str,
    *,
    model,
    optimizer,
    scheduler,
    expected: Mapping[str, Any],
) -> tuple[int, float, float, list[dict[str, Any]]]:
    import torch

    payload = torch.load(
        Path(checkpoint_directory) / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Reactive DDP resume checkpoint has no config")
    mismatches = {
        name: (config.get(name), value)
        for name, value in expected.items()
        if config.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"Reactive DDP resume contract differs: {mismatches}"
        )
    _base_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    training_state = payload.get("training_state") or {}
    history_path = Path(checkpoint_directory) / "history.json"
    history = json.loads(history_path.read_text(encoding="ascii"))
    if (
        not isinstance(history, list)
        or not history
        or any(not isinstance(item, dict) for item in history)
    ):
        raise ValueError("Reactive DDP resume checkpoint has invalid history")
    return (
        int(payload["epoch"]) + 1,
        float(training_state.get("best_selection_score", -float("inf"))),
        float(training_state.get("best_ade_6p4s_m", float("inf"))),
        history,
    )


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Train one fixed Reactive stage on every Ray worker."""
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    from data_parsing.pre_extracted import (
        BEVClassRepeatPolicy,
        derive_bev_pos_weights,
        derive_bev_repeat_factors,
        discover_bev_sample_statistics,
        discover_validation_sample_uids,
        make_multi_dataset_loader,
        passthrough_nodesplitter,
        select_bev_validation_sample_uids,
        summarize_bev_positive_samples,
        summarize_bev_training_statistics,
    )
    from model_components.auto_e2e import AutoE2E
    from training.reactive_multitask import (
        ReactiveMultitaskObjective,
        configure_model_for_stage,
        reactive_model_kwargs,
    )
    from training.reactive_stage_runner import (
        load_stage_a_parent,
        save_reactive_checkpoint,
    )

    validate_reactive_stage_config(config)
    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    if world_size != int(config["num_workers"]):
        raise RuntimeError(
            f"Ray world size {world_size} differs from requested workers"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Reactive distributed training requires CUDA")
    device = get_device()
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    stage = ReactiveTrainingStage(config["stage"])
    plan = build_reactive_dataset_plan(
        list(config["source_uris"]),
        stage=stage,
    )
    required_gate_dataset_digest = str(
        config.get("required_gate_dataset_manifest_sha256", "")
    )
    if (
        required_gate_dataset_digest
        and plan.dataset_manifest_sha256
        != required_gate_dataset_digest
    ):
        raise ValueError(
            "BEV overfit gate dataset differs from the full training dataset"
        )
    overfit_sample_count = int(config["overfit_sample_count"])
    assignment_shards = plan.shards
    overfit_shard_limit = int(config.get("overfit_shard_limit", 0))
    if overfit_sample_count and overfit_shard_limit:
        assignment_shards = plan.shards[:overfit_shard_limit]
    assignments = assign_reactive_shards(
        assignment_shards,
        world_size=world_size,
    )
    assignment_sha256 = reactive_assignment_sha256(assignments)
    rank_shards = assignments[rank]
    rank_sample_count = sum(
        shard.sample_count for shard in rank_shards
    )
    cache_root = (
        Path(config.get("local_cache_root") or "/tmp/auto-e2e-reactive")
        / config["run_name"]
        / f"rank-{rank:03d}"
    )
    local_directories = stage_rank_reactive_shards(
        rank_shards,
        cache_root=cache_root,
    )
    local_bev_records = (
        discover_bev_sample_statistics(local_directories)
        if stage is ReactiveTrainingStage.NUPLAN_FULL
        else None
    )
    overfit_sample_uids: tuple[str, ...] | None = None
    overfit_sample_uid_sha256 = ""
    if overfit_sample_count:
        assert local_bev_records is not None
        local_summaries = summarize_bev_positive_samples(
            local_bev_records,
            val_fraction=float(config["val_fraction"]),
        )
        rank_summaries: list[Any] = [None] * world_size
        dist.all_gather_object(rank_summaries, local_summaries)
        overfit_sample_uids = _select_bev_overfit_subset(
            rank_summaries,
            sample_count=overfit_sample_count,
        )
        overfit_sample_uid_sha256 = hashlib.sha256(
            "\n".join(overfit_sample_uids).encode("utf-8")
        ).hexdigest()
    validation_sample_uids: tuple[str, ...] | None = None
    validation_sample_uid_sha256 = ""
    validation_sample_count = 0
    if not overfit_sample_count:
        local_validation_limit = math.ceil(
            int(config.get("validation_sample_limit", 1024))
            / world_size
        )
        if stage is ReactiveTrainingStage.NUPLAN_FULL:
            assert local_bev_records is not None
            validation_sample_uids = select_bev_validation_sample_uids(
                local_bev_records,
                val_fraction=float(config["val_fraction"]),
                sample_limit=local_validation_limit,
            )
        else:
            validation_sample_uids = discover_validation_sample_uids(
                local_directories,
                val_fraction=float(config["val_fraction"]),
                sample_limit=local_validation_limit,
            )
        rank_validation_uids: list[Any] = [None] * world_size
        dist.all_gather_object(
            rank_validation_uids,
            validation_sample_uids,
        )
        global_validation_uids = sorted(
            str(sample_uid)
            for rank_uids in rank_validation_uids
            for sample_uid in rank_uids
        )
        if len(set(global_validation_uids)) != len(
            global_validation_uids
        ):
            raise ValueError(
                "Reactive validation subset contains duplicate samples"
            )
        validation_sample_count = len(global_validation_uids)
        validation_sample_uid_sha256 = hashlib.sha256(
            "\n".join(global_validation_uids).encode("utf-8")
        ).hexdigest()
    bev_pos_weights = (1.0,) * 8
    bev_repeat_factors = (1,) * 8
    bev_repeat_policy = None
    raw_bev_statistics = None
    effective_bev_statistics = None
    if stage is ReactiveTrainingStage.NUPLAN_FULL:
        assert local_bev_records is not None
        local_raw_statistics = summarize_bev_training_statistics(
            local_bev_records,
            val_fraction=float(config["val_fraction"]),
            sample_uids=overfit_sample_uids,
        )
        raw_bev_statistics = _all_reduce_bev_statistics(
            local_raw_statistics,
            device,
        )
        # Validate support before repetition can inflate sample counts.
        derive_bev_pos_weights(
            raw_bev_statistics,
            max_weight=float(config["bev_pos_weight_cap"]),
            min_positive_samples=int(
                config["bev_min_positive_samples"]
            ),
            min_positive_cells=int(
                config["bev_min_positive_cells"]
            ),
        )
        bev_repeat_factors = derive_bev_repeat_factors(
            raw_bev_statistics,
            frequency_threshold=float(
                config["bev_repeat_frequency_threshold"]
            ),
            max_repeat=int(config["bev_max_repeat"]),
        )
        if overfit_sample_count:
            effective_bev_statistics = raw_bev_statistics
        else:
            local_effective_statistics = (
                summarize_bev_training_statistics(
                    local_bev_records,
                    val_fraction=float(config["val_fraction"]),
                    repeat_factors=bev_repeat_factors,
                    sample_uids=overfit_sample_uids,
                )
            )
            effective_bev_statistics = _all_reduce_bev_statistics(
                local_effective_statistics,
                device,
            )
        bev_pos_weights = tuple(round(value, 6) for value in (
            derive_bev_pos_weights(
                effective_bev_statistics,
                max_weight=float(config["bev_pos_weight_cap"]),
            )
        ))
        if not overfit_sample_count:
            mean_repeat = (
                effective_bev_statistics.effective_exposure_count
                / effective_bev_statistics.sample_count
            )
            bev_repeat_policy = BEVClassRepeatPolicy(
                repeat_factors=bev_repeat_factors,
                mean_repeat=mean_repeat,
            )

    seed = int(config["training_seed"])
    _seed_epoch(seed, rank, 0)
    constructor_kwargs = reactive_model_kwargs(
        stage,
        num_views=plan.num_views,
    )
    model = AutoE2E(
        backbone=str(config["backbone"]),
        embed_dim=256,
        is_pretrained=bool(config["is_pretrained"]),
        **constructor_kwargs,
    ).to(device)
    lineage: dict[str, str] = {}
    parent_uri = str(config.get("parent_checkpoint_uri") or "")
    if parent_uri:
        parent_path = cache_root / "stage-a-parent.pt"
        parent_path.parent.mkdir(parents=True, exist_ok=True)
        _download_checkpoint(parent_uri, parent_path)
        lineage.update(load_stage_a_parent(model, parent_path))
    configure_model_for_stage(model, stage)
    objective = ReactiveMultitaskObjective(
        stage,
        bev_pos_weight=bev_pos_weights,
        bev_weight=float(config["bev_weight"]),
        route_weight=float(config["route_weight"]),
        corridor_pos_weight=float(config["corridor_pos_weight"]),
    ).to(device)

    model = prepare_model(
        model,
        parallel_strategy="ddp",
        parallel_strategy_kwargs={
            "find_unused_parameters": True,
            "gradient_as_bucket_view": True,
        },
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=1,
        min_lr=1e-5,
    )

    if overfit_sample_count:
        per_rank_overfit_samples = overfit_sample_count // world_size
        calculated_steps = max(1, math.ceil(
            per_rank_overfit_samples
            / (
                int(config["per_rank_batch_size"])
                * int(config["gradient_accumulation_steps"])
            )
        ))
    else:
        calculated_steps = optimizer_steps_per_epoch(
            total_samples=plan.total_samples,
            val_fraction=float(config["val_fraction"]),
            world_size=world_size,
            per_rank_batch_size=int(config["per_rank_batch_size"]),
            gradient_accumulation_steps=int(
                config["gradient_accumulation_steps"]
            ),
        )
    optimizer_steps = int(config["steps_per_epoch"]) or calculated_steps
    global_batch = (
        world_size
        * int(config["per_rank_batch_size"])
        * int(config["gradient_accumulation_steps"])
    )
    expected_resume = {
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "distributed_assignment_sha256": assignment_sha256,
        "distributed_global_batch": global_batch,
        "distributed_precision": str(config["precision"]),
        "distributed_world_size": world_size,
        "training_stage": stage.value,
        "bev_pos_weights": list(bev_pos_weights),
        "bev_repeat_factors": list(bev_repeat_factors),
        "bev_taxonomy_version": "bev_segmentation_v2",
        "validation_sample_count": validation_sample_count,
        "validation_sample_uid_sha256": validation_sample_uid_sha256,
    }
    start_epoch = 1
    best_selection_score = -float("inf")
    best_ade = float("inf")
    epoch_history: list[dict[str, Any]] = []
    restored = train.get_checkpoint()
    if overfit_sample_count and restored is not None:
        raise ValueError("BEV overfit mode cannot resume a checkpoint")
    if restored is not None:
        with restored.as_directory() as checkpoint_directory:
            (
                start_epoch,
                best_selection_score,
                best_ade,
                epoch_history,
            ) = _load_resume_checkpoint(
                checkpoint_directory,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected=expected_resume,
            )
    if start_epoch > int(config["epochs"]):
        raise ValueError(
            "resume checkpoint already completed the requested epochs"
        )

    hostnames: list[str | None] = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    unique_hostnames = sorted({str(host) for host in hostnames})
    if len(unique_hostnames) != world_size:
        raise RuntimeError(
            "one-GPU-per-node invariant failed: "
            f"world_size={world_size} hosts={unique_hostnames}"
        )

    model_config = {
        "backbone": str(config["backbone"]),
        "embed_dim": 256,
        "is_pretrained": bool(config["is_pretrained"]),
        **constructor_kwargs,
        "distributed_assignment_sha256": assignment_sha256,
        "distributed_global_batch": global_batch,
        "distributed_precision": str(config["precision"]),
        "distributed_world_size": world_size,
        "bev_pos_weights": list(bev_pos_weights),
        "bev_repeat_factors": list(bev_repeat_factors),
        "bev_taxonomy_version": "bev_segmentation_v2",
        "validation_sample_count": validation_sample_count,
        "validation_sample_uid_sha256": validation_sample_uid_sha256,
    }
    if raw_bev_statistics is not None:
        model_config["bev_raw_statistics"] = (
            raw_bev_statistics.metadata()
        )
    if effective_bev_statistics is not None:
        model_config["bev_effective_statistics"] = (
            effective_bev_statistics.metadata()
        )
    if overfit_sample_uids is not None:
        model_config["bev_overfit_sample_count"] = len(
            overfit_sample_uids
        )
        model_config["bev_overfit_sample_uid_sha256"] = (
            overfit_sample_uid_sha256
        )
        model_config["bev_overfit_staged_shard_count"] = len(
            assignment_shards
        )
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        _seed_epoch(seed, rank, epoch)
        train_loader = make_multi_dataset_loader(
            local_directories,
            batch_size=int(config["per_rank_batch_size"]),
            num_workers=int(config["num_loader_workers"]),
            split="train",
            val_fraction=float(config["val_fraction"]),
            shuffle=int(config["shuffle_buffer"]),
            shuffle_seed=seed + epoch,
            pin_memory=True,
            sample_uids=overfit_sample_uids,
            decode_future_frames=False,
            bev_repeat_policy=bev_repeat_policy,
            nodesplitter=passthrough_nodesplitter,
        )
        train_metrics = _train_fixed_steps(
            model,
            train_loader,
            objective,
            optimizer,
            device=device,
            optimizer_steps=optimizer_steps,
            gradient_accumulation_steps=int(
                config["gradient_accumulation_steps"]
            ),
            grad_clip=float(config["grad_clip"]),
            precision=str(config["precision"]),
        )
        validation_loader = make_multi_dataset_loader(
            local_directories,
            batch_size=int(config["per_rank_batch_size"]),
            num_workers=min(int(config["num_loader_workers"]), 1),
            split=("train" if overfit_sample_count else "val"),
            val_fraction=float(config["val_fraction"]),
            shuffle=0,
            pin_memory=True,
            max_active_loaders=1,
            sample_uids=(
                overfit_sample_uids
                if overfit_sample_count
                else validation_sample_uids
            ),
            decode_future_frames=False,
            nodesplitter=passthrough_nodesplitter,
        )
        validation = _evaluate_global_reactive(
            model,
            validation_loader,
            objective,
            stage=stage,
            device=device,
            probability_bins=int(config["bev_ap_bins"]),
            ade_scale_m=float(config["selection_ade_scale_m"]),
        )
        scheduler.step(validation["selection_score"])
        overfit_gate_pass = False
        if (
            overfit_sample_count
            and epoch == int(config["epochs"])
        ):
            from data_processing.reactive_training_artifacts import (
                BEV_SEGMENTATION_CLASSES,
            )

            class_average_precisions = {
                class_name: validation[
                    f"bev_{class_name}_average_precision"
                ]
                for class_name in BEV_SEGMENTATION_CLASSES
            }
            class_recalls = {
                class_name: validation[f"bev_{class_name}_recall"]
                for class_name in BEV_SEGMENTATION_CLASSES
            }
            minimum_ap_class = min(
                class_average_precisions,
                key=class_average_precisions.__getitem__,
            )
            minimum_recall_class = min(
                class_recalls,
                key=class_recalls.__getitem__,
            )
            minimum_ap = class_average_precisions[minimum_ap_class]
            minimum_recall = class_recalls[minimum_recall_class]
            overfit_gate_pass = (
                minimum_ap >= float(config["overfit_min_ap"])
                and minimum_recall
                >= float(config["overfit_min_recall"])
            )
            if not overfit_gate_pass:
                raise RuntimeError(
                    "BEV overfit gate failed: "
                    f"minimum_ap={minimum_ap:.6f} "
                    f"class={minimum_ap_class} "
                    f"minimum_recall={minimum_recall:.6f} "
                    f"class={minimum_recall_class}"
                )
        maximum_delta = _maximum_parameter_delta(
            model,
            world_size=world_size,
        )
        if maximum_delta > 1e-6:
            raise RuntimeError(
                "Reactive DDP replicas diverged: "
                f"maximum_parameter_delta={maximum_delta}"
            )
        rank_evidence: list[dict[str, Any] | None] = [
            None
        ] * world_size
        dist.all_gather_object(
            rank_evidence,
            {
                "assigned_samples": rank_sample_count,
                "consumed_samples": int(
                    train_metrics["local_consumed_samples"]
                ),
                "hostname": socket.gethostname(),
                "loader_restarts": int(
                    train_metrics["local_loader_restarts"]
                ),
                "rank": rank,
                "shards": [shard.identity for shard in rank_shards],
            },
        )
        ade_within_guard = validation["ade_6p4s_m"] <= (
            best_ade
            + float(config["selection_ade_regression_margin_m"])
        )
        is_best = (
            validation["selection_score"] > best_selection_score
            and ade_within_guard
        )
        checkpoint_selection_score = (
            validation["selection_score"]
            if ade_within_guard
            else -1.0
        )
        if is_best:
            best_selection_score = validation["selection_score"]
        best_ade = min(best_ade, validation["ade_6p4s_m"])
        checkpoint_sha256: str | None = None
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            checkpoint = None
            if rank == 0:
                checkpoint_path = (
                    Path(checkpoint_directory) / "checkpoint.pt"
                )
                checkpoint_sha256 = save_reactive_checkpoint(
                    checkpoint_path,
                    _base_model(model),
                    stage=stage,
                    dataset_manifest_sha256=(
                        plan.dataset_manifest_sha256
                    ),
                    epoch=epoch,
                    model_config=model_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=validation,
                    training_state={
                        "assignment_sha256": assignment_sha256,
                        "best_ade_6p4s_m": best_ade,
                        "best_selection_score": best_selection_score,
                        "global_batch": global_batch,
                        "optimizer_steps_per_epoch": optimizer_steps,
                        "rank_evidence": rank_evidence,
                        "world_size": world_size,
                    },
                    lineage=lineage,
                )
            checkpoint_digest: list[str | None] = [checkpoint_sha256]
            dist.broadcast_object_list(checkpoint_digest, src=0)
            metrics = {
                "checkpoint_sha256": str(checkpoint_digest[0]),
                "checkpoint_selection_score": (
                    checkpoint_selection_score
                ),
                "dataset_manifest_sha256": (
                    plan.dataset_manifest_sha256
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "epoch": epoch,
                "is_best": int(is_best),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "maximum_parameter_delta": maximum_delta,
                "optimizer_steps_per_epoch": optimizer_steps,
                "overfit_gate_pass": int(overfit_gate_pass),
                "overfit_sample_count": overfit_sample_count,
                "overfit_sample_uid_sha256": (
                    overfit_sample_uid_sha256
                ),
                "train_bev_segmentation": train_metrics[
                    "bev_segmentation"
                ],
                "train_bev_segmentation_bce": train_metrics[
                    "bev_segmentation_bce"
                ],
                "train_bev_segmentation_dice": train_metrics[
                    "bev_segmentation_dice"
                ],
                "train_loader_restarts": train_metrics[
                    "loader_restarts"
                ],
                "train_route_reconstruction": train_metrics[
                    "route_reconstruction"
                ],
                "train_total": train_metrics["total"],
                "train_trajectory": train_metrics["trajectory"],
                "validation_ade_6p4s_m": validation["ade_6p4s_m"],
                "validation_complete_samples": validation[
                    "complete_samples"
                ],
                "validation_fde_6p4s_m": validation["fde_6p4s_m"],
                "validation_selection_score": validation[
                    "selection_score"
                ],
                "validation_sample_count": validation_sample_count,
                "validation_sample_uid_sha256": (
                    validation_sample_uid_sha256
                ),
                "world_size": world_size,
            }
            for name, value in validation.items():
                if name not in {
                    "ade_6p4s_m",
                    "complete_samples",
                    "fde_6p4s_m",
                    "selection_score",
                }:
                    metrics[f"validation_{name}"] = value
            for class_index, value in enumerate(bev_pos_weights):
                metrics[f"bev_pos_weight_{class_index}"] = value
                metrics[f"bev_repeat_factor_{class_index}"] = (
                    bev_repeat_factors[class_index]
                )
            epoch_history.append(metrics)
            if rank == 0:
                history_path = (
                    Path(checkpoint_directory) / "history.json"
                )
                history_path.write_text(
                    json.dumps(
                        epoch_history,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="ascii",
                )
                checkpoint = Checkpoint.from_directory(
                    checkpoint_directory
                )
            train.report(metrics, checkpoint=checkpoint)


def _result_checkpoint_entry(entry) -> tuple[Any, dict[str, Any]]:
    if isinstance(entry, tuple) and len(entry) == 2:
        checkpoint, metrics = entry
    else:
        checkpoint = getattr(entry, "checkpoint", None)
        metrics = getattr(entry, "metrics", None)
    if checkpoint is None or not isinstance(metrics, Mapping):
        raise ValueError("Ray best checkpoint entry is invalid")
    return checkpoint, dict(metrics)


def _select_result_checkpoint(
    result,
    *,
    overfit_mode: bool,
) -> tuple[Any, dict[str, Any]]:
    """Select the gated final checkpoint or the best ADE-guarded checkpoint."""
    if result.checkpoint is None:
        raise RuntimeError("Reactive Ray training returned no checkpoint")
    final_metrics = dict(result.metrics)
    if overfit_mode:
        return result.checkpoint, final_metrics

    candidates = []
    for entry in getattr(result, "best_checkpoints", ()) or ():
        checkpoint, metrics = _result_checkpoint_entry(entry)
        if int(metrics.get("is_best", 0)) != 1:
            continue
        score = float(metrics.get("checkpoint_selection_score", -1.0))
        if not math.isfinite(score) or score < 0.0:
            continue
        candidates.append((
            score,
            int(metrics.get("epoch", 0)),
            checkpoint,
            metrics,
        ))
    if not candidates:
        raise RuntimeError(
            "Reactive Ray training retained no ADE-guarded best checkpoint"
        )
    _, _, checkpoint, metrics = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return checkpoint, metrics


def _checkpoint_history(checkpoint) -> list[dict[str, Any]]:
    with checkpoint.as_directory() as checkpoint_directory:
        path = Path(checkpoint_directory) / "history.json"
        history = json.loads(path.read_text(encoding="ascii"))
    if (
        not isinstance(history, list)
        or not history
        or any(not isinstance(item, dict) for item in history)
    ):
        raise ValueError("Reactive Ray checkpoint history is invalid")
    return history


def run_reactive_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Launch the fixed-size Ray Train worker group."""
    validate_reactive_stage_config(config)
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer

    if not ray.is_initialized():
        ray.init(address="auto")
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=dict(config),
        scaling_config=train.ScalingConfig(
            num_workers=int(config["num_workers"]),
            use_gpu=True,
            resources_per_worker={
                "CPU": int(config["worker_cpus"]),
                "GPU": 1,
            },
            placement_strategy="SPREAD",
        ),
        run_config=train.RunConfig(
            name=str(config["run_name"]),
            storage_path=str(config["storage_path"]),
            failure_config=train.FailureConfig(
                max_failures=(
                    0 if int(config["overfit_sample_count"]) else 2
                ),
            ),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="checkpoint_selection_score",
                checkpoint_score_order="max",
            ),
        ),
    )
    result = trainer.fit()
    if result.checkpoint is None:
        raise RuntimeError("Reactive Ray training returned no checkpoint")
    history = _checkpoint_history(result.checkpoint)
    checkpoint, metrics = _select_result_checkpoint(
        result,
        overfit_mode=bool(int(config["overfit_sample_count"])),
    )
    checkpoint_uri = normalize_ray_checkpoint_uri(
        str(checkpoint.path),
        str(config["storage_path"]),
    )
    if int(metrics.get("world_size", 0)) != int(config["num_workers"]):
        raise RuntimeError(
            f"Reactive Ray result has unexpected world size: {metrics}"
        )
    selected_digest = str(metrics.get("checkpoint_sha256", ""))
    selected_rows = [
        row
        for row in history
        if str(row.get("checkpoint_sha256", "")) == selected_digest
    ]
    if (
        len(selected_rows) != 1
        or int(selected_rows[0].get("epoch", -1))
        != int(metrics.get("epoch", -2))
    ):
        raise RuntimeError(
            "selected Reactive checkpoint does not match epoch history"
        )
    final_metrics = dict(result.metrics)
    if (
        str(history[-1].get("checkpoint_sha256", ""))
        != str(final_metrics.get("checkpoint_sha256", ""))
    ):
        raise RuntimeError(
            "final Reactive checkpoint does not match epoch history"
        )
    return {
        "checkpoint_file_uri": f"{checkpoint_uri}/checkpoint.pt",
        "checkpoint_uri": checkpoint_uri,
        "final_metrics": final_metrics,
        "history": history,
        "metrics": metrics,
        "selected_epoch": int(metrics["epoch"]),
        "run_name": str(config["run_name"]),
        "storage_path": str(config["storage_path"]),
    }
