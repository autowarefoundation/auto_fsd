"""Distributed Reactive dataset and fixed-step contracts."""

from __future__ import annotations

import hashlib
import json

import pytest

from data_parsing.pre_extracted import (
    make_pre_extracted_loader,
    passthrough_nodesplitter,
)
from distributed_training.reactive_canary_data import (
    write_reactive_canary_dataset,
)
from distributed_training.reactive_data import (
    RestartingIterator,
    assign_reactive_shards,
    build_reactive_dataset_plan,
    optimizer_steps_per_epoch,
    reactive_assignment_sha256,
    stage_rank_reactive_shards,
)
from distributed_training.reactive_stage import (
    _finalize_trajectory_validation,
    _trajectory_validation_batch_sums,
    clip_finite_gradients_float64,
    normalize_ray_checkpoint_uri,
    reactive_worker_resources,
    select_best_ray_checkpoint,
    select_distributed_validation_groups,
    validate_reactive_stage_config,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from training.reactive_multitask import ReactiveTrainingStage


def _write_source(
    root,
    *,
    dataset: str,
    shard_counts: list[int],
    include_bev: bool,
    num_views: int,
):
    root.mkdir()
    names = []
    hashes = {}
    counts = {}
    for index, sample_count in enumerate(shard_counts):
        name = f"part-{index:03d}.tar"
        payload = f"{root.name}:{index}:{sample_count}".encode()
        (root / name).write_bytes(payload)
        names.append(name)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        counts[name] = sample_count
    manifest = {
        "dataset": dataset,
        "has_bev_segmentation": include_bev,
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
        "map_context_channels": 14,
        "navigation_geometry": (
            AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ),
        "num_views": num_views,
        "partition_id": root.name,
        "route_channels": 2,
        "shard_names": names,
        "shard_sample_counts": counts,
        "shard_sha256": hashes,
        "source_revision": "test-revision",
        "total_samples": sum(shard_counts),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="ascii",
    )
    return manifest


def test_dataset_plan_and_assignment_are_deterministic(tmp_path):
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    _write_source(
        source_a,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[9, 4],
        include_bev=True,
        num_views=8,
    )
    _write_source(
        source_b,
        dataset="nuplan/nuplan-v1.1",
        shard_counts=[8, 5],
        include_bev=True,
        num_views=8,
    )

    plan = build_reactive_dataset_plan(
        [str(source_b), str(source_a)],
        stage=ReactiveTrainingStage.NUPLAN_FULL,
    )
    assignments = assign_reactive_shards(
        plan.shards,
        world_size=2,
    )

    assert plan.total_samples == 26
    assert plan.num_views == 8
    assert [sum(item.sample_count for item in rank) for rank in assignments] == [
        13,
        13,
    ]
    assert assignments == assign_reactive_shards(
        tuple(reversed(plan.shards)),
        world_size=2,
    )
    assert reactive_assignment_sha256(assignments) == (
        reactive_assignment_sha256(assignments)
    )


def test_dataset_plan_rejects_legacy_manifest_without_per_shard_counts(
    tmp_path,
):
    source = tmp_path / "legacy"
    manifest = _write_source(
        source,
        dataset="yaak-ai/L2D",
        shard_counts=[2],
        include_bev=False,
        num_views=6,
    )
    manifest.pop("shard_sample_counts")
    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="shard_sample_counts"):
        build_reactive_dataset_plan(
            [str(source)],
            stage=ReactiveTrainingStage.L2D_CONTINUATION,
        )


def test_rank_staging_verifies_tar_and_manifest_digests(tmp_path):
    source = tmp_path / "source"
    _write_source(
        source,
        dataset="yaak-ai/L2D",
        shard_counts=[3, 2],
        include_bev=False,
        num_views=6,
    )
    plan = build_reactive_dataset_plan(
        [str(source)],
        stage=ReactiveTrainingStage.L2D_CONTINUATION,
    )
    assignments = assign_reactive_shards(
        plan.shards,
        world_size=2,
    )

    local_directories = stage_rank_reactive_shards(
        assignments[0],
        cache_root=tmp_path / "cache",
    )

    assert len(local_directories) == 1
    staged = tmp_path / "cache" / hashlib.sha256(
        str(source).encode()
    ).hexdigest()[:16]
    assert (staged / "manifest.json").is_file()
    assert sorted(path.name for path in staged.glob("*.tar")) == [
        assignments[0][0].shard_name
    ]

    source_shard = source / assignments[1][0].shard_name
    source_shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        stage_rank_reactive_shards(
            assignments[1],
            cache_root=tmp_path / "cache-corrupt",
        )


def test_fixed_step_count_uses_global_effective_batch():
    assert optimizer_steps_per_epoch(
        total_samples=100,
        val_fraction=0.1,
        world_size=8,
        per_rank_batch_size=1,
        gradient_accumulation_steps=1,
    ) == 12
    assert optimizer_steps_per_epoch(
        total_samples=100,
        val_fraction=0.1,
        world_size=4,
        per_rank_batch_size=1,
        gradient_accumulation_steps=4,
    ) == 6


def test_exact_validation_groups_keep_training_samples_on_every_rank():
    selected = select_distributed_validation_groups(
        [
            {
                "nuplan-log-de6928217e1d599af467": 1,
                "nuplan-log-6c2206ef7923aaa0fcc1": 2,
                "nuplan-log-03e7cb00494f4098167a": 1,
            },
            {
                "nuplan-log-237f1f6f15baa6a1faa6": 1,
                "nuplan-log-4d499edda36f30464fe8": 1,
                "nuplan-log-6c2206ef7923aaa0fcc1": 2,
            },
        ],
        val_fraction=0.25,
    )

    assert selected == ("nuplan-log-de6928217e1d599af467",)


def test_exact_validation_groups_fail_when_a_rank_would_be_empty():
    with pytest.raises(ValueError, match="repack shards across ranks"):
        select_distributed_validation_groups(
            [
                {"scene-a": 4},
                {"scene-b": 4},
            ],
            val_fraction=0.5,
        )


def test_restarting_iterator_repeats_finite_loader():
    iterator = RestartingIterator([1, 2])

    assert [next(iterator) for _ in range(5)] == [1, 2, 1, 2, 1]
    assert iterator.restarts == 2

    with pytest.raises(ValueError, match="yielded no batches"):
        next(RestartingIterator([]))


def test_float64_gradient_clipping_handles_large_finite_values():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([5.0e23, -2.0e23])

    norm, finite = clip_finite_gradients_float64([parameter], 1.0)

    assert finite
    assert torch.isfinite(norm)
    assert norm.item() > 1.0e23
    assert torch.linalg.vector_norm(parameter.grad).item() == pytest.approx(
        1.0,
        rel=1e-6,
    )


def test_float64_gradient_clipping_rejects_non_finite_values():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float("nan"), 1.0])

    _, finite = clip_finite_gradients_float64([parameter], 1.0)

    assert not finite
    assert torch.isnan(parameter.grad[0])


def test_trajectory_validation_prefers_complete_horizon_metrics():
    torch = pytest.importorskip("torch")
    predicted = torch.zeros((2, 4, 2))
    target = torch.zeros_like(predicted)
    target[:, :, 0] = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )

    metrics = _finalize_trajectory_validation(
        _trajectory_validation_batch_sums(
            predicted,
            target,
            valid,
        )
    )

    assert metrics["ade_6p4s_m"] == pytest.approx(2.5)
    assert metrics["fde_6p4s_m"] == pytest.approx(4.0)
    assert metrics["selection_ade_m"] == pytest.approx(2.5)
    assert metrics["available_horizon_ade_m"] == pytest.approx(2.75)
    assert metrics["available_horizon_fde_m"] == pytest.approx(4.0)
    assert metrics["complete_samples"] == 1.0
    assert metrics["valid_samples"] == 2.0
    assert metrics["valid_points"] == 6.0
    assert metrics["partial_horizon_fallback"] == 0.0


def test_trajectory_validation_falls_back_to_available_horizon():
    torch = pytest.importorskip("torch")
    predicted = torch.zeros((2, 4, 2))
    target = torch.zeros_like(predicted)
    target[:, :, 0] = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    metrics = _finalize_trajectory_validation(
        _trajectory_validation_batch_sums(
            predicted,
            target,
            valid,
        )
    )

    assert "ade_6p4s_m" not in metrics
    assert "fde_6p4s_m" not in metrics
    assert metrics["selection_ade_m"] == pytest.approx(2.75)
    assert metrics["selection_fde_m"] == pytest.approx(4.0)
    assert metrics["complete_samples"] == 0.0
    assert metrics["valid_samples"] == 2.0
    assert metrics["valid_points"] == 5.0
    assert metrics["partial_horizon_fallback"] == 1.0


def test_trajectory_validation_rejects_batches_without_valid_points():
    torch = pytest.importorskip("torch")
    predicted = torch.zeros((1, 4, 2))
    target = torch.zeros_like(predicted)
    valid = torch.zeros((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="no valid trajectories"):
        _finalize_trajectory_validation(
            _trajectory_validation_batch_sums(
                predicted,
                target,
                valid,
            )
        )


def test_ray_checkpoint_uri_restores_s3_scheme_within_storage():
    storage = "s3://checkpoints/ray-train"

    assert normalize_ray_checkpoint_uri(
        "checkpoints/ray-train/run/checkpoint_0001",
        storage,
    ) == "s3://checkpoints/ray-train/run/checkpoint_0001"
    assert normalize_ray_checkpoint_uri(
        "s3://checkpoints/ray-train/run/checkpoint_0001/",
        storage,
    ) == "s3://checkpoints/ray-train/run/checkpoint_0001"


def test_ray_checkpoint_uri_rejects_paths_outside_storage():
    with pytest.raises(ValueError, match="outside"):
        normalize_ray_checkpoint_uri(
            "other-bucket/ray-train/run/checkpoint_0001",
            "s3://checkpoints/ray-train",
        )


class _FakeCheckpoint:
    def __init__(self, path: str):
        self.path = path


class _FakeRayResult:
    def __init__(self, best_checkpoints):
        self.best_checkpoints = best_checkpoints

    def get_best_checkpoint(self, *, metric: str, mode: str):
        assert mode == "min"
        return min(
            self.best_checkpoints,
            key=lambda item: float(item[1][metric]),
        )[0]


def test_select_best_ray_checkpoint_returns_matching_epoch_metrics():
    latest = _FakeCheckpoint("s3://checkpoints/run/checkpoint-epoch-4")
    best = _FakeCheckpoint("s3://checkpoints/run/checkpoint-epoch-3/")
    result = _FakeRayResult([
        (
            latest,
            {
                "checkpoint_sha256": "4" * 64,
                "epoch": 4,
                "validation_selection_ade_m": 29.87,
            },
        ),
        (
            best,
            {
                "checkpoint_sha256": "3" * 64,
                "epoch": 3,
                "validation_selection_ade_m": 29.74,
            },
        ),
    ])

    checkpoint, metrics = select_best_ray_checkpoint(result)

    assert checkpoint is best
    assert metrics["epoch"] == 3
    assert metrics["checkpoint_sha256"] == "3" * 64


def test_select_best_ray_checkpoint_rejects_missing_scored_metrics():
    best = _FakeCheckpoint("s3://checkpoints/run/checkpoint-epoch-3")
    result = _FakeRayResult([
        (
            best,
            {
                "checkpoint_sha256": "invalid",
                "validation_selection_ade_m": 29.74,
            },
        )
    ])

    with pytest.raises(RuntimeError, match="checkpoint_sha256"):
        select_best_ray_checkpoint(result)


def test_rank_owned_nodesplitter_preserves_every_assigned_shard():
    urls = ["rank-000-part-000.tar", "rank-000-part-003.tar"]

    assert list(passthrough_nodesplitter(iter(urls))) == urls


def _stage_config(stage: str) -> dict[str, object]:
    return {
        "backbone": "swin_v2_tiny",
        "bev_pos_weights": [1.0] * 8,
        "epochs": 2,
        "grad_clip": 1.0,
        "gradient_accumulation_steps": 1,
        "is_pretrained": False,
        "learning_rate": 1e-4,
        "num_loader_workers": 1,
        "num_workers": 8,
        "parent_checkpoint_uri": (
            "s3://checkpoints/stage-a/checkpoint.pt"
            if stage == "l2d_continuation"
            else ""
        ),
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "run_name": "reactive-stage-test",
        "source_uris": ["s3://datasets/reactive"],
        "stage": stage,
        "steps_per_epoch": 0,
        "storage_path": "s3://checkpoints/ray-train",
        "val_fraction": 0.1,
        "weight_decay": 0.01,
        "worker_cpus": 4,
    }


@pytest.mark.parametrize("stage", ["nuplan_full", "l2d_continuation"])
def test_validate_stage_config_accepts_locked_program(stage):
    validate_reactive_stage_config(_stage_config(stage))


def test_validate_stage_config_rejects_parent_and_batch_contract_changes():
    stage_a = _stage_config("nuplan_full")
    stage_a["parent_checkpoint_uri"] = "s3://checkpoints/parent.pt"
    with pytest.raises(ValueError, match="Stage A"):
        validate_reactive_stage_config(stage_a)

    stage_b = _stage_config("l2d_continuation")
    stage_b["per_rank_batch_size"] = 2
    with pytest.raises(ValueError, match="per_rank_batch_size"):
        validate_reactive_stage_config(stage_b)


def test_reactive_worker_resources_follow_the_pod_cpu_contract():
    config = _stage_config("nuplan_full")
    config["worker_cpus"] = 3
    assert reactive_worker_resources(config) == {"CPU": 3, "GPU": 1}

    config["worker_cpus"] = 0
    with pytest.raises(ValueError, match="worker_cpus"):
        validate_reactive_stage_config(config)


@pytest.mark.parametrize(
    ("stage", "expected_views", "expected_bev"),
    [
        (ReactiveTrainingStage.NUPLAN_FULL, 8, True),
        (ReactiveTrainingStage.L2D_CONTINUATION, 6, False),
    ],
)
def test_canary_dataset_uses_production_loader_contract(
    tmp_path,
    stage,
    expected_views,
    expected_bev,
):
    dataset = tmp_path / stage.value
    manifest = write_reactive_canary_dataset(
        dataset,
        stage=stage,
    )
    plan = build_reactive_dataset_plan(
        [str(dataset)],
        stage=stage,
    )
    train_batches = list(make_pre_extracted_loader(
        str(dataset),
        batch_size=1,
        num_workers=0,
        split="train",
        val_fraction=0.5,
        shuffle=0,
        decode_future_frames=False,
        nodesplitter=passthrough_nodesplitter,
    ))
    validation_batches = list(make_pre_extracted_loader(
        str(dataset),
        batch_size=1,
        num_workers=0,
        split="val",
        val_fraction=0.5,
        shuffle=0,
        decode_future_frames=False,
        nodesplitter=passthrough_nodesplitter,
    ))

    assert manifest["total_samples"] == 6
    assert plan.total_samples == 6
    assert len(train_batches) == 4
    assert len(validation_batches) == 2
    sample = train_batches[0]
    assert sample["visual_tiles"].shape == (
        1,
        expected_views,
        3,
        256,
        256,
    )
    assert sample["map_context"].shape == (1, 14, 450, 300)
    assert sample["route_mask"].shape == (1, 2, 450, 300)
    assert bool(sample["bev_segmentation_available"][0]) is expected_bev
