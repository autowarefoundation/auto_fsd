"""Deterministic parallel packing contracts for full nuPlan snapshots."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from data_parsing.nuplan import packing as nuplan_packing
from data_parsing.nuplan.packing import (
    _NuPlanNoScenariosError,
    _NuPlanPackPartition,
    _NuPlanPackWorkerConfig,
    _merge_nuplan_pack_partitions,
    _pack_nuplan_partition,
    _partition_weighted_nuplan_db_files,
)


def _write_shard(path: Path, sample_uids: list[str]) -> str:
    with tarfile.open(path, mode="w") as archive:
        for sample_uid in sample_uids:
            payload = b"{}"
            info = tarfile.TarInfo(f"{sample_uid}.meta.json")
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nuplan_db_partitioning_is_deterministic_and_balanced(
    tmp_path: Path,
):
    weighted = [
        (tmp_path / "a.db", 100),
        (tmp_path / "b.db", 80),
        (tmp_path / "c.db", 60),
        (tmp_path / "d.db", 40),
        (tmp_path / "empty.db", 0),
    ]

    first = _partition_weighted_nuplan_db_files(weighted, 2)
    second = _partition_weighted_nuplan_db_files(
        list(reversed(weighted)),
        2,
    )

    assert first == second
    assert [item.scenario_estimate for item in first] == [140, 140]
    assert sorted(
        Path(path).name
        for item in first
        for path in item.db_files
    ) == ["a.db", "b.db", "c.db", "d.db", "empty.db"]


def test_nuplan_partition_merge_revalidates_shards_and_uids(
    tmp_path: Path,
):
    output = tmp_path / "output"
    partition_root = output / ".partitions"
    output.mkdir()
    partition_root.mkdir()
    partitions = _partition_weighted_nuplan_db_files(
        [
            (tmp_path / "a.db", 10),
            (tmp_path / "b.db", 9),
        ],
        2,
    )
    manifests = []
    expected_uids = []
    for partition in partitions:
        worker_directory = (
            partition_root / f"worker-{partition.index:03d}"
        )
        worker_directory.mkdir()
        sample_uids = [
            f"nuplan-worker-{partition.index}-0",
            f"nuplan-worker-{partition.index}-1",
        ]
        expected_uids.extend(sample_uids)
        shard_name = "nuplan-000000.tar"
        shard_hash = _write_shard(
            worker_directory / shard_name,
            sample_uids,
        )
        manifests.append({
            "bev_segmentation_count": 2,
            "dataset": "nuplan/nuplan-v1.1",
            "rejected_samples": [],
            "rejection_count": 0,
            "rejection_fraction": 0.0,
            "sample_uid_digest": "worker-only",
            "shard_names": [shard_name],
            "shard_sample_counts": {shard_name: 2},
            "shard_sha256": {shard_name: shard_hash},
            "split_group_count": 1,
            "total_samples": 2,
            "trajectory_xy_count": 2,
        })

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=manifests,
        max_rejection_fraction=0.0,
    )

    assert merged["packing_workers"] == 2
    assert merged["total_samples"] == 4
    assert merged["split_group_count"] == 2
    assert merged["shard_names"] == [
        "nuplan-000000.tar",
        "nuplan-000001.tar",
    ]
    assert merged["sample_uid_digest"] == hashlib.sha256(
        "\n".join(sorted(expected_uids)).encode("ascii")
    ).hexdigest()
    assert not partition_root.exists()
    shard_hashes = merged["shard_sha256"]
    assert isinstance(shard_hashes, dict)
    for shard_name in merged["shard_names"]:
        shard_path = output / shard_name
        assert shard_hashes[shard_name] == hashlib.sha256(
            shard_path.read_bytes()
        ).hexdigest()


def test_nuplan_partition_merge_skips_filtered_empty_worker(
    tmp_path: Path,
):
    output = tmp_path / "output"
    partition_root = output / ".partitions"
    output.mkdir()
    partition_root.mkdir()
    partitions = _partition_weighted_nuplan_db_files(
        [
            (tmp_path / "empty-after-filter.db", 10),
            (tmp_path / "accepted.db", 9),
        ],
        2,
    )
    empty_partition, accepted_partition = partitions
    (
        partition_root / f"worker-{empty_partition.index:03d}"
    ).mkdir()
    worker_directory = (
        partition_root / f"worker-{accepted_partition.index:03d}"
    )
    worker_directory.mkdir()
    shard_name = "nuplan-000000.tar"
    shard_hash = _write_shard(
        worker_directory / shard_name,
        ["nuplan-accepted-0"],
    )
    accepted_manifest = {
        "bev_segmentation_count": 1,
        "dataset": "nuplan/nuplan-v1.1",
        "rejected_samples": [],
        "rejection_count": 0,
        "rejection_fraction": 0.0,
        "sample_uid_digest": "worker-only",
        "shard_names": [shard_name],
        "shard_sample_counts": {shard_name: 1},
        "shard_sha256": {shard_name: shard_hash},
        "split_group_count": 1,
        "total_samples": 1,
        "trajectory_xy_count": 1,
    }

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=[None, accepted_manifest],
        max_rejection_fraction=0.0,
    )

    assert merged["total_samples"] == 1
    assert merged["packing_workers"] == 2
    assert merged["packing_nonempty_workers"] == 1
    assert merged["packing_partitions"] == [
        {
            "db_file_count": 1,
            "is_empty": True,
            "scenario_estimate": 10,
        },
        {
            "db_file_count": 1,
            "is_empty": False,
            "scenario_estimate": 9,
        },
    ]
    assert not partition_root.exists()


def _worker_config(tmp_path: Path) -> _NuPlanPackWorkerConfig:
    return _NuPlanPackWorkerConfig(
        partition=_NuPlanPackPartition(
            index=0,
            db_files=(str(tmp_path / "empty.db"),),
            scenario_estimate=10,
        ),
        data_root=str(tmp_path / "data"),
        map_root=str(tmp_path / "maps"),
        sensor_root=str(tmp_path / "sensors"),
        output_directory=str(tmp_path / "output"),
        source_revision="nuplan-v1.1-complete",
        map_version="nuplan-maps-v1.0",
        image_size=256,
        samples_per_shard=100,
    )


def test_nuplan_partition_worker_reports_filtered_empty_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def filtered_empty(**_kwargs):
        raise _NuPlanNoScenariosError(
            "nuPlan scenario builder returned no scenarios"
        )

    monkeypatch.setattr(
        nuplan_packing,
        "pack_nuplan_local_dataset",
        filtered_empty,
    )

    assert _pack_nuplan_partition(_worker_config(tmp_path)) is None


def test_nuplan_partition_worker_preserves_other_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def corrupted(**_kwargs):
        raise ValueError("corrupt nuPlan partition")

    monkeypatch.setattr(
        nuplan_packing,
        "pack_nuplan_local_dataset",
        corrupted,
    )

    with pytest.raises(ValueError, match="corrupt nuPlan partition"):
        _pack_nuplan_partition(_worker_config(tmp_path))
