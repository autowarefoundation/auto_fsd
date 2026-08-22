"""Deterministic parallel packing contracts for full nuPlan snapshots."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from data_parsing.nuplan import packing as nuplan_packing
from data_parsing.nuplan.packing import (
    NUPLAN_PACK_MANIFEST_VERSION,
    _NUPLAN_MANIFEST_INVARIANT_KEYS,
    _NuPlanNoScenariosError,
    _NuPlanPackPartition,
    _NuPlanPackWorkerConfig,
    _initialize_nuplan_pack_worker,
    _merge_nuplan_pack_partitions,
    _nuplan_db_scenario_count,
    _pack_nuplan_partition,
    _partition_weighted_nuplan_db_files,
    pack_nuplan_local_dataset,
    pack_nuplan_reactive_scenarios,
)


def _scenario(
    sample_uid: str,
    split_group_uid: str,
    *,
    rejected: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        log_name=split_group_uid,
        rejected=rejected,
        sample_uid=sample_uid,
        split_group_uid=split_group_uid,
        token=f"token-{sample_uid}",
    )


def _sample_builder(
    scenario: SimpleNamespace,
    **_kwargs: object,
) -> tuple[str, str, dict[str, bytes]]:
    if scenario.rejected:
        raise ValueError("synthetic rejection")
    return (
        scenario.sample_uid,
        scenario.split_group_uid,
        {"meta.json": b"{}"},
    )


def _worker_manifest(
    worker_directory: Path,
    scenarios: list[SimpleNamespace],
) -> dict[str, object]:
    return pack_nuplan_reactive_scenarios(
        scenarios,
        worker_directory,
        source_revision="nuplan-v1.1-mini",
        map_version="nuplan-maps-v1.0",
        samples_per_shard=2,
        max_rejection_fraction=1.0,
        require_accepted=False,
        sample_builder=_sample_builder,
    )


def _partition_layout(
    tmp_path: Path,
    weighted: list[tuple[str, int]],
) -> tuple[Path, Path, list[_NuPlanPackPartition]]:
    output = tmp_path / "output"
    partition_root = output / ".partitions"
    output.mkdir()
    partition_root.mkdir()
    partitions = _partition_weighted_nuplan_db_files(
        [(tmp_path / name, weight) for name, weight in weighted],
        len(weighted),
    )
    return output, partition_root, partitions


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
    assert [
        [Path(path).name for path in partition.db_files]
        for partition in first
    ] == [
        ["a.db", "d.db", "empty.db"],
        ["b.db", "c.db"],
    ]


def test_nuplan_db_partitioning_rejects_duplicate_paths(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="contains duplicates"):
        _partition_weighted_nuplan_db_files(
            [(tmp_path / "same.db", 2), (tmp_path / "same.db", 1)],
            2,
        )


def test_nuplan_db_scenario_count_quotes_uri_path(tmp_path: Path):
    db_path = tmp_path / "log?special.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE scenario_tag (token TEXT)")
        connection.executemany(
            "INSERT INTO scenario_tag VALUES (?)",
            [("a",), ("b",), ("c",)],
        )
        connection.commit()
    finally:
        connection.close()

    assert _nuplan_db_scenario_count(db_path) == 3


def test_nuplan_partition_merge_preserves_real_v2_manifest(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10), ("b.db", 9)],
    )
    manifests = []
    expected_uids = []
    for partition in partitions:
        sample_uids = [
            f"nuplan-worker-{partition.index}-0",
            f"nuplan-worker-{partition.index}-1",
        ]
        expected_uids.extend(sample_uids)
        manifests.append(_worker_manifest(
            partition_root / f"worker-{partition.index:03d}",
            [
                _scenario(
                    sample_uid,
                    f"group-{partition.index}",
                )
                for sample_uid in sample_uids
            ],
        ))

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=manifests,
        max_rejection_fraction=0.0,
    )

    assert set(manifests[0]).issubset(merged)
    assert all(
        merged[key] == manifests[0][key]
        for key in _NUPLAN_MANIFEST_INVARIANT_KEYS
    )
    assert merged["schema_version"] == NUPLAN_PACK_MANIFEST_VERSION
    assert merged["packing_workers"] == 2
    assert merged["total_samples"] == 4
    assert merged["bev_segmentation_count"] == 4
    assert merged["bev_statistics_count"] == 4
    assert merged["split_group_count"] == 2
    assert merged["split_group_uids"] == ["group-0", "group-1"]
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
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("empty-after-filter.db", 10), ("accepted.db", 9)],
    )
    empty_partition, accepted_partition = partitions
    (
        partition_root / f"worker-{empty_partition.index:03d}"
    ).mkdir()
    accepted_manifest = _worker_manifest(
        partition_root / f"worker-{accepted_partition.index:03d}",
        [_scenario("nuplan-accepted-0", "group-accepted")],
    )

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=[None, accepted_manifest],
        max_rejection_fraction=0.0,
    )

    assert merged["total_samples"] == 1
    assert merged["bev_statistics_count"] == 1
    assert merged["packing_workers"] == 2
    assert merged["packing_nonempty_workers"] == 1
    assert merged["packing_partitions"] == [
        {
            "accepted_count": 0,
            "db_file_count": 1,
            "is_empty": True,
            "rejected_count": 0,
            "scenario_estimate": 10,
        },
        {
            "accepted_count": 1,
            "db_file_count": 1,
            "is_empty": False,
            "rejected_count": 0,
            "scenario_estimate": 9,
        },
    ]
    assert not partition_root.exists()


def test_nuplan_partition_merge_accounts_for_all_rejected_worker(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("rejected.db", 10), ("accepted.db", 9)],
    )
    rejected_manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("rejected", "group-rejected", rejected=True)],
    )
    accepted_manifest = _worker_manifest(
        partition_root / "worker-001",
        [_scenario("accepted", "group-accepted")],
    )

    merged = _merge_nuplan_pack_partitions(
        output=output,
        partition_root=partition_root,
        partitions=partitions,
        manifests=[rejected_manifest, accepted_manifest],
        max_rejection_fraction=0.5,
    )

    assert rejected_manifest["total_samples"] == 0
    assert rejected_manifest["rejection_count"] == 1
    assert rejected_manifest["shard_names"] == []
    assert merged["total_samples"] == 1
    assert merged["rejection_count"] == 1
    assert merged["rejection_fraction"] == 0.5
    packing_partitions = cast(
        list[dict[str, object]],
        merged["packing_partitions"],
    )
    assert packing_partitions[0]["rejected_count"] == 1


def test_nuplan_partition_merge_enforces_global_rejection_before_move(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10), ("b.db", 9)],
    )
    manifests = [
        _worker_manifest(
            partition_root / f"worker-{index:03d}",
            [
                _scenario(f"accepted-{index}", f"group-{index}"),
                _scenario(
                    f"rejected-{index}",
                    f"group-{index}",
                    rejected=True,
                ),
            ],
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="rejection policy failed"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=manifests,
            max_rejection_fraction=0.49,
        )

    assert not list(output.glob("*.tar"))
    assert len(list(partition_root.rglob("*.tar"))) == 2


def test_nuplan_partition_merge_rejects_duplicate_sample_uid(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10), ("b.db", 9)],
    )
    manifests = [
        _worker_manifest(
            partition_root / f"worker-{index:03d}",
            [_scenario("duplicate", f"group-{index}")],
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="duplicate sample UIDs"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=manifests,
            max_rejection_fraction=0.0,
        )

    assert not list(output.glob("*.tar"))


def test_nuplan_partition_merge_rejects_corrupted_shard(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10)],
    )
    manifest = _worker_manifest(
        partition_root / "worker-000",
        [_scenario("accepted", "group-accepted")],
    )
    shard_names = cast(list[str], manifest["shard_names"])
    shard_path = partition_root / "worker-000" / shard_names[0]
    with shard_path.open("ab") as shard:
        shard.write(b"corruption")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest],
            max_rejection_fraction=0.0,
        )


def test_nuplan_partition_merge_rejects_unreported_shard(
    tmp_path: Path,
):
    output, partition_root, partitions = _partition_layout(
        tmp_path,
        [("a.db", 10)],
    )
    worker_directory = partition_root / "worker-000"
    manifest = _worker_manifest(
        worker_directory,
        [_scenario("accepted", "group-accepted")],
    )
    shard_names = cast(list[str], manifest["shard_names"])
    shutil.copyfile(
        worker_directory / shard_names[0],
        worker_directory / "nuplan-unreported.tar",
    )

    with pytest.raises(ValueError, match="account for every shard"):
        _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=[manifest],
            max_rejection_fraction=0.0,
        )


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


def test_nuplan_partition_worker_allows_all_rejected_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def all_rejected(**kwargs):
        captured.update(kwargs)
        return {"rejection_count": 2, "total_samples": 0}

    monkeypatch.setattr(
        nuplan_packing,
        "pack_nuplan_local_dataset",
        all_rejected,
    )

    assert _pack_nuplan_partition(_worker_config(tmp_path)) == {
        "rejection_count": 2,
        "total_samples": 0,
    }
    assert captured["require_accepted"] is False


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


def _create_tagged_db(path: Path, scenario_count: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE scenario_tag (token TEXT)")
        connection.executemany(
            "INSERT INTO scenario_tag VALUES (?)",
            [(f"token-{index}",) for index in range(scenario_count)],
        )
        connection.commit()
    finally:
        connection.close()


def test_nuplan_local_parallel_path_builds_spawned_worker_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = tmp_path / "data"
    map_root = tmp_path / "maps"
    sensor_root = tmp_path / "sensors"
    data_root.mkdir()
    map_root.mkdir()
    sensor_root.mkdir()
    db_files = [data_root / "a.db", data_root / "b.db"]
    for db_file in db_files:
        _create_tagged_db(db_file, 2)
    executor_state: dict[str, object] = {}
    context = object()

    class InlineExecutor:
        def __init__(self, **kwargs: object):
            executor_state.update(kwargs)

        def __enter__(self) -> InlineExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def map(self, function, configs):
            executor_state["function"] = function
            executor_state["configs"] = list(configs)
            return [
                function(config)
                for config in executor_state["configs"]
            ]

    def fake_partition(
        config: _NuPlanPackWorkerConfig,
    ) -> dict[str, object]:
        return _worker_manifest(
            Path(config.output_directory),
            [_scenario(
                f"sample-{config.partition.index}",
                f"group-{config.partition.index}",
            )],
        )

    monkeypatch.setattr(
        nuplan_packing.multiprocessing,
        "get_context",
        lambda method: (
            executor_state.update(context_method=method) or context
        ),
    )
    monkeypatch.setattr(
        nuplan_packing,
        "ProcessPoolExecutor",
        InlineExecutor,
    )
    monkeypatch.setattr(
        nuplan_packing,
        "_pack_nuplan_partition",
        fake_partition,
    )
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "8")

    merged = pack_nuplan_local_dataset(
        data_root=data_root,
        map_root=map_root,
        sensor_root=sensor_root,
        db_files=db_files,
        output_directory=tmp_path / "output",
        source_revision="nuplan-v1.1-mini",
        map_version="nuplan-maps-v1.0",
        pack_workers=2,
    )

    configs = cast(
        list[_NuPlanPackWorkerConfig],
        executor_state["configs"],
    )
    assert executor_state["context_method"] == "spawn"
    assert executor_state["mp_context"] is context
    assert executor_state["max_workers"] == 2
    assert executor_state["initializer"] is _initialize_nuplan_pack_worker
    assert len(configs) == 2
    assert merged["packing_workers"] == 2
    assert merged["total_samples"] == 2
    assert all(
        os.environ[variable] == "1"
        for variable in (
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    )


def test_nuplan_local_parallel_path_validates_limits_before_db_work(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="packing limits"):
        pack_nuplan_local_dataset(
            data_root=tmp_path / "missing-data",
            map_root=tmp_path / "missing-maps",
            sensor_root=tmp_path / "missing-sensors",
            db_files=[tmp_path / "missing.db"],
            output_directory=tmp_path / "output",
            source_revision="nuplan-v1.1-mini",
            map_version="nuplan-maps-v1.0",
            max_rejection_fraction=1.1,
            pack_workers=2,
        )


def test_nuplan_local_parallel_path_rejects_duplicate_log_names(
    tmp_path: Path,
):
    data_root = tmp_path / "data"
    map_root = tmp_path / "maps"
    sensor_root = tmp_path / "sensors"
    first_root = data_root / "first"
    second_root = data_root / "second"
    first_root.mkdir(parents=True)
    second_root.mkdir()
    map_root.mkdir()
    sensor_root.mkdir()
    first = first_root / "same.db"
    second = second_root / "same.db"
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="duplicate log names"):
        pack_nuplan_local_dataset(
            data_root=data_root,
            map_root=map_root,
            sensor_root=sensor_root,
            db_files=[first, second],
            output_directory=tmp_path / "output",
            source_revision="nuplan-v1.1-mini",
            map_version="nuplan-maps-v1.0",
            pack_workers=2,
        )


def test_nuplan_pack_worker_initializer_pins_native_threads(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(setNumThreads=calls.append),
    )
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, "8")

    _initialize_nuplan_pack_worker()

    assert calls == [1]
    assert all(
        os.environ[variable] == "1"
        for variable in (
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    )
