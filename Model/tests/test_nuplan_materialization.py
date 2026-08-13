"""nuPlan immutable snapshot materialization contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from data_parsing.nuplan.materialization import (
    discover_materialized_nuplan,
    extract_nuplan_archive,
    load_nuplan_snapshot_manifest,
    materialize_nuplan_snapshot,
    select_snapshot_archives,
    verify_archive_file,
)
from data_parsing.nuplan.packing import NUPLAN_CAMERA_CHANNELS


def _write_zip(path: Path, members: dict[str, bytes]) -> dict[str, str | int]:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    payload = path.read_bytes()
    return {
        "md5": hashlib.md5(
            payload,
            usedforsecurity=False,
        ).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _archive(
    archive_id: str,
    component: str,
    extract_to: str,
    path: Path,
    digest: dict[str, str | int],
) -> dict[str, object]:
    return {
        "archive_id": archive_id,
        "component": component,
        "extract_to": extract_to,
        "filename": path.name,
        "md5": digest["md5"],
        "object_uri": f"s3://datasets/{path.name}",
        "sha256": digest["sha256"],
        "size_bytes": digest["size_bytes"],
    }


def _snapshot(tmp_path: Path):
    map_path = tmp_path / "maps.zip"
    db_path = tmp_path / "db.zip"
    camera_path = tmp_path / "camera.zip"
    lidar_path = tmp_path / "lidar.zip"
    map_digest = _write_zip(
        map_path,
        {
            "nuplan-maps-v1.0/us-ma-boston/1/map.gpkg": b"map",
            "nuplan-maps-v1.0/nuplan-maps-v1.0.json": b"{}",
            "LICENSE": b"license",
        },
    )
    db_digest = _write_zip(
        db_path,
        {
            "data/cache/mini/log-complete.db": b"db",
            "data/cache/mini/log-missing.db": b"db",
        },
    )
    camera_digest = _write_zip(
        camera_path,
        {
            f"camera-root/log-complete/{channel}/frame.jpg": b"jpg"
            for channel in NUPLAN_CAMERA_CHANNELS
        },
    )
    lidar_digest = _write_zip(
        lidar_path,
        {
            "lidar-root/log-complete/MergedPointCloud/frame.pcd": b"pcd",
        },
    )
    archives = [
        _archive(
            "maps",
            "maps",
            "maps",
            map_path,
            map_digest,
        ),
        _archive(
            "db",
            "database",
            "nuplan-v1.1/splits/mini",
            db_path,
            db_digest,
        ),
        _archive(
            "camera",
            "sensor_blobs",
            "nuplan-v1.1/sensor_blobs",
            camera_path,
            camera_digest,
        ),
        _archive(
            "lidar",
            "sensor_blobs",
            "nuplan-v1.1/sensor_blobs",
            lidar_path,
            lidar_digest,
        ),
    ]
    manifest = {
        "archives": archives,
        "dataset_revision": "nuplan-v1.1",
        "map_version": "nuplan-maps-v1.1",
        "schema_version": "nuplan_raw_snapshot_v1",
        "snapshot_id": "nuplan-test",
        "source_contract_sha256": "a" * 64,
        "total_size_bytes": sum(
            int(str(archive["size_bytes"])) for archive in archives
        ),
    }
    paths = {
        str(archive["archive_id"]): (
            tmp_path / str(archive["filename"])
        )
        for archive in archives
    }
    return manifest, paths


def test_materializes_devkit_layout_and_filters_incomplete_sensor_logs(
    tmp_path,
):
    manifest, paths = _snapshot(tmp_path)
    decoded = load_nuplan_snapshot_manifest(
        json.dumps(manifest).encode("ascii")
    )

    materialized = materialize_nuplan_snapshot(
        decoded,
        paths,
        tmp_path / "dataset",
    )

    assert materialized.sensor_log_names == ("log-complete",)
    assert materialized.map_version == "nuplan-maps-v1.0"
    assert [path.name for path in materialized.db_files] == [
        "log-complete.db"
    ]
    assert (
        materialized.map_root
        / "us-ma-boston"
        / "1"
        / "map.gpkg"
    ).read_bytes() == b"map"
    assert (
        materialized.sensor_root
        / "log-complete"
        / "CAM_F0"
        / "frame.jpg"
    ).read_bytes() == b"jpg"
    assert (
        materialized.sensor_root
        / "log-complete"
        / "MergedPointCloud"
        / "frame.pcd"
    ).read_bytes() == b"pcd"


def test_selected_archives_must_include_all_components(tmp_path):
    manifest, _ = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="missing modalities"):
        select_snapshot_archives(manifest, ["maps", "db", "camera"])

    with pytest.raises(ValueError, match="unknown"):
        select_snapshot_archives(manifest, ["missing"])


def test_manifest_inventory_may_include_unselected_non_zip_files(tmp_path):
    manifest, _ = _snapshot(tmp_path)
    inventory = {
        "archive_id": "sensor-mini-inventory",
        "component": "sensor_blobs",
        "extract_to": "nuplan-v1.1/sensor_blobs",
        "filename": "nuplan_mini_sensor.txt",
        "md5": "",
        "object_uri": "s3://datasets/nuplan_mini_sensor.txt",
        "sha256": "",
        "size_bytes": 4,
    }
    manifest["archives"].append(inventory)
    manifest["total_size_bytes"] += inventory["size_bytes"]

    decoded = load_nuplan_snapshot_manifest(
        json.dumps(manifest).encode("ascii")
    )
    selected = select_snapshot_archives(
        decoded,
        ["maps", "db", "camera", "lidar"],
    )

    assert [archive["archive_id"] for archive in selected] == [
        "maps",
        "db",
        "camera",
        "lidar",
    ]
    with pytest.raises(ValueError, match="must be ZIP files"):
        select_snapshot_archives(
            decoded,
            ["maps", "db", "camera", "lidar", "sensor-mini-inventory"],
        )


def test_archive_digest_rejects_modified_payload(tmp_path):
    manifest, paths = _snapshot(tmp_path)
    archive = manifest["archives"][0]
    verify_archive_file(paths["maps"], archive)
    paths["maps"].write_bytes(paths["maps"].read_bytes() + b"changed")

    with pytest.raises(ValueError, match="size mismatch"):
        verify_archive_file(paths["maps"], archive)


def test_zip_member_path_and_symlink_are_rejected(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.db", b"db")
    archive_spec = {
        "archive_id": "unsafe",
        "component": "database",
        "extract_to": "nuplan-v1.1/splits/mini",
    }

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        extract_nuplan_archive(
            archive_path,
            archive_spec,
            tmp_path / "dataset",
            map_version="nuplan-maps-v1.0",
        )

    symlink_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_path, "w") as archive:
        info = zipfile.ZipInfo("data/cache/mini/link.db")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"target")
    with pytest.raises(ValueError, match="symlink"):
        extract_nuplan_archive(
            symlink_path,
            archive_spec,
            tmp_path / "dataset",
            map_version="nuplan-maps-v1.0",
        )


def test_discovery_fails_without_complete_sensor_data(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "maps").mkdir(parents=True)
    (
        dataset_root
        / "maps"
        / "nuplan-maps-v1.0.json"
    ).write_text("{}")
    (dataset_root / "nuplan-v1.1" / "sensor_blobs").mkdir(parents=True)

    with pytest.raises(ValueError, match="complete camera and LiDAR"):
        discover_materialized_nuplan(dataset_root)
