"""Contracts for one-time authorized nuPlan source acquisition."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from Platform.pipelines.nuplan_acquisition import (
    ARCHIVE_RECEIPT_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    archive_object_key,
    build_snapshot_manifest,
    canonical_json_bytes,
    copy_s3_object_multipart,
    load_source_manifest_bytes,
    official_nuplan_open_data_region,
    snapshot_manifest_key,
    upload_https_stream_multipart,
    validate_s3_source_head,
    validate_public_https_uri,
    validate_source_manifest,
)


def _source_manifest():
    return {
        "archives": [
            {
                "archive_id": "maps-v1",
                "component": "maps",
                "expected_sha256": "a" * 64,
                "expected_size_bytes": 10,
                "extract_to": "maps",
                "filename": "nuplan-maps-v1.0.zip",
                "source_uri": "https://downloads.example.com/maps.zip?token=secret",
            },
            {
                "archive_id": "mini-db",
                "component": "database",
                "expected_etag": "b" * 32 + "-2",
                "expected_size_bytes": 20,
                "extract_to": "nuplan-v1.1/splits/mini",
                "filename": "nuplan-v1.1-mini.zip",
                "source_uri": "s3://authorized-source/nuplan/mini-db.zip",
            },
            {
                "archive_id": "mini-sensors",
                "component": "sensor_blobs",
                "expected_sha256": "c" * 64,
                "expected_size_bytes": 30,
                "extract_to": "nuplan-v1.1/sensor_blobs",
                "filename": "nuplan-v1.1-mini-sensor-blobs.zip",
                "source_uri": "https://downloads.example.com/sensors.zip?token=secret",
            },
        ],
        "authorization_reference": "operator-accepted-nuplan-terms",
        "dataset_revision": "nuplan-v1.1",
        "map_version": "nuplan-maps-v1.0",
        "schema_version": SOURCE_SCHEMA_VERSION,
        "snapshot_id": "nuplan-v1.1-mini-20260809",
        "terms_of_use_accepted": True,
    }


def _receipts(manifest, source_contract_sha):
    sizes = {
        archive["archive_id"]: archive["expected_size_bytes"]
        for archive in manifest["archives"]
    }
    source_etags = {
        archive["archive_id"]: archive["expected_etag"]
        for archive in manifest["archives"]
    }
    return [
        {
            "archive_id": "maps-v1",
            "component": "maps",
            "md5": "1" * 32,
            "object_uri": "s3://datasets/maps.zip",
            "schema_version": ARCHIVE_RECEIPT_SCHEMA_VERSION,
            "sha256": "a" * 64,
            "size_bytes": sizes["maps-v1"],
            "source_contract_sha256": source_contract_sha,
        },
        {
            "archive_id": "mini-db",
            "checksum_crc64nvme": "crc64-mini-db",
            "component": "database",
            "destination_etag": "d" * 32 + "-2",
            "md5": "",
            "object_uri": "s3://datasets/mini-db.zip",
            "schema_version": ARCHIVE_RECEIPT_SCHEMA_VERSION,
            "sha256": "",
            "size_bytes": sizes["mini-db"],
            "source_contract_sha256": source_contract_sha,
            "source_etag": source_etags["mini-db"],
            "transfer_mode": "s3_server_side_multipart_copy",
        },
        {
            "archive_id": "mini-sensors",
            "component": "sensor_blobs",
            "md5": "3" * 32,
            "object_uri": "s3://datasets/mini-sensors.zip",
            "schema_version": ARCHIVE_RECEIPT_SCHEMA_VERSION,
            "sha256": "c" * 64,
            "size_bytes": sizes["mini-sensors"],
            "source_contract_sha256": source_contract_sha,
        },
    ]


def test_source_manifest_requires_terms_and_all_components():
    manifest = _source_manifest()
    normalized = validate_source_manifest(manifest)

    assert len(normalized["archives"]) == 3
    assert {
        archive["component"] for archive in normalized["archives"]
    } == {"maps", "database", "sensor_blobs"}

    manifest["terms_of_use_accepted"] = False
    with pytest.raises(ValueError, match="explicitly true"):
        validate_source_manifest(manifest)

    manifest = _source_manifest()
    manifest["archives"].pop()
    with pytest.raises(ValueError, match="missing required components"):
        validate_source_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_uri", "http://downloads.example.com/maps.zip", "https"),
        ("source_uri", "https://127.0.0.1/maps.zip", "private address"),
        ("extract_to", "../maps", "normalized relative"),
        ("filename", "../../maps.zip", "invalid value"),
    ],
)
def test_source_manifest_rejects_unsafe_archive_fields(field, value, message):
    manifest = _source_manifest()
    manifest["archives"][0][field] = value

    with pytest.raises(ValueError, match=message):
        validate_source_manifest(manifest)


def test_source_manifest_accepts_multipart_etag_for_s3_source():
    manifest = _source_manifest()
    archive = manifest["archives"][1]
    archive["expected_etag"] = '"08ABC074DB9227E758CC41C6B1EE223C-1020"'

    normalized = validate_source_manifest(manifest)

    assert (
        normalized["archives"][1]["expected_etag"]
        == "08abc074db9227e758cc41c6b1ee223c-1020"
    )


def test_source_manifest_rejects_s3_source_without_etag():
    manifest = _source_manifest()
    archive = manifest["archives"][1]
    archive.pop("expected_etag")
    archive["expected_md5"] = "b" * 32

    with pytest.raises(ValueError, match="s3 source must declare expected_etag"):
        validate_source_manifest(manifest)


def test_source_manifest_rejects_stream_hashes_for_s3_source():
    manifest = _source_manifest()
    manifest["archives"][1]["expected_md5"] = "b" * 32

    with pytest.raises(ValueError, match="must use expected_etag"):
        validate_source_manifest(manifest)


def test_source_manifest_rejects_etag_for_https_source():
    manifest = _source_manifest()
    archive = manifest["archives"][0]
    archive.pop("expected_sha256")
    archive["expected_etag"] = "0efaac6b7c603ae6f341021b35059bcc-116"

    with pytest.raises(ValueError, match="only valid for s3"):
        validate_source_manifest(manifest)


def test_s3_source_head_validates_size_and_exact_etag():
    archive = {
        "archive_id": "mini-db",
        "expected_etag": "08abc074db9227e758cc41c6b1ee223c-1020",
        "expected_size_bytes": 8_550_100_030,
    }

    if_match = validate_s3_source_head(
        {
            "ContentLength": 8_550_100_030,
            "ETag": '"08abc074db9227e758cc41c6b1ee223c-1020"',
        },
        archive,
    )

    assert if_match == '"08abc074db9227e758cc41c6b1ee223c-1020"'

    with pytest.raises(ValueError, match="ETag differs"):
        validate_s3_source_head(
            {
                "ContentLength": 8_550_100_030,
                "ETag": '"f8abc074db9227e758cc41c6b1ee223c-1020"',
            },
            archive,
        )


def test_only_official_nuplan_prefix_uses_anonymous_open_data_access():
    assert (
        official_nuplan_open_data_region(
            "motional-nuplan",
            "public/nuplan-v1.1/nuplan-v1.1_mini.zip",
        )
        == "ap-northeast-1"
    )
    assert (
        official_nuplan_open_data_region(
            "motional-nuplan",
            "private/nuplan-v1.1/nuplan-v1.1_mini.zip",
        )
        is None
    )
    assert (
        official_nuplan_open_data_region(
            "operator-owned-source",
            "public/nuplan-v1.1/nuplan-v1.1_mini.zip",
        )
        is None
    )


def test_snapshot_manifest_redacts_authorized_source_urls():
    raw = canonical_json_bytes(_source_manifest())
    manifest, source_contract_sha = load_source_manifest_bytes(raw)
    snapshot = build_snapshot_manifest(
        source_manifest=manifest,
        source_contract_sha256=source_contract_sha,
        receipts=_receipts(manifest, source_contract_sha),
    )
    encoded = canonical_json_bytes(snapshot)

    assert b"downloads.example.com" not in encoded
    assert b"token=secret" not in encoded
    assert snapshot["component_counts"] == {
        "database": 1,
        "maps": 1,
        "sensor_blobs": 1,
    }
    by_id = {item["archive_id"]: item for item in snapshot["archives"]}
    assert "source_etag" not in by_id["maps-v1"]
    assert "source_etag" not in by_id["mini-sensors"]
    assert by_id["mini-db"]["source_etag"] == "b" * 32 + "-2"
    assert by_id["mini-db"]["transfer_mode"] == (
        "s3_server_side_multipart_copy"
    )
    assert by_id["mini-db"]["checksum_crc64nvme"] == "crc64-mini-db"
    assert snapshot["total_size_bytes"] == 60
    assert archive_object_key(
        manifest,
        manifest["archives"][0],
    ).startswith(
        "nuplan/raw-snapshots/nuplan-v1.1/"
        "nuplan-v1.1-mini-20260809/archives/maps/"
    )
    assert snapshot_manifest_key(manifest).endswith("/manifest.json")


def test_snapshot_manifest_publishes_pinned_s3_source_etag():
    source = _source_manifest()
    source["archives"][1]["expected_etag"] = (
        "08abc074db9227e758cc41c6b1ee223c-1020"
    )
    manifest, source_contract_sha = load_source_manifest_bytes(
        canonical_json_bytes(source)
    )

    snapshot = build_snapshot_manifest(
        source_manifest=manifest,
        source_contract_sha256=source_contract_sha,
        receipts=_receipts(manifest, source_contract_sha),
    )

    by_id = {item["archive_id"]: item for item in snapshot["archives"]}
    assert by_id["mini-db"]["source_etag"] == (
        "08abc074db9227e758cc41c6b1ee223c-1020"
    )


def test_signed_url_refresh_does_not_change_source_contract_identity():
    first = _source_manifest()
    second = _source_manifest()
    second["archives"][0]["source_uri"] = (
        "https://downloads.example.com/maps.zip?token=refreshed"
    )

    _, first_digest = load_source_manifest_bytes(canonical_json_bytes(first))
    _, second_digest = load_source_manifest_bytes(canonical_json_bytes(second))

    assert first_digest == second_digest


def test_public_https_validation_rejects_private_dns_resolution(monkeypatch):
    monkeypatch.setattr(
        "Platform.pipelines.nuplan_acquisition.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("10.0.0.7", 443)),
        ],
    )

    with pytest.raises(ValueError, match="resolves to a private address"):
        validate_public_https_uri("https://downloads.example.com/archive.zip")


class _MultipartS3:
    def __init__(self):
        self.parts = []
        self.completed = False
        self.aborted = False

    def create_multipart_upload(self, **kwargs):
        self.create_request = kwargs
        return {"UploadId": "upload-1"}

    def upload_part(self, **kwargs):
        self.parts.append(kwargs["Body"])
        return {"ETag": f"etag-{kwargs['PartNumber']}"}

    def complete_multipart_upload(self, **kwargs):
        self.complete_request = kwargs
        self.completed = True

    def abort_multipart_upload(self, **kwargs):
        self.abort_request = kwargs
        self.aborted = True


def test_streaming_multipart_upload_verifies_content_hashes():
    payload = b"a" * (5 * 1024 * 1024) + b"tail"
    sha256 = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    s3 = _MultipartS3()

    result = upload_https_stream_multipart(
        s3_client=s3,
        stream=io.BytesIO(payload),
        bucket="datasets",
        key="nuplan/archive.zip",
        metadata={"snapshot-id": "mini"},
        expected_size_bytes=len(payload),
        expected_sha256=sha256,
        expected_md5=md5,
        part_size=5 * 1024 * 1024,
    )

    assert result == {
        "md5": md5,
        "sha256": sha256,
        "size_bytes": len(payload),
    }
    assert len(s3.parts) == 2
    assert s3.completed is True
    assert s3.aborted is False


def test_streaming_multipart_upload_aborts_before_completion_on_mismatch():
    payload = b"archive"
    s3 = _MultipartS3()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        upload_https_stream_multipart(
            s3_client=s3,
            stream=io.BytesIO(payload),
            bucket="datasets",
            key="nuplan/archive.zip",
            metadata={"snapshot-id": "mini"},
            expected_size_bytes=len(payload),
            expected_sha256="f" * 64,
            part_size=5 * 1024 * 1024,
        )

    assert s3.completed is False
    assert s3.aborted is True


def test_streaming_multipart_rejects_more_than_s3_part_limit_before_upload():
    s3 = _MultipartS3()

    with pytest.raises(ValueError, match="multipart limit"):
        upload_https_stream_multipart(
            s3_client=s3,
            stream=io.BytesIO(b""),
            bucket="datasets",
            key="nuplan/archive.zip",
            metadata={"snapshot-id": "mini"},
            expected_size_bytes=10_001 * 5 * 1024 * 1024,
            expected_sha256="f" * 64,
            part_size=5 * 1024 * 1024,
        )

    assert not hasattr(s3, "create_request")


class _MultipartCopyS3:
    def __init__(self, *, fail_part: int | None = None):
        self.copy_requests: list[dict[str, object]] = []
        self.completed = False
        self.aborted = False
        self.fail_part = fail_part

    def create_multipart_upload(self, **kwargs):
        self.create_request = kwargs
        return {"UploadId": "copy-upload-1"}

    def upload_part_copy(self, **kwargs):
        self.copy_requests.append(kwargs)
        if kwargs["PartNumber"] == self.fail_part:
            raise RuntimeError("copy failed")
        return {
            "CopyPartResult": {
                "ETag": f'"copy-etag-{kwargs["PartNumber"]}"',
            },
        }

    def complete_multipart_upload(self, **kwargs):
        self.complete_request = kwargs
        self.completed = True
        return {"ChecksumCRC64NVME": "crc64-result"}

    def abort_multipart_upload(self, **kwargs):
        self.abort_request = kwargs
        self.aborted = True

    def head_object(self, **kwargs):
        self.head_request = kwargs
        return {
            "ChecksumCRC64NVME": "crc64-result",
            "ContentLength": 5 * 1024 * 1024 + 17,
            "ETag": '"destination-etag-2"',
        }


def test_s3_multipart_copy_uses_ranges_etag_and_full_object_checksum():
    s3 = _MultipartCopyS3()

    result = copy_s3_object_multipart(
        s3_client=s3,
        source_bucket="motional-nuplan",
        source_key="public/nuplan-v1.1/archive.zip",
        source_etag="a" * 32 + "-2",
        destination_bucket="datasets",
        destination_key="nuplan/archive.zip",
        metadata={"snapshot-id": "full"},
        expected_size_bytes=5 * 1024 * 1024 + 17,
        part_size=5 * 1024 * 1024,
    )

    assert result == {
        "checksum_crc64nvme": "crc64-result",
        "destination_etag": "destination-etag-2",
        "md5": "",
        "sha256": "",
        "size_bytes": 5 * 1024 * 1024 + 17,
        "source_etag": "a" * 32 + "-2",
        "transfer_mode": "s3_server_side_multipart_copy",
    }
    assert [request["CopySourceRange"] for request in s3.copy_requests] == [
        f"bytes=0-{5 * 1024 * 1024 - 1}",
        f"bytes={5 * 1024 * 1024}-{5 * 1024 * 1024 + 16}",
    ]
    assert {
        request["CopySourceIfMatch"] for request in s3.copy_requests
    } == {'"' + "a" * 32 + '-2"'}
    assert s3.create_request["ChecksumAlgorithm"] == "CRC64NVME"
    assert s3.complete_request["ChecksumType"] == "FULL_OBJECT"
    assert s3.head_request["ChecksumMode"] == "ENABLED"
    assert s3.completed is True
    assert s3.aborted is False


def test_s3_multipart_copy_aborts_on_part_failure():
    s3 = _MultipartCopyS3(fail_part=2)

    with pytest.raises(RuntimeError, match="copy failed"):
        copy_s3_object_multipart(
            s3_client=s3,
            source_bucket="motional-nuplan",
            source_key="public/nuplan-v1.1/archive.zip",
            source_etag="a" * 32 + "-2",
            destination_bucket="datasets",
            destination_key="nuplan/archive.zip",
            metadata={"snapshot-id": "full"},
            expected_size_bytes=5 * 1024 * 1024 + 17,
            part_size=5 * 1024 * 1024,
        )

    assert s3.completed is False
    assert s3.aborted is True


def test_s3_multipart_copy_rejects_part_limit_before_upload():
    s3 = _MultipartCopyS3()

    with pytest.raises(ValueError, match="multipart limit"):
        copy_s3_object_multipart(
            s3_client=s3,
            source_bucket="motional-nuplan",
            source_key="public/nuplan-v1.1/archive.zip",
            source_etag="a" * 32 + "-2",
            destination_bucket="datasets",
            destination_key="nuplan/archive.zip",
            metadata={"snapshot-id": "full"},
            expected_size_bytes=10_001 * 5 * 1024 * 1024,
            part_size=5 * 1024 * 1024,
        )

    assert not hasattr(s3, "create_request")


def test_flyte_role_can_read_only_the_official_nuplan_v11_prefix():
    terraform = (
        Path(__file__).parents[2]
        / "Platform"
        / "infra"
        / "modules"
        / "flyte"
        / "main.tf"
    ).read_text(encoding="utf-8")

    assert '"s3:GetObject"' in terraform
    assert (
        '"arn:aws:s3:::motional-nuplan/public/nuplan-v1.1/*"'
        in terraform
    )
    assert '"arn:aws:s3:::motional-nuplan/*"' not in terraform
