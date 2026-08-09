"""Pure contracts for one-time authorized nuPlan dataset acquisition."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO
from urllib.parse import urlsplit


SOURCE_SCHEMA_VERSION = "nuplan_authorized_source_v1"
ARCHIVE_RECEIPT_SCHEMA_VERSION = "nuplan_archive_receipt_v1"
SNAPSHOT_SCHEMA_VERSION = "nuplan_raw_snapshot_v1"
REQUIRED_COMPONENTS = frozenset({"maps", "database", "sensor_blobs"})
MIN_MULTIPART_PART_SIZE = 5 * 1024 * 1024
DEFAULT_MULTIPART_PART_SIZE = 128 * 1024 * 1024
MAX_MULTIPART_PARTS = 10_000

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_ETAG_RE = re.compile(r"^[0-9a-f]{32}(?:-[1-9][0-9]{0,4})?$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{field} has an invalid value: {value!r}")
    return value


def _validate_extract_to(value: Any, component: str) -> str:
    path = _require_text(value, "archives[].extract_to")
    parts = path.split("/")
    if (
        path.startswith("/")
        or path.endswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(
            f"archives[].extract_to must be a normalized relative path: {path!r}"
        )
    required_prefix = {
        "maps": "maps",
        "database": "nuplan-v1.1/splits",
        "sensor_blobs": "nuplan-v1.1/sensor_blobs",
    }[component]
    if path != required_prefix and not path.startswith(f"{required_prefix}/"):
        raise ValueError(
            f"{component} archive extract_to must be under {required_prefix!r}"
        )
    return path


def _validate_source_uri(value: Any) -> str:
    uri = _require_text(value, "archives[].source_uri")
    parsed = urlsplit(uri)
    if parsed.scheme == "s3":
        if (
            not parsed.netloc
            or not parsed.path.lstrip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("s3 source_uri must contain only bucket and object key")
        return uri
    if parsed.scheme != "https":
        raise ValueError("source_uri must use https:// or s3://")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(
            "https source_uri must not contain credentials or a fragment"
        )
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    ):
        raise ValueError("https source_uri must not target a private address")
    return uri


def validate_resolved_https_host(hostname: str) -> None:
    """Reject DNS rebinding to private networks before opening an HTTPS source."""
    for result in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(result[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ):
            raise ValueError(
                f"HTTPS source hostname resolves to a private address: {hostname}"
            )


def validate_public_https_uri(uri: str) -> str:
    """Validate an HTTPS source or redirect before opening a connection."""
    validated = _validate_source_uri(uri)
    parsed = urlsplit(validated)
    if parsed.scheme != "https":
        raise ValueError("redirect target must use https://")
    validate_resolved_https_host(parsed.hostname or "")
    return validated


def normalize_s3_etag(value: Any, field: str = "ETag") -> str:
    """Return a lowercase unquoted S3 ETag suitable for If-Match checks."""
    etag = _require_text(value, field).strip('"').lower()
    if _ETAG_RE.fullmatch(etag) is None:
        raise ValueError(f"{field} has an invalid S3 ETag: {value!r}")
    return etag


def validate_s3_source_head(
    head: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> str:
    """Validate one declared S3 source and return its quoted If-Match value."""
    content_length = head.get("ContentLength")
    if (
        isinstance(content_length, bool)
        or not isinstance(content_length, int)
        or content_length != int(archive["expected_size_bytes"])
    ):
        raise ValueError(
            "source S3 object size differs for "
            f"{archive['archive_id']!r}: expected "
            f"{archive['expected_size_bytes']}, got {content_length}"
        )
    actual_etag = normalize_s3_etag(
        head.get("ETag"),
        f"{archive['archive_id']}.source_etag",
    )
    expected_etag = archive.get("expected_etag", "")
    if expected_etag and actual_etag != expected_etag:
        raise ValueError(
            "source S3 object ETag differs for "
            f"{archive['archive_id']!r}: expected {expected_etag}, "
            f"got {actual_etag}"
        )
    return f'"{actual_etag}"'


def validate_source_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"source manifest schema_version must be {SOURCE_SCHEMA_VERSION!r}"
        )
    snapshot_id = _require_text(
        payload.get("snapshot_id"),
        "snapshot_id",
        pattern=_ID_RE,
    )
    dataset_revision = _require_text(
        payload.get("dataset_revision"),
        "dataset_revision",
        pattern=_ID_RE,
    )
    map_version = _require_text(
        payload.get("map_version"),
        "map_version",
        pattern=_ID_RE,
    )
    if payload.get("terms_of_use_accepted") is not True:
        raise ValueError("terms_of_use_accepted must be explicitly true")
    authorization_reference = _require_text(
        payload.get("authorization_reference"),
        "authorization_reference",
    )
    if len(authorization_reference) > 256:
        raise ValueError("authorization_reference must be at most 256 characters")

    raw_archives = payload.get("archives")
    if not isinstance(raw_archives, list) or not raw_archives:
        raise ValueError("archives must be a non-empty list")
    archives: list[dict[str, Any]] = []
    archive_ids: set[str] = set()
    filenames: set[tuple[str, str]] = set()
    for index, raw_archive in enumerate(raw_archives):
        if not isinstance(raw_archive, Mapping):
            raise ValueError(f"archives[{index}] must be an object")
        archive_id = _require_text(
            raw_archive.get("archive_id"),
            f"archives[{index}].archive_id",
            pattern=_ID_RE,
        )
        if archive_id in archive_ids:
            raise ValueError(f"duplicate archive_id {archive_id!r}")
        archive_ids.add(archive_id)
        component = _require_text(
            raw_archive.get("component"),
            f"archives[{index}].component",
        )
        if component not in REQUIRED_COMPONENTS:
            raise ValueError(
                f"archives[{index}].component must be one of "
                f"{sorted(REQUIRED_COMPONENTS)}"
            )
        filename = _require_text(
            raw_archive.get("filename"),
            f"archives[{index}].filename",
            pattern=_FILENAME_RE,
        )
        filename_key = (component, filename)
        if filename_key in filenames:
            raise ValueError(
                f"duplicate filename {filename!r} for component {component!r}"
            )
        filenames.add(filename_key)
        expected_size_bytes = raw_archive.get("expected_size_bytes")
        if (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes <= 0
        ):
            raise ValueError(
                f"archives[{index}].expected_size_bytes must be positive"
            )
        expected_sha256 = raw_archive.get("expected_sha256", "")
        expected_md5 = raw_archive.get("expected_md5", "")
        expected_etag = raw_archive.get("expected_etag", "")
        if expected_sha256:
            _require_text(
                expected_sha256,
                f"archives[{index}].expected_sha256",
                pattern=_SHA256_RE,
            )
        if expected_md5:
            _require_text(
                expected_md5,
                f"archives[{index}].expected_md5",
                pattern=_MD5_RE,
            )
        if expected_etag:
            expected_etag = normalize_s3_etag(
                expected_etag,
                f"archives[{index}].expected_etag",
            )
        if not expected_sha256 and not expected_md5 and not expected_etag:
            raise ValueError(
                f"archives[{index}] must declare expected_sha256, "
                "expected_md5, or expected_etag"
            )
        source_uri = _validate_source_uri(raw_archive.get("source_uri"))
        if expected_etag and urlsplit(source_uri).scheme != "s3":
            raise ValueError(
                f"archives[{index}].expected_etag is only valid for s3:// sources"
            )
        archives.append({
            "archive_id": archive_id,
            "component": component,
            "expected_etag": expected_etag,
            "expected_md5": expected_md5,
            "expected_sha256": expected_sha256,
            "expected_size_bytes": expected_size_bytes,
            "extract_to": _validate_extract_to(
                raw_archive.get("extract_to"),
                component,
            ),
            "filename": filename,
            "source_uri": source_uri,
        })

    components = {archive["component"] for archive in archives}
    missing_components = REQUIRED_COMPONENTS - components
    if missing_components:
        raise ValueError(
            "source manifest is missing required components: "
            f"{sorted(missing_components)}"
        )
    return {
        "archives": archives,
        "authorization_reference": authorization_reference,
        "dataset_revision": dataset_revision,
        "map_version": map_version,
        "schema_version": SOURCE_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "terms_of_use_accepted": True,
    }


def load_source_manifest_bytes(payload: bytes) -> tuple[dict[str, Any], str]:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("nuPlan source manifest is not valid UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError("nuPlan source manifest root must be an object")
    manifest = validate_source_manifest(parsed)
    contract = {
        **manifest,
        "archives": [
            {
                key: value
                for key, value in archive.items()
                if key != "source_uri"
            }
            for archive in manifest["archives"]
        ],
    }
    return manifest, sha256_bytes(canonical_json_bytes(contract))


def snapshot_prefix(source_manifest: Mapping[str, Any]) -> str:
    return (
        "nuplan/raw-snapshots/"
        f"{source_manifest['dataset_revision']}/{source_manifest['snapshot_id']}"
    )


def archive_object_key(
    source_manifest: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> str:
    return (
        f"{snapshot_prefix(source_manifest)}/archives/"
        f"{archive['component']}/{archive['archive_id']}/{archive['filename']}"
    )


def archive_receipt_key(
    source_manifest: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> str:
    return f"{archive_object_key(source_manifest, archive)}.receipt.json"


def snapshot_manifest_key(source_manifest: Mapping[str, Any]) -> str:
    return f"{snapshot_prefix(source_manifest)}/manifest.json"


def digest_stream(
    stream: BinaryIO,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    total_size = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        sha256.update(chunk)
        md5.update(chunk)
        total_size += len(chunk)
    return {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "size_bytes": total_size,
    }


def validate_archive_digest(
    result: Mapping[str, Any],
    *,
    expected_size_bytes: int,
    expected_sha256: str = "",
    expected_md5: str = "",
    label: str,
) -> None:
    if int(result["size_bytes"]) != expected_size_bytes:
        raise ValueError(
            f"archive size mismatch for {label}: "
            f"expected {expected_size_bytes}, got {result['size_bytes']}"
        )
    if expected_sha256 and result["sha256"] != expected_sha256:
        raise ValueError(
            f"archive SHA-256 mismatch for {label}: "
            f"expected {expected_sha256}, got {result['sha256']}"
        )
    if expected_md5 and result["md5"] != expected_md5:
        raise ValueError(
            f"archive MD5 mismatch for {label}: "
            f"expected {expected_md5}, got {result['md5']}"
        )


def upload_stream_multipart(
    *,
    s3_client: Any,
    stream: BinaryIO,
    bucket: str,
    key: str,
    metadata: Mapping[str, str],
    expected_size_bytes: int,
    expected_sha256: str = "",
    expected_md5: str = "",
    part_size: int = DEFAULT_MULTIPART_PART_SIZE,
) -> dict[str, Any]:
    """Stream one source archive to S3 while verifying its declared integrity."""
    if part_size < MIN_MULTIPART_PART_SIZE:
        raise ValueError("part_size must satisfy the S3 multipart minimum")
    required_parts = (expected_size_bytes + part_size - 1) // part_size
    if required_parts > MAX_MULTIPART_PARTS:
        raise ValueError(
            "archive requires more than the S3 multipart limit of "
            f"{MAX_MULTIPART_PARTS} parts"
        )
    create_response = s3_client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ContentType="application/octet-stream",
        Metadata=dict(metadata),
    )
    upload_id = create_response["UploadId"]
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    total_size = 0
    parts: list[dict[str, Any]] = []
    try:
        part_number = 1
        while True:
            buffer = bytearray()
            while len(buffer) < part_size:
                chunk = stream.read(part_size - len(buffer))
                if not chunk:
                    break
                buffer.extend(chunk)
            if not buffer:
                break
            if part_number > MAX_MULTIPART_PARTS:
                raise ValueError(
                    "archive exceeded the S3 multipart limit of "
                    f"{MAX_MULTIPART_PARTS} parts"
                )
            payload = bytes(buffer)
            sha256.update(payload)
            md5.update(payload)
            total_size += len(payload)
            response = s3_client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=payload,
            )
            parts.append({
                "ETag": response["ETag"],
                "PartNumber": part_number,
            })
            part_number += 1

        actual_sha256 = sha256.hexdigest()
        actual_md5 = md5.hexdigest()
        result = {
            "md5": actual_md5,
            "sha256": actual_sha256,
            "size_bytes": total_size,
        }
        validate_archive_digest(
            result,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            expected_md5=expected_md5,
            label=key,
        )
        if not parts:
            raise ValueError(f"archive source is empty for {key}")
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except BaseException:
        s3_client.abort_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
        )
        raise
    return result


def build_snapshot_manifest(
    *,
    source_manifest: Mapping[str, Any],
    source_contract_sha256: str,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _SHA256_RE.fullmatch(source_contract_sha256):
        raise ValueError("source_contract_sha256 must be lowercase SHA-256")
    by_archive_id: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("schema_version") != ARCHIVE_RECEIPT_SCHEMA_VERSION:
            raise ValueError("archive receipt has an unsupported schema")
        archive_id = _require_text(
            receipt.get("archive_id"),
            "receipt.archive_id",
            pattern=_ID_RE,
        )
        if archive_id in by_archive_id:
            raise ValueError(f"duplicate receipt for archive {archive_id!r}")
        if receipt.get("source_contract_sha256") != source_contract_sha256:
            raise ValueError(
                f"archive receipt {archive_id!r} has the wrong source contract"
            )
        if "source_uri" in receipt:
            raise ValueError("archive receipt must not disclose source_uri")
        by_archive_id[archive_id] = receipt

    expected_ids = {
        archive["archive_id"] for archive in source_manifest["archives"]
    }
    if set(by_archive_id) != expected_ids:
        raise ValueError(
            "archive receipts do not exactly match the source manifest"
        )
    archives = []
    for archive in source_manifest["archives"]:
        receipt = by_archive_id[archive["archive_id"]]
        if int(receipt["size_bytes"]) != int(archive["expected_size_bytes"]):
            raise ValueError(
                f"receipt size mismatch for {archive['archive_id']!r}"
            )
        if (
            archive["expected_sha256"]
            and receipt["sha256"] != archive["expected_sha256"]
        ):
            raise ValueError(
                f"receipt SHA-256 mismatch for {archive['archive_id']!r}"
            )
        if archive["expected_md5"] and receipt["md5"] != archive["expected_md5"]:
            raise ValueError(f"receipt MD5 mismatch for {archive['archive_id']!r}")
        archives.append({
            "archive_id": archive["archive_id"],
            "component": archive["component"],
            "extract_to": archive["extract_to"],
            "filename": archive["filename"],
            "md5": receipt["md5"],
            "object_uri": receipt["object_uri"],
            "sha256": receipt["sha256"],
            "size_bytes": int(receipt["size_bytes"]),
            **(
                {"source_etag": archive["expected_etag"]}
                if archive["expected_etag"]
                else {}
            ),
        })

    archives.sort(key=lambda item: item["archive_id"])
    component_counts = {
        component: sum(
            archive["component"] == component for archive in archives
        )
        for component in sorted(REQUIRED_COMPONENTS)
    }
    return {
        "archives": archives,
        "authorization_reference": source_manifest["authorization_reference"],
        "component_counts": component_counts,
        "dataset": "nuplan/nuplan-v1.1",
        "dataset_revision": source_manifest["dataset_revision"],
        "map_version": source_manifest["map_version"],
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": source_manifest["snapshot_id"],
        "source_contract_sha256": source_contract_sha256,
        "terms_of_use_accepted": True,
        "total_size_bytes": sum(
            int(archive["size_bytes"]) for archive in archives
        ),
    }
