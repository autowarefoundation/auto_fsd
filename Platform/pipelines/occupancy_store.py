"""Immutable publication contract for semantic occupancy model sets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

OCCUPANCY_SET_SCHEMA = "semantic_occupancy_set_v1"
OCCUPANCY_SET_PREFIX = "semantic-occupancy-sets/schema=v1"
OCCUPANCY_ARTIFACT_KINDS = frozenset({
    "native-semantic-occupancy",
    "detection-derived-occupancy",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^v[1-9][0-9]*\.[0-9]+$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sha256(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _segment(value: str, label: str) -> str:
    if not value or not _SEGMENT_RE.fullmatch(value):
        raise ValueError(f"{label} must be one canonical path segment")
    return value


def occupancy_set_s3_key(
    dataset: str,
    dataset_version: str,
    model_artifact_id: str,
) -> str:
    """Return the dataset-first discovery key written last by a producer."""
    dataset = _segment(dataset, "dataset")
    if not _VERSION_RE.fullmatch(dataset_version):
        raise ValueError("dataset_version must match v<major>.<minor>")
    model_artifact_id = _sha256(model_artifact_id, "model_artifact_id")
    return (
        f"{OCCUPANCY_SET_PREFIX}/dataset={dataset}/"
        f"version={dataset_version}/model={model_artifact_id}/manifest.json"
    )


def _string_list(
    values: Sequence[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    output = [str(value) for value in values]
    if (not allow_empty and not output) or any(
        not value or value.strip() != value for value in output
    ):
        raise ValueError(f"{label} must contain non-empty trimmed strings")
    if len(set(output)) != len(output):
        raise ValueError(f"{label} must not contain duplicates")
    return output


def occupancy_set_manifest(
    *,
    artifact_kind: str,
    artifact_schema: str,
    created_at: str,
    dataset: str,
    dataset_version: str,
    dataset_manifest_sha256: str,
    display_name: str,
    geometry_id: str,
    head_version: str,
    input_contract: str,
    limitations: Sequence[str],
    model_artifact_id: str,
    model_family: str,
    model_source: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    supported_classes: Sequence[str],
    taxonomy_version: str,
    teacher_available: bool,
) -> dict[str, Any]:
    """Build and validate the manifest that atomically publishes one model."""
    if artifact_kind not in OCCUPANCY_ARTIFACT_KINDS:
        raise ValueError(f"unsupported artifact_kind {artifact_kind!r}")
    model_artifact_id = _sha256(model_artifact_id, "model_artifact_id")
    dataset_manifest_sha256 = _sha256(
        dataset_manifest_sha256,
        "dataset_manifest_sha256",
    )
    occupancy_set_s3_key(dataset, dataset_version, model_artifact_id)
    for label, value in (
        ("artifact_schema", artifact_schema),
        ("created_at", created_at),
        ("display_name", display_name),
        ("geometry_id", geometry_id),
        ("head_version", head_version),
        ("input_contract", input_contract),
        ("model_family", model_family),
        ("taxonomy_version", taxonomy_version),
    ):
        if not value or value.strip() != value:
            raise ValueError(f"{label} must be a non-empty trimmed string")

    source = {
        str(key): value
        for key, value in model_source.items()
    }
    required_source = {
        "config",
        "license_spdx",
        "repository",
        "repository_revision",
        "weight_sha256",
    }
    if set(source) != required_source:
        raise ValueError(
            "model_source must contain exactly "
            + ", ".join(sorted(required_source))
        )
    _sha256(str(source["weight_sha256"]), "model_source.weight_sha256")
    if source["weight_sha256"] != model_artifact_id:
        raise ValueError("model artifact ID must equal the weight digest")
    if any(
        not isinstance(source[key], str)
        or not source[key]
        or source[key].strip() != source[key]
        for key in required_source
    ):
        raise ValueError("model_source values must be non-empty strings")

    supported = _string_list(
        supported_classes,
        "supported_classes",
    )
    disclosed_limitations = _string_list(
        limitations,
        "limitations",
    )
    entries = []
    seen_shards: set[str] = set()
    total_samples = 0
    for raw_entry in shards:
        required_entry = {
            "byte_size",
            "s3_key",
            "sample_count",
            "sha256",
            "shard",
            "teacher_present",
        }
        if set(raw_entry) != required_entry:
            raise ValueError(
                "occupancy shard entry has an unexpected field set"
            )
        shard = _segment(str(raw_entry["shard"]), "shard")
        if shard in seen_shards:
            raise ValueError(f"duplicate occupancy shard {shard!r}")
        seen_shards.add(shard)
        payload_sha256 = _sha256(str(raw_entry["sha256"]), "shard sha256")
        byte_size = int(raw_entry["byte_size"])
        sample_count = int(raw_entry["sample_count"])
        teacher_present = bool(raw_entry["teacher_present"])
        if byte_size <= 0 or sample_count <= 0:
            raise ValueError("occupancy shard sizes must be positive")
        if teacher_present != teacher_available:
            raise ValueError(
                "shard teacher availability differs from the model set"
            )
        key = str(raw_entry["s3_key"])
        canonical_prefix = (
            f"semantic-occupancy/schema={artifact_schema}/"
            f"model={model_artifact_id}/"
            f"manifest={dataset_manifest_sha256}/"
            f"geometry={geometry_id}/taxonomy={taxonomy_version}/"
            f"head={head_version}/dataset={dataset}/shard={shard}/"
        )
        if (
            not key.startswith(canonical_prefix)
            or not key.endswith("/occupancy.bin.gz")
        ):
            raise ValueError("occupancy shard key is not canonical")
        entries.append({
            "byte_size": byte_size,
            "s3_key": key,
            "sample_count": sample_count,
            "sha256": payload_sha256,
            "shard": shard,
            "teacher_present": teacher_present,
        })
        total_samples += sample_count
    if not entries:
        raise ValueError("occupancy set must contain at least one shard")
    entries.sort(key=lambda entry: entry["shard"])

    return {
        "schema_version": OCCUPANCY_SET_SCHEMA,
        "artifact_kind": artifact_kind,
        "artifact_schema": artifact_schema,
        "created_at": created_at,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "display_name": display_name,
        "geometry_id": geometry_id,
        "head_version": head_version,
        "input_contract": input_contract,
        "limitations": disclosed_limitations,
        "model_artifact_id": model_artifact_id,
        "model_family": model_family,
        "model_source": source,
        "sample_count": total_samples,
        "shard_count": len(entries),
        "shards": entries,
        "supported_classes": supported,
        "taxonomy_version": taxonomy_version,
        "teacher_available": teacher_available,
    }


def encode_occupancy_set_manifest(
    manifest: Mapping[str, Any],
) -> tuple[bytes, str]:
    """Return stable ASCII JSON and its content digest."""
    payload = (
        json.dumps(
            manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    return payload, hashlib.sha256(payload).hexdigest()
