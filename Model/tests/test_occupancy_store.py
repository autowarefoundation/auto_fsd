import hashlib

import pytest

from Platform.pipelines.occupancy_store import (
    OCCUPANCY_SET_SCHEMA,
    encode_occupancy_set_manifest,
    occupancy_model_artifact_id,
    occupancy_set_manifest,
    occupancy_set_s3_key,
)


WEIGHT_SHA256 = "a" * 64
DATASET_MANIFEST_SHA256 = "b" * 64
MODEL_SOURCE = {
    "code_license_spdx": "Apache-2.0",
    "config": "bevformerv2-r50-t8-24ep.py",
    "license_spdx": "NOASSERTION",
    "repository": "https://github.com/fundamentalvision/BEVFormer",
    "repository_revision": (
        "66b65f3a1f58caf0507cb2a971b9c0e7f842376c"
    ),
    "training_data_license_spdx": "CC-BY-NC-SA-4.0",
    "weight_sha256": WEIGHT_SHA256,
    "weight_source_url": (
        "https://github.com/fundamentalvision/BEVFormer"
        "#model-zoo"
    ),
}
PRODUCER_CONFIG = {
    "box_rasterization": "oriented-footprint-v1",
    "score_threshold": 0.2,
}


def _model_artifact_id(
    producer_config: dict = PRODUCER_CONFIG,
) -> str:
    return occupancy_model_artifact_id(
        artifact_kind="detection-derived-occupancy",
        artifact_schema="v1",
        geometry_id="autoe2e-bev-450x300-0p4m-v1",
        head_version="bevformer-v2-r50-t8-box-raster-v1",
        input_contract="kitscenes-packed-six-camera-v1",
        model_source=MODEL_SOURCE,
        producer_config=producer_config,
        taxonomy_version="autoe2e-bev-semantic-v1",
    )


MODEL_ARTIFACT_ID = _model_artifact_id()


def _shard(name: str, digest: str) -> dict:
    return {
        "byte_size": 1024,
        "s3_key": (
            "semantic-occupancy/schema=v1/"
            f"model={MODEL_ARTIFACT_ID}/"
            f"manifest={DATASET_MANIFEST_SHA256}/"
            "geometry=autoe2e-bev-450x300-0p4m-v1/"
            "taxonomy=autoe2e-bev-semantic-v1/"
            "head=bevformer-v2-r50-t8-box-raster-v1/"
            f"dataset=kitscenes/shard={name}/occupancy.bin.gz"
        ),
        "sample_count": 32,
        "sha256": digest,
        "shard": name,
        "teacher_present": False,
    }


def _manifest(**overrides):
    values = {
        "artifact_kind": "detection-derived-occupancy",
        "artifact_schema": "v1",
        "created_at": "2026-08-19T00:00:00Z",
        "dataset": "kitscenes",
        "dataset_version": "v3.3",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "display_name": "BEVFormer V2 R50 t8",
        "geometry_id": "autoe2e-bev-450x300-0p4m-v1",
        "head_version": "bevformer-v2-r50-t8-box-raster-v1",
        "input_contract": "kitscenes-packed-six-camera-v1",
        "limitations": [
            "Object occupancy is derived from predicted box footprints.",
        ],
        "model_artifact_id": MODEL_ARTIFACT_ID,
        "model_family": "BEVFormer V2",
        "model_source": MODEL_SOURCE,
        "producer_config": PRODUCER_CONFIG,
        "shards": [
            _shard("train-000001.tar", "d" * 64),
            _shard("train-000000.tar", "c" * 64),
        ],
        "supported_classes": [
            "vehicle",
            "vulnerable_road_user",
            "other_obstacle",
        ],
        "taxonomy_version": "autoe2e-bev-semantic-v1",
        "teacher_available": False,
    }
    values.update(overrides)
    return occupancy_set_manifest(**values)


def test_occupancy_set_key_is_dataset_first_and_model_immutable():
    assert occupancy_set_s3_key(
        "kitscenes",
        "v3.3",
        MODEL_ARTIFACT_ID,
        DATASET_MANIFEST_SHA256,
    ) == (
        "semantic-occupancy-sets/schema=v2/"
        "dataset=kitscenes/version=v3.3/"
        f"model={MODEL_ARTIFACT_ID}/"
        f"manifest={DATASET_MANIFEST_SHA256}/manifest.json"
    )


def test_manifest_sorts_shards_and_accounts_for_the_complete_set():
    manifest = _manifest()

    assert manifest["schema_version"] == OCCUPANCY_SET_SCHEMA
    assert manifest["shard_count"] == 2
    assert manifest["sample_count"] == 64
    assert [entry["shard"] for entry in manifest["shards"]] == [
        "train-000000.tar",
        "train-000001.tar",
    ]
    assert manifest["teacher_available"] is False
    assert manifest["model_source"]["weight_sha256"] == WEIGHT_SHA256
    assert manifest["producer_config"] == PRODUCER_CONFIG


def test_model_artifact_id_covers_producer_config():
    changed_config = {
        **PRODUCER_CONFIG,
        "score_threshold": 0.3,
    }

    assert _model_artifact_id(changed_config) != MODEL_ARTIFACT_ID
    with pytest.raises(ValueError, match="complete producer recipe"):
        _manifest(producer_config=changed_config)


def test_manifest_encoding_is_deterministic_and_content_addressed():
    payload, digest = encode_occupancy_set_manifest(_manifest())

    assert payload.endswith(b"\n")
    assert payload.isascii()
    assert digest == hashlib.sha256(payload).hexdigest()
    assert encode_occupancy_set_manifest(_manifest()) == (payload, digest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_kind", "semantic-looking-boxes"),
        ("dataset", "../kitscenes"),
        ("dataset_version", "latest"),
        ("dataset_manifest_sha256", "B" * 64),
        ("model_artifact_id", "short"),
        ("supported_classes", ["vehicle", "vehicle"]),
        ("limitations", []),
    ],
)
def test_manifest_rejects_noncanonical_identity_fields(field, value):
    with pytest.raises(ValueError):
        _manifest(**{field: value})


def test_manifest_rejects_noncanonical_or_duplicate_shard_pointers():
    escaping = _shard("train-000000.tar", "c" * 64)
    escaping["s3_key"] = "semantic-occupancy/occupancy.bin.gz"
    with pytest.raises(ValueError, match="not canonical"):
        _manifest(shards=[escaping])

    with pytest.raises(ValueError, match="duplicate"):
        _manifest(
            shards=[
                _shard("train-000000.tar", "c" * 64),
                _shard("train-000000.tar", "d" * 64),
            ]
        )


def test_manifest_rejects_mixed_teacher_availability():
    teacher_shard = _shard("train-000000.tar", "c" * 64)
    teacher_shard["teacher_present"] = True

    with pytest.raises(ValueError, match="teacher availability"):
        _manifest(shards=[teacher_shard])
