"""Flyte workflows for immutable nuPlan acquisition and BEV v2 packing."""

from __future__ import annotations

import functools
import os
from typing import List, NamedTuple, Optional

from flytekit import (
    PodTemplate,
    Resources,
    dynamic,
    map_task,
    task,
    workflow,
)
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)


DATA_PREP_IMAGE = os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    "auto-e2e/data-prep:latest",
)


class NuPlanRawSnapshotOutput(NamedTuple):
    manifest: FlyteFile
    manifest_sha256: str
    snapshot_prefix: str
    archive_count: int
    total_size_bytes: int


def _data_prep_pod_template() -> PodTemplate:
    return PodTemplate(
        annotations={"karpenter.sh/do-not-disrupt": "true"},
    )


def _nuplan_pack_worker_count(
    db_file_count: int,
    limit_total_scenarios: int,
) -> int:
    if db_file_count <= 0:
        raise ValueError("nuPlan pack requires at least one DB file")
    if limit_total_scenarios < 0:
        raise ValueError("limit_total_scenarios must be non-negative")
    return 1 if limit_total_scenarios else min(8, db_file_count)


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    requests=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    limits=Resources(
        cpu="2",
        mem="4Gi",
        ephemeral_storage="4Gi",
    ),
    retries=2,
)
def acquire_nuplan_archive(
    source_manifest: FlyteFile,
    archive_index: int,
    datasets_bucket: str,
    aws_region: str = "us-west-2",
) -> FlyteFile:
    """Import one authorized archive into an immutable S3 snapshot."""
    import json
    import tempfile
    from contextlib import closing
    from pathlib import Path
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlsplit
    from urllib.request import HTTPRedirectHandler, Request, build_opener

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError

    from Platform.pipelines.nuplan_acquisition import (
        ARCHIVE_RECEIPT_SCHEMA_VERSION,
        archive_object_key,
        archive_receipt_key,
        canonical_json_bytes,
        copy_s3_object_multipart,
        digest_stream,
        load_source_manifest_bytes,
        official_nuplan_open_data_region,
        upload_https_stream_multipart,
        validate_archive_digest,
        validate_public_https_uri,
        validate_s3_source_head,
    )

    if (
        not datasets_bucket
        or datasets_bucket.startswith("s3://")
        or "/" in datasets_bucket
    ):
        raise ValueError("datasets_bucket must be one S3 bucket name")
    if archive_index < 0:
        raise ValueError("archive_index must be non-negative")

    source_bytes = Path(source_manifest.download()).read_bytes()
    manifest, source_contract_sha256 = load_source_manifest_bytes(source_bytes)
    if archive_index >= len(manifest["archives"]):
        raise IndexError(
            f"archive_index {archive_index} is outside "
            f"{len(manifest['archives'])} source archives"
        )
    archive = manifest["archives"][archive_index]
    parsed_source = urlsplit(archive["source_uri"])
    object_key = archive_object_key(manifest, archive)
    receipt_key = archive_receipt_key(manifest, archive)
    s3 = boto3.client("s3", region_name=aws_region)

    def receipt_output(payload: bytes) -> FlyteFile:
        path = Path(tempfile.mkdtemp(prefix="nuplan-archive-receipt-"))
        output = path / "receipt.json"
        output.write_bytes(payload)
        return FlyteFile(str(output))

    try:
        response = s3.get_object(
            Bucket=datasets_bucket,
            Key=receipt_key,
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "404",
            "NoSuchKey",
        }:
            raise
    else:
        with closing(response["Body"]) as receipt_stream:
            receipt_bytes = receipt_stream.read()
        receipt = json.loads(receipt_bytes)
        if (
            receipt.get("schema_version")
            != ARCHIVE_RECEIPT_SCHEMA_VERSION
            or receipt.get("archive_id") != archive["archive_id"]
            or receipt.get("source_contract_sha256")
            != source_contract_sha256
            or receipt.get("object_uri")
            != f"s3://{datasets_bucket}/{object_key}"
        ):
            raise ValueError(
                "existing nuPlan receipt conflicts with "
                f"{archive['archive_id']!r}"
            )
        head_arguments = {
            "Bucket": datasets_bucket,
            "Key": object_key,
        }
        if receipt.get("transfer_mode") == (
            "s3_server_side_multipart_copy"
        ):
            head_arguments["ChecksumMode"] = "ENABLED"
        head = s3.head_object(**head_arguments)
        if int(head["ContentLength"]) != int(receipt["size_bytes"]):
            raise ValueError(
                "existing nuPlan object size differs from receipt: "
                f"{object_key}"
            )
        if (
            receipt.get("checksum_crc64nvme")
            and head.get("ChecksumCRC64NVME")
            != receipt["checksum_crc64nvme"]
        ):
            raise ValueError(
                "existing nuPlan object checksum differs from receipt: "
                f"{object_key}"
            )
        return receipt_output(receipt_bytes)

    upload = None
    try:
        head = s3.head_object(
            Bucket=datasets_bucket,
            Key=object_key,
            **(
                {"ChecksumMode": "ENABLED"}
                if parsed_source.scheme == "s3"
                else {}
            ),
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "404",
            "NoSuchKey",
        }:
            raise
    else:
        if int(head["ContentLength"]) != int(
            archive["expected_size_bytes"]
        ):
            raise ValueError(
                f"existing nuPlan object has the wrong size: {object_key}"
            )
        expected_metadata = {
            "archive-id": archive["archive_id"],
            "snapshot-id": manifest["snapshot_id"],
            "source-contract-sha256": source_contract_sha256,
            **(
                {"source-etag": archive["expected_etag"]}
                if archive["expected_etag"]
                else {}
            ),
        }
        actual_metadata = head.get("Metadata", {})
        if any(
            actual_metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise ValueError(
                "existing nuPlan object metadata differs from its source "
                f"contract: {object_key}"
            )
        if parsed_source.scheme == "s3":
            checksum = head.get("ChecksumCRC64NVME")
            if not isinstance(checksum, str) or not checksum:
                raise ValueError(
                    "existing server-side copied nuPlan object lacks "
                    f"CRC64NVME: {object_key}"
                )
            upload = {
                "checksum_crc64nvme": checksum,
                "destination_etag": str(head["ETag"]).strip('"').lower(),
                "md5": "",
                "sha256": "",
                "size_bytes": int(head["ContentLength"]),
                "source_etag": archive["expected_etag"],
                "transfer_mode": "s3_server_side_multipart_copy",
            }
        else:
            existing_object = s3.get_object(
                Bucket=datasets_bucket,
                Key=object_key,
            )
            with closing(existing_object["Body"]) as existing_stream:
                upload = digest_stream(existing_stream)
            validate_archive_digest(
                upload,
                expected_size_bytes=archive["expected_size_bytes"],
                expected_sha256=archive["expected_sha256"],
                expected_md5=archive["expected_md5"],
                label=object_key,
            )
            upload.update({
                "checksum_crc64nvme": "",
                "destination_etag": str(head["ETag"]).strip('"').lower(),
                "source_etag": "",
                "transfer_mode": "https_stream_hash",
            })
        print(
            "Recovered nuPlan archive receipt from existing verified object "
            f"id={archive['archive_id']}"
        )

    if upload is None:
        if parsed_source.scheme == "s3":
            source_bucket = parsed_source.netloc
            source_key = parsed_source.path.lstrip("/")
            open_data_region = official_nuplan_open_data_region(
                source_bucket,
                source_key,
            )
            if open_data_region is None:
                source_s3 = boto3.client("s3")
            else:
                source_s3 = boto3.client(
                    "s3",
                    region_name=open_data_region,
                    config=Config(signature_version=UNSIGNED),
                )
            source_head = source_s3.head_object(
                Bucket=source_bucket,
                Key=source_key,
            )
            validate_s3_source_head(source_head, archive)
            upload = copy_s3_object_multipart(
                s3_client=s3,
                source_bucket=source_bucket,
                source_key=source_key,
                source_etag=archive["expected_etag"],
                destination_bucket=datasets_bucket,
                destination_key=object_key,
                metadata={
                    "archive-id": archive["archive_id"],
                    "snapshot-id": manifest["snapshot_id"],
                    "source-contract-sha256": source_contract_sha256,
                    "source-etag": archive["expected_etag"],
                },
                expected_size_bytes=archive["expected_size_bytes"],
            )
        else:
            validate_public_https_uri(archive["source_uri"])

            class PublicHTTPSRedirectHandler(HTTPRedirectHandler):
                def redirect_request(
                    self,
                    request,
                    file_pointer,
                    code,
                    message,
                    headers,
                    new_url,
                ):
                    validate_public_https_uri(new_url)
                    return super().redirect_request(
                        request,
                        file_pointer,
                        code,
                        message,
                        headers,
                        new_url,
                    )

            request = Request(
                archive["source_uri"],
                headers={
                    "Accept-Encoding": "identity",
                    "User-Agent": "auto-e2e-nuplan-acquisition/1",
                },
            )
            try:
                source_response = build_opener(
                    PublicHTTPSRedirectHandler()
                ).open(
                    request,
                    timeout=120,
                )
            except HTTPError as error:
                raise RuntimeError(
                    "authorized HTTPS source returned "
                    f"status={error.code} "
                    f"archive_id={archive['archive_id']}"
                ) from None
            except URLError as error:
                raise RuntimeError(
                    "authorized HTTPS source connection failed "
                    f"archive_id={archive['archive_id']} "
                    f"reason_type={type(error.reason).__name__}"
                ) from None
            with closing(source_response) as source_stream:
                upload = upload_https_stream_multipart(
                    s3_client=s3,
                    stream=source_stream,
                    bucket=datasets_bucket,
                    key=object_key,
                    metadata={
                        "archive-id": archive["archive_id"],
                        "snapshot-id": manifest["snapshot_id"],
                        "source-contract-sha256": source_contract_sha256,
                    },
                    expected_size_bytes=archive["expected_size_bytes"],
                    expected_sha256=archive["expected_sha256"],
                    expected_md5=archive["expected_md5"],
                )
            upload.update({
                "checksum_crc64nvme": "",
                "source_etag": "",
                "transfer_mode": "https_stream_hash",
            })

    head = s3.head_object(
        Bucket=datasets_bucket,
        Key=object_key,
        **(
            {"ChecksumMode": "ENABLED"}
            if upload["transfer_mode"]
            == "s3_server_side_multipart_copy"
            else {}
        ),
    )
    if int(head["ContentLength"]) != int(upload["size_bytes"]):
        raise ValueError(
            "uploaded nuPlan archive size differs after completion: "
            f"{object_key}"
        )
    upload.setdefault(
        "destination_etag",
        str(head["ETag"]).strip('"').lower(),
    )
    receipt = {
        "archive_id": archive["archive_id"],
        "checksum_crc64nvme": upload.get("checksum_crc64nvme", ""),
        "component": archive["component"],
        "destination_etag": upload["destination_etag"],
        "md5": upload["md5"],
        "object_uri": f"s3://{datasets_bucket}/{object_key}",
        "schema_version": ARCHIVE_RECEIPT_SCHEMA_VERSION,
        "sha256": upload["sha256"],
        "size_bytes": upload["size_bytes"],
        "source_contract_sha256": source_contract_sha256,
        "source_etag": upload.get("source_etag", ""),
        "transfer_mode": upload["transfer_mode"],
    }
    receipt_bytes = canonical_json_bytes(receipt)
    try:
        s3.put_object(
            Bucket=datasets_bucket,
            Key=receipt_key,
            Body=receipt_bytes,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
        }:
            raise
        existing = s3.get_object(
            Bucket=datasets_bucket,
            Key=receipt_key,
        )["Body"].read()
        if existing != receipt_bytes:
            raise ValueError(
                "concurrent nuPlan receipt differs for "
                f"{archive['archive_id']!r}"
            ) from error
    print(
        "Imported nuPlan archive "
        f"id={archive['archive_id']} component={archive['component']} "
        f"size_bytes={upload['size_bytes']} "
        f"transfer_mode={upload['transfer_mode']} "
        f"integrity={upload.get('checksum_crc64nvme') or upload['sha256']}"
    )
    return receipt_output(receipt_bytes)


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    requests=Resources(cpu="1", mem="2Gi", ephemeral_storage="2Gi"),
    limits=Resources(cpu="1", mem="2Gi", ephemeral_storage="2Gi"),
)
def finalize_nuplan_raw_snapshot(
    source_manifest: FlyteFile,
    archive_receipts: List[FlyteFile],
    datasets_bucket: str,
    aws_region: str = "us-west-2",
) -> NuPlanRawSnapshotOutput:
    """Publish the redacted manifest after every archive is verified."""
    import json
    from pathlib import Path

    import boto3
    from botocore.exceptions import ClientError

    from Platform.pipelines.nuplan_acquisition import (
        build_snapshot_manifest,
        canonical_json_bytes,
        load_source_manifest_bytes,
        sha256_bytes,
        snapshot_manifest_key,
        snapshot_prefix,
    )

    source_bytes = Path(source_manifest.download()).read_bytes()
    manifest, source_contract_sha256 = load_source_manifest_bytes(source_bytes)
    receipts = [
        json.loads(Path(receipt.download()).read_text(encoding="utf-8"))
        for receipt in archive_receipts
    ]
    snapshot = build_snapshot_manifest(
        source_manifest=manifest,
        source_contract_sha256=source_contract_sha256,
        receipts=receipts,
    )
    payload = canonical_json_bytes(snapshot)
    payload_sha256 = sha256_bytes(payload)
    key = snapshot_manifest_key(manifest)
    s3 = boto3.client("s3", region_name=aws_region)
    try:
        s3.put_object(
            Bucket=datasets_bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            Metadata={"manifest-sha256": payload_sha256},
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
        }:
            raise
        existing = s3.get_object(
            Bucket=datasets_bucket,
            Key=key,
        )["Body"].read()
        if existing != payload:
            raise ValueError(
                "existing nuPlan snapshot manifest differs for "
                f"{manifest['snapshot_id']!r}"
            ) from error
    return NuPlanRawSnapshotOutput(
        manifest=FlyteFile(f"s3://{datasets_bucket}/{key}"),
        manifest_sha256=payload_sha256,
        snapshot_prefix=(
            f"s3://{datasets_bucket}/{snapshot_prefix(manifest)}"
        ),
        archive_count=len(receipts),
        total_size_bytes=int(snapshot["total_size_bytes"]),
    )


@dynamic(
    container_image=DATA_PREP_IMAGE,
    environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE},
)
def _acquire_nuplan_raw_snapshot(
    source_manifest: FlyteFile,
    datasets_bucket: str,
    aws_region: str,
    concurrency: int,
) -> NuPlanRawSnapshotOutput:
    """Fan out archive imports without serializing source URLs to nodes."""
    from pathlib import Path

    from Platform.pipelines.nuplan_acquisition import (
        load_source_manifest_bytes,
    )

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    manifest, _ = load_source_manifest_bytes(
        Path(source_manifest.download()).read_bytes()
    )
    importer = map_task(
        functools.partial(
            acquire_nuplan_archive,
            source_manifest=source_manifest,
            datasets_bucket=datasets_bucket,
            aws_region=aws_region,
        ),
        concurrency=concurrency,
    )
    receipts = importer(
        archive_index=list(range(len(manifest["archives"])))
    )
    return finalize_nuplan_raw_snapshot(
        source_manifest=source_manifest,
        archive_receipts=receipts,
        datasets_bucket=datasets_bucket,
        aws_region=aws_region,
    )


@workflow
def wf_acquire_nuplan_raw_snapshot(
    source_manifest: FlyteFile,
    datasets_bucket: str,
    aws_region: str = "us-west-2",
    concurrency: int = 4,
) -> NuPlanRawSnapshotOutput:
    """Acquire authorized archives once into an immutable S3 snapshot."""
    return _acquire_nuplan_raw_snapshot(
        source_manifest=source_manifest,
        datasets_bucket=datasets_bucket,
        aws_region=aws_region,
        concurrency=concurrency,
    )


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_data_prep_pod_template(),
    requests=Resources(
        cpu="16",
        mem="64Gi",
        ephemeral_storage="500Gi",
    ),
    limits=Resources(
        cpu="16",
        mem="64Gi",
        ephemeral_storage="500Gi",
    ),
    cache=True,
    cache_version="nuplan-snapshot-pack-v4-parallel",
    retries=1,
)
def pack_nuplan_snapshot_reactive_dataset(
    snapshot_manifest: FlyteFile,
    datasets_bucket: str,
    archive_ids: Optional[List[str]] = None,
    limit_total_scenarios: int = 0,
    image_size: int = 256,
    samples_per_shard: int = 1000,
    max_rejection_fraction: float = 0.0,
    aws_region: str = "us-west-2",
) -> FlyteDirectory:
    """Materialize an immutable raw snapshot and emit BEV v2 shards."""
    import hashlib
    import json
    import tempfile
    from pathlib import Path
    from urllib.parse import urlsplit

    import boto3
    from boto3.s3.transfer import TransferConfig

    from data_parsing.nuplan import pack_nuplan_local_dataset
    from data_parsing.nuplan.materialization import (
        discover_materialized_nuplan,
        extract_nuplan_archive,
        load_nuplan_snapshot_manifest,
        select_snapshot_archives,
        verify_archive_file,
    )

    if (
        not datasets_bucket
        or datasets_bucket.startswith("s3://")
        or "/" in datasets_bucket
    ):
        raise ValueError("datasets_bucket must be one S3 bucket name")
    if limit_total_scenarios < 0:
        raise ValueError("limit_total_scenarios must be non-negative")
    if image_size <= 0 or samples_per_shard <= 0:
        raise ValueError("image_size and samples_per_shard must be positive")
    if not 0.0 <= max_rejection_fraction < 1.0:
        raise ValueError("max_rejection_fraction must be in [0, 1)")

    manifest_bytes = Path(snapshot_manifest.download()).read_bytes()
    manifest = load_nuplan_snapshot_manifest(manifest_bytes)
    selected = select_snapshot_archives(manifest, archive_ids)
    workspace = Path(tempfile.mkdtemp(prefix="nuplan-snapshot-pack-"))
    dataset_root = workspace / "dataset"
    archive_root = workspace / "archives"
    output = workspace / "shards"
    archive_root.mkdir(parents=True)
    output.mkdir()
    s3 = boto3.client("s3", region_name=aws_region)
    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )
    extracted: list[dict[str, object]] = []
    for archive in selected:
        parsed = urlsplit(str(archive["object_uri"]))
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if bucket != datasets_bucket:
            raise ValueError(
                "nuPlan snapshot object is outside datasets_bucket: "
                f"{archive['archive_id']!r}"
            )
        head = s3.head_object(Bucket=bucket, Key=key)
        if int(head["ContentLength"]) != int(archive["size_bytes"]):
            raise ValueError(
                "nuPlan snapshot object size changed for "
                f"{archive['archive_id']!r}"
            )
        archive_path = (
            archive_root
            / str(archive["archive_id"])
            / str(archive["filename"])
        )
        archive_path.parent.mkdir(parents=True)
        s3.download_file(
            bucket,
            key,
            str(archive_path),
            Config=transfer_config,
        )
        verify_archive_file(archive_path, archive)
        stats = extract_nuplan_archive(
            archive_path,
            archive,
            dataset_root,
            map_version=str(manifest["map_version"]),
        )
        archive_path.unlink()
        extracted.append({
            "archive_id": archive["archive_id"],
            **stats,
        })

    materialized = discover_materialized_nuplan(dataset_root)
    pack_workers = _nuplan_pack_worker_count(
        len(materialized.db_files),
        limit_total_scenarios,
    )
    packed = pack_nuplan_local_dataset(
        data_root=materialized.data_root,
        map_root=materialized.map_root,
        sensor_root=materialized.sensor_root,
        db_files=materialized.db_files,
        output_directory=output,
        source_revision=str(manifest["dataset_revision"]),
        map_version=materialized.map_version,
        limit_total_scenarios=limit_total_scenarios,
        image_size=image_size,
        samples_per_shard=samples_per_shard,
        max_rejection_fraction=max_rejection_fraction,
        pack_workers=pack_workers,
    )
    if (
        packed.get("bev_taxonomy_version")
        != BEV_SEGMENTATION_TAXONOMY_VERSION
    ):
        raise ValueError("nuPlan packer did not emit the current BEV taxonomy")
    if packed.get("schema_version") != "nuplan_reactive_manifest_v2":
        raise ValueError("nuPlan packer did not emit manifest schema v2")
    packed_counts = {
        name: packed.get(name)
        for name in (
            "bev_statistics_count",
            "bev_segmentation_count",
            "total_samples",
        )
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in packed_counts.values()
    ):
        raise ValueError("nuPlan packer emitted invalid sample counts")
    if (
        packed_counts["bev_statistics_count"]
        != packed_counts["total_samples"]
    ):
        raise ValueError("nuPlan packer omitted BEV v2 sample statistics")
    if (
        packed_counts["bev_segmentation_count"]
        != packed_counts["total_samples"]
    ):
        raise ValueError("nuPlan packer omitted BEV segmentation samples")

    packed.update({
        "raw_archive_ids": [
            archive["archive_id"] for archive in selected
        ],
        "raw_archive_materialization": extracted,
        "raw_snapshot_id": manifest["snapshot_id"],
        "raw_snapshot_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "raw_snapshot_map_version": manifest["map_version"],
        "raw_source_contract_sha256": (
            manifest["source_contract_sha256"]
        ),
        "materialized_map_version": materialized.map_version,
        "sensor_complete_log_count": len(
            materialized.sensor_log_names
        ),
    })
    (output / "manifest.json").write_text(
        json.dumps(packed, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return FlyteDirectory(str(output))


@workflow
def wf_pack_nuplan_snapshot_reactive_dataset(
    snapshot_manifest: FlyteFile,
    datasets_bucket: str,
    archive_ids: Optional[List[str]] = None,
    limit_total_scenarios: int = 0,
    image_size: int = 256,
    samples_per_shard: int = 1000,
    max_rejection_fraction: float = 0.0,
    aws_region: str = "us-west-2",
) -> FlyteDirectory:
    """Build Stage A BEV v2 shards from an immutable raw snapshot."""
    return pack_nuplan_snapshot_reactive_dataset(
        snapshot_manifest=snapshot_manifest,
        datasets_bucket=datasets_bucket,
        archive_ids=archive_ids,
        limit_total_scenarios=limit_total_scenarios,
        image_size=image_size,
        samples_per_shard=samples_per_shard,
        max_rejection_fraction=max_rejection_fraction,
        aws_region=aws_region,
    )
