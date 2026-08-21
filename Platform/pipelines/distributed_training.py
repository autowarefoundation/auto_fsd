"""Flyte Ray tasks for distributed AutoE2E training."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, NamedTuple, Optional

from flytekit import (
    PodTemplate,
    Resources,
    current_context,
    task,
    workflow,
)
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekitplugins.ray import (
    HeadNodeConfig,
    RayJobConfig,
    WorkerNodeConfig,
)
from kubernetes.client import (
    V1Affinity,
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1LabelSelector,
    V1LabelSelectorRequirement,
    V1PodAffinity,
    V1PodAffinityTerm,
    V1PodAntiAffinity,
    V1PodSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)


TRAINING_IMAGE = os.environ.get(
    "AUTO_E2E_TRAINING_IMAGE",
    "auto-e2e/training:latest",
)
RAY_STORAGE_PATH = os.environ.get(
    "AUTO_E2E_RAY_STORAGE_PATH",
    "s3://auto-e2e-platform-checkpoints/ray-train",
)
RAY_TASK_ENVIRONMENT = {
    "AWS_DEFAULT_REGION": "us-west-2",
    "AUTO_E2E_RAY_STORAGE_PATH": RAY_STORAGE_PATH,
    "RAY_TRAIN_V2_ENABLED": "1",
}
BEV_OVERFIT_SAMPLE_COUNT = 64
BEV_OVERFIT_MIN_AP = 0.9
BEV_OVERFIT_MIN_RECALL = 0.9


class RaySmokeOutput(NamedTuple):
    report: FlyteFile


class ReactiveRayOutput(NamedTuple):
    checkpoint: FlyteFile
    metadata: FlyteFile
    checkpoint_uri: str
    checkpoint_sha256: str


class ReactiveDistributedProgramOutput(NamedTuple):
    stage_a_checkpoint: FlyteFile
    stage_a_metadata: FlyteFile
    stage_a_checkpoint_uri: str
    stage_a_checkpoint_sha256: str
    stage_b_checkpoint: FlyteFile
    stage_b_metadata: FlyteFile
    stage_b_checkpoint_uri: str
    stage_b_checkpoint_sha256: str


class ReactiveCanaryOutput(NamedTuple):
    stage_a_checkpoint: FlyteFile
    stage_b_checkpoint: FlyteFile
    stage_a_metadata: FlyteFile
    stage_b_metadata: FlyteFile
    gate_report: FlyteFile


def _head_pod_template() -> PodTemplate:
    return PodTemplate(
        primary_container_name="ray-head",
        labels={"auto-e2e.training/role": "ray-head"},
        annotations={"karpenter.sh/do-not-disrupt": "true"},
        pod_spec=V1PodSpec(
            service_account_name="default",
            containers=[
                V1Container(
                    name="ray-head",
                    resources=V1ResourceRequirements(
                        requests={"cpu": "2", "memory": "16Gi"},
                        limits={"cpu": "2", "memory": "16Gi"},
                    ),
                    volume_mounts=[
                        V1VolumeMount(
                            name="dshm",
                            mount_path="/dev/shm",
                        ),
                    ],
                ),
            ],
            volumes=[
                V1Volume(
                        name="dshm",
                        empty_dir=V1EmptyDirVolumeSource(
                            medium="Memory",
                            size_limit="8Gi",
                        ),
                ),
            ],
        ),
    )


def _worker_pod_template(
    *,
    workload_type: str,
    cpu: str,
    memory: str,
    shm_size: str,
) -> PodTemplate:
    return PodTemplate(
        primary_container_name="ray-worker",
        labels={"auto-e2e.training/role": "ray-gpu-worker"},
        annotations={"karpenter.sh/do-not-disrupt": "true"},
        pod_spec=V1PodSpec(
            service_account_name="default",
            node_selector={"workload-type": workload_type},
            tolerations=[
                V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule",
                ),
            ],
            affinity=V1Affinity(
                pod_affinity=V1PodAffinity(
                    required_during_scheduling_ignored_during_execution=[
                        V1PodAffinityTerm(
                            label_selector=V1LabelSelector(
                                match_expressions=[
                                    V1LabelSelectorRequirement(
                                        key="auto-e2e.training/role",
                                        operator="In",
                                        values=["ray-gpu-worker"],
                                    ),
                                ],
                            ),
                            topology_key="topology.kubernetes.io/zone",
                        ),
                    ],
                ),
                pod_anti_affinity=V1PodAntiAffinity(
                    required_during_scheduling_ignored_during_execution=[
                        V1PodAffinityTerm(
                            label_selector=V1LabelSelector(
                                match_expressions=[
                                    V1LabelSelectorRequirement(
                                        key="auto-e2e.training/role",
                                        operator="In",
                                        values=["ray-gpu-worker"],
                                    ),
                                ],
                            ),
                            topology_key="kubernetes.io/hostname",
                        ),
                    ],
                ),
            ),
            containers=[
                V1Container(
                    name="ray-worker",
                    env=[
                        V1EnvVar(name="NCCL_DEBUG", value="INFO"),
                        V1EnvVar(
                            name="TORCH_DISTRIBUTED_DEBUG",
                            value="DETAIL",
                        ),
                    ],
                    resources=V1ResourceRequirements(
                        requests={
                            "cpu": cpu,
                            "memory": memory,
                            "nvidia.com/gpu": "1",
                        },
                        limits={
                            "cpu": cpu,
                            "memory": memory,
                            "nvidia.com/gpu": "1",
                        },
                    ),
                    volume_mounts=[
                        V1VolumeMount(
                            name="dshm",
                            mount_path="/dev/shm",
                        ),
                    ],
                ),
            ],
            volumes=[
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory",
                        size_limit=shm_size,
                    ),
                ),
            ],
        ),
    )


def _ray_job_config(
    replicas: int,
    *,
    worker_workload_type: str,
    worker_cpu: str = "4",
    worker_memory: str = "16Gi",
    worker_shm_size: str = "8Gi",
) -> RayJobConfig:
    return RayJobConfig(
        head_node_config=HeadNodeConfig(
            ray_start_params={
                "dashboard-host": "0.0.0.0",
                "num-cpus": "0",
            },
            pod_template=_head_pod_template(),
        ),
        worker_node_config=[
            WorkerNodeConfig(
                group_name="gpu-workers",
                replicas=replicas,
                min_replicas=replicas,
                max_replicas=replicas,
                ray_start_params={
                    "num-cpus": worker_cpu,
                    "num-gpus": "1",
                },
                pod_template=_worker_pod_template(
                    workload_type=worker_workload_type,
                    cpu=worker_cpu,
                    memory=worker_memory,
                    shm_size=worker_shm_size,
                ),
            ),
        ],
        enable_autoscaling=False,
        address="auto",
        shutdown_after_job_finishes=True,
        ttl_seconds_after_finished=300,
    )


RAY_2 = _ray_job_config(
    2,
    worker_workload_type="gpu-canary",
    worker_cpu="3",
    worker_memory="12Gi",
    worker_shm_size="4Gi",
)
RAY_4 = _ray_job_config(
    4,
    worker_workload_type="gpu-training",
)
RAY_REACTIVE_4 = _ray_job_config(
    4,
    worker_workload_type="gpu-performance",
    worker_cpu="3",
    worker_memory="12Gi",
    worker_shm_size="4Gi",
)
RAY_8 = _ray_job_config(
    8,
    worker_workload_type="gpu-training",
)


@task(
    task_config=RAY_4,
    container_image=TRAINING_IMAGE,
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "training",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def ray_ddp_smoke_4(steps: int = 4) -> RaySmokeOutput:
    from distributed_training.ray_smoke import run_smoke

    context = current_context()
    execution_name = (
        context.execution_id.name
        if context.execution_id is not None
        else "local"
    )
    run_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "-",
        f"{execution_name}-ray-ddp-smoke-4",
    )
    result = run_smoke(
        num_workers=4,
        steps=steps,
        storage_path=RAY_STORAGE_PATH,
        run_name=run_name,
    )
    report_path = Path("/tmp/ray-ddp-smoke/report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return RaySmokeOutput(report=FlyteFile(str(report_path)))


def _flyte_remote_uri(value: FlyteDirectory | FlyteFile) -> str:
    remote_source = str(getattr(value, "remote_source", "") or "")
    uri = remote_source or str(value)
    if not uri.startswith("s3://"):
        raise ValueError(
            "distributed Ray workers require immutable S3 inputs"
        )
    return uri.rstrip("/")


def _reactive_worker_cpus(num_workers: int) -> int:
    """Match Ray actors to the corresponding worker pod CPU limit."""
    return 3 if num_workers in {2, 4} else 4


def _run_reactive_stage_task(
    *,
    shards: List[FlyteDirectory],
    stage: str,
    num_workers: int,
    parent_checkpoint: Optional[FlyteFile],
    backbone: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    val_fraction: float,
    num_loader_workers: int,
    training_seed: int,
    precision: str,
    gradient_accumulation_steps: int,
    steps_per_epoch: int,
    shuffle_buffer: int,
    is_pretrained: bool,
    bev_weight: float,
    route_weight: float,
    corridor_pos_weight: float,
    bev_pos_weight_cap: float,
    bev_repeat_frequency_threshold: float,
    bev_max_repeat: int,
    bev_min_positive_samples: int,
    bev_min_positive_cells: int,
    overfit_sample_count: int,
    overfit_shard_limit: int,
    overfit_min_ap: float,
    overfit_min_recall: float,
    validation_sample_limit: int,
    required_gate_dataset_manifest_sha256: str = "",
) -> ReactiveRayOutput:
    from distributed_training.reactive_stage import run_reactive_stage

    context = current_context()
    execution_name = (
        context.execution_id.name
        if context.execution_id is not None
        else "local"
    )
    run_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "-",
        (
            f"{execution_name}-{stage}-ray-{num_workers}-"
            f"{'overfit-' + str(overfit_sample_count) if overfit_sample_count else 'full'}"
        ),
    )
    source_uris = [_flyte_remote_uri(shard) for shard in shards]
    parent_uri = (
        _flyte_remote_uri(parent_checkpoint)
        if parent_checkpoint is not None
        else ""
    )
    result = run_reactive_stage({
        "backbone": backbone,
        "bev_ap_bins": 1024,
        "bev_max_repeat": bev_max_repeat,
        "bev_min_positive_cells": bev_min_positive_cells,
        "bev_min_positive_samples": bev_min_positive_samples,
        "bev_pos_weight_cap": bev_pos_weight_cap,
        "bev_repeat_frequency_threshold": (
            bev_repeat_frequency_threshold
        ),
        "bev_weight": bev_weight,
        "corridor_pos_weight": corridor_pos_weight,
        "epochs": epochs,
        "grad_clip": grad_clip,
        "gradient_accumulation_steps": (
            gradient_accumulation_steps
        ),
        "is_pretrained": is_pretrained,
        "learning_rate": learning_rate,
        "local_cache_root": "/tmp/auto-e2e-reactive",
        "num_loader_workers": num_loader_workers,
        "num_workers": num_workers,
        "overfit_min_ap": overfit_min_ap,
        "overfit_min_recall": overfit_min_recall,
        "overfit_sample_count": overfit_sample_count,
        "overfit_shard_limit": overfit_shard_limit,
        "parent_checkpoint_uri": parent_uri,
        "per_rank_batch_size": 1,
        "precision": precision,
        "route_weight": route_weight,
        "run_name": run_name,
        "required_gate_dataset_manifest_sha256": (
            required_gate_dataset_manifest_sha256
        ),
        "selection_ade_regression_margin_m": 0.5,
        "selection_ade_scale_m": 5.0,
        "shuffle_buffer": shuffle_buffer,
        "source_uris": source_uris,
        "stage": stage,
        "steps_per_epoch": steps_per_epoch,
        "storage_path": RAY_STORAGE_PATH,
        "training_seed": training_seed,
        "use_gpu": True,
        "val_fraction": val_fraction,
        "validation_sample_limit": validation_sample_limit,
        "weight_decay": weight_decay,
        "worker_cpus": _reactive_worker_cpus(num_workers),
    })
    metadata_path = (
        Path("/tmp/reactive-ray")
        / run_name
        / "metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    metrics = result["metrics"]
    return ReactiveRayOutput(
        checkpoint=FlyteFile(result["checkpoint_file_uri"]),
        metadata=FlyteFile(str(metadata_path)),
        checkpoint_uri=str(result["checkpoint_file_uri"]),
        checkpoint_sha256=str(metrics["checkpoint_sha256"]),
    )


def _validated_bev_overfit_gate_dataset(
    source: FlyteFile,
) -> str:
    """Return the gated dataset digest after validating all overfit evidence."""
    import math

    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )

    payload = json.loads(Path(source.download()).read_text())
    metrics = payload.get("metrics")
    history = payload.get("history")
    if not isinstance(metrics, dict):
        raise ValueError("BEV overfit gate metadata omitted final metrics")
    if not isinstance(history, list) or not history:
        raise ValueError("BEV overfit gate metadata omitted epoch history")
    final_history = history[-1]
    if not isinstance(final_history, dict):
        raise ValueError("BEV overfit gate final history is invalid")

    integer_contract = {
        "overfit_gate_pass": 1,
        "world_size": 4,
    }
    for name, expected in integer_contract.items():
        if int(metrics.get(name, -1)) != expected:
            raise ValueError(
                f"BEV overfit gate has invalid {name}: {metrics.get(name)!r}"
            )
        if int(final_history.get(name, -1)) != expected:
            raise ValueError(
                f"BEV overfit gate history has invalid {name}"
            )
    sample_count = int(metrics.get("overfit_sample_count", -1))
    if not BEV_OVERFIT_SAMPLE_COUNT <= sample_count <= 128:
        raise ValueError(
            "BEV overfit gate sample count must be between 64 and 128"
        )
    if int(final_history.get("overfit_sample_count", -1)) != sample_count:
        raise ValueError(
            "BEV overfit gate history has invalid overfit_sample_count"
        )

    for name in (
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "overfit_sample_uid_sha256",
    ):
        value = str(metrics.get(name, ""))
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"BEV overfit gate has invalid {name}")
        if str(final_history.get(name, "")) != value:
            raise ValueError(
                f"BEV overfit gate history disagrees on {name}"
            )

    pos_weights = []
    for class_index, class_name in enumerate(BEV_SEGMENTATION_CLASSES):
        average_precision = float(metrics.get(
            f"validation_bev_{class_name}_average_precision",
            float("nan"),
        ))
        recall = float(metrics.get(
            f"validation_bev_{class_name}_recall",
            float("nan"),
        ))
        positive_cells = float(metrics.get(
            f"validation_bev_{class_name}_positive_cells",
            float("nan"),
        ))
        if (
            not math.isfinite(average_precision)
            or average_precision < BEV_OVERFIT_MIN_AP
        ):
            raise ValueError(
                f"BEV overfit gate average precision failed for {class_name}"
            )
        if (
            not math.isfinite(recall)
            or recall < BEV_OVERFIT_MIN_RECALL
        ):
            raise ValueError(
                f"BEV overfit gate recall failed for {class_name}"
            )
        if (
            not math.isfinite(positive_cells)
            or positive_cells <= 0.0
        ):
            raise ValueError(
                f"BEV overfit gate has no positives for {class_name}"
            )
        weight = float(metrics.get(
            f"bev_pos_weight_{class_index}",
            float("nan"),
        ))
        if not math.isfinite(weight) or weight < 1.0:
            raise ValueError(
                f"BEV overfit gate has invalid weight for {class_name}"
            )
        pos_weights.append(weight)
    if all(abs(weight - 1.0) <= 1e-12 for weight in pos_weights):
        raise ValueError("BEV overfit gate derived only unit pos weights")

    return str(metrics["dataset_manifest_sha256"])


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="2", mem="8Gi"),
)
def build_reactive_canary_dataset(stage: str) -> FlyteDirectory:
    """Create deterministic production-schema shards for the GPU gate."""
    import tempfile
    from pathlib import Path

    from distributed_training.reactive_canary_data import (
        write_reactive_canary_dataset,
    )
    from training.reactive_multitask import ReactiveTrainingStage

    training_stage = ReactiveTrainingStage(stage)
    output = Path(tempfile.mkdtemp(prefix=f"reactive-{stage}-"))
    write_reactive_canary_dataset(
        output,
        stage=training_stage,
        shard_count=2,
        train_samples_per_shard=2,
        validation_samples_per_shard=1,
    )
    return FlyteDirectory(str(output))


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def verify_reactive_canary_training(
    stage_a_metadata: FlyteFile,
    stage_b_metadata: FlyteFile,
) -> FlyteFile:
    """Fail when the real-model two-stage GPU canary is not learning."""
    import math
    import tempfile
    from pathlib import Path

    reports = {}
    for stage_name, source in (
        ("stage_a", stage_a_metadata),
        ("stage_b", stage_b_metadata),
    ):
        payload = json.loads(Path(source.download()).read_text())
        history = payload.get("history")
        if not isinstance(history, list) or len(history) < 2:
            raise ValueError(
                f"{stage_name} canary needs at least two reported epochs"
            )
        required = (
            "train_bev_segmentation",
            "train_route_reconstruction",
            "train_total",
            "train_trajectory",
            "validation_ade_6p4s_m",
            "validation_selection_score",
        )
        for epoch in history:
            if any(
                name not in epoch
                or not math.isfinite(float(epoch[name]))
                for name in required
            ):
                raise ValueError(
                    f"{stage_name} canary emitted non-finite metrics"
                )
        reports[stage_name] = history

    stage_a = reports["stage_a"]
    stage_b = reports["stage_b"]
    if float(stage_a[0]["train_bev_segmentation"]) <= 0.0:
        raise ValueError("Stage A canary did not execute the BEV loss")
    from data_processing.reactive_training_artifacts import (
        BEV_SEGMENTATION_CLASSES,
    )

    for epoch in stage_a:
        for class_name in BEV_SEGMENTATION_CLASSES:
            for suffix in (
                "average_precision",
                "positive_cells",
                "recall",
            ):
                name = f"validation_bev_{class_name}_{suffix}"
                if (
                    name not in epoch
                    or not math.isfinite(float(epoch[name]))
                ):
                    raise ValueError(
                        f"Stage A canary omitted class metric {name}"
                    )
            if float(
                epoch[
                    f"validation_bev_{class_name}_positive_cells"
                ]
            ) <= 0.0:
                raise ValueError(
                    f"Stage A canary has no {class_name} positives"
                )
    if all(
        abs(float(stage_a[0][f"bev_pos_weight_{index}"]) - 1.0)
        <= 1e-12
        for index in range(8)
    ):
        raise ValueError("Stage A canary derived only unit BEV weights")
    if any(
        abs(float(epoch["train_bev_segmentation"])) > 1e-12
        for epoch in stage_b
    ):
        raise ValueError("Stage B canary executed the BEV loss")
    initial_total = float(stage_a[0]["train_total"])
    minimum_later_total = min(
        float(epoch["train_total"]) for epoch in stage_a[1:]
    )
    if minimum_later_total >= initial_total:
        raise ValueError(
            "Stage A canary total loss did not decrease: "
            f"initial={initial_total} later_min={minimum_later_total}"
        )
    initial_bev = float(stage_a[0]["train_bev_segmentation"])
    minimum_later_bev = min(
        float(epoch["train_bev_segmentation"])
        for epoch in stage_a[1:]
    )
    if minimum_later_bev >= initial_bev:
        raise ValueError(
            "Stage A canary BEV loss did not decrease: "
            f"initial={initial_bev} later_min={minimum_later_bev}"
        )

    report = {
        "schema_version": "reactive_ddp_canary_report_v2",
        "stage_a_initial_bev": initial_bev,
        "stage_a_minimum_later_bev": minimum_later_bev,
        "stage_a_initial_total": initial_total,
        "stage_a_minimum_later_total": minimum_later_total,
        "stage_a_epochs": len(stage_a),
        "stage_b_epochs": len(stage_b),
        "stage_b_bev_loss_disabled": True,
        "thresholds_pass": True,
    }
    output = (
        Path(tempfile.mkdtemp(prefix="reactive-canary-report-"))
        / "report.json"
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return FlyteFile(str(output))


@task(
    task_config=RAY_2,
    container_image=TRAINING_IMAGE,
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-canary",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_stage_ray_2(
    shards: List[FlyteDirectory],
    stage: str,
    parent_checkpoint: Optional[FlyteFile] = None,
    backbone: str = "swin_v2_tiny",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    training_seed: int = 149,
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    steps_per_epoch: int = 2,
    shuffle_buffer: int = 64,
    is_pretrained: bool = False,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    corridor_pos_weight: float = 1.0,
) -> ReactiveRayOutput:
    """Run a two-node real-model integration canary."""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=2,
        parent_checkpoint=parent_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        shuffle_buffer=shuffle_buffer,
        is_pretrained=is_pretrained,
        bev_weight=bev_weight,
        route_weight=route_weight,
        corridor_pos_weight=corridor_pos_weight,
        bev_pos_weight_cap=64.0,
        bev_repeat_frequency_threshold=0.05,
        bev_max_repeat=4,
        bev_min_positive_samples=1,
        bev_min_positive_cells=1,
        overfit_sample_count=0,
        overfit_shard_limit=0,
        overfit_min_ap=0.9,
        overfit_min_recall=0.9,
        validation_sample_limit=256,
    )


@task(
    task_config=RAY_REACTIVE_4,
    container_image=TRAINING_IMAGE,
    retries=0,
    labels={
        "kueue.x-k8s.io/queue-name": "gpu-performance",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_stage_ray_4(
    shards: List[FlyteDirectory],
    stage: str,
    parent_checkpoint: Optional[FlyteFile] = None,
    gate_metadata: Optional[FlyteFile] = None,
    backbone: str = "swin_v2_tiny",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.2,
    num_loader_workers: int = 2,
    training_seed: int = 149,
    precision: str = "bf16",
    gradient_accumulation_steps: int = 1,
    steps_per_epoch: int = 0,
    shuffle_buffer: int = 1000,
    is_pretrained: bool = True,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    corridor_pos_weight: float = 1.0,
    overfit_sample_count: int = 0,
    overfit_min_ap: float = 0.9,
    overfit_min_recall: float = 0.9,
) -> ReactiveRayOutput:
    """Run a four-rank Reactive performance training stage."""
    if overfit_sample_count:
        if gate_metadata is not None:
            raise ValueError("BEV overfit runs cannot consume gate metadata")
        required_gate_dataset = ""
    else:
        if gate_metadata is None:
            raise ValueError(
                "four-rank full training requires BEV overfit gate metadata"
            )
        required_gate_dataset = _validated_bev_overfit_gate_dataset(
            gate_metadata
        )
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=4,
        parent_checkpoint=parent_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        shuffle_buffer=shuffle_buffer,
        is_pretrained=is_pretrained,
        bev_weight=bev_weight,
        route_weight=route_weight,
        corridor_pos_weight=corridor_pos_weight,
        bev_pos_weight_cap=64.0,
        bev_repeat_frequency_threshold=0.05,
        bev_max_repeat=4,
        bev_min_positive_samples=(
            1 if overfit_sample_count else 20
        ),
        bev_min_positive_cells=(
            1 if overfit_sample_count else 2000
        ),
        overfit_sample_count=overfit_sample_count,
        overfit_shard_limit=(32 if overfit_sample_count else 0),
        overfit_min_ap=overfit_min_ap,
        overfit_min_recall=overfit_min_recall,
        required_gate_dataset_manifest_sha256=required_gate_dataset,
        validation_sample_limit=1024,
    )


@task(
    task_config=RAY_8,
    container_image=TRAINING_IMAGE,
    retries=1,
    labels={
        "kueue.x-k8s.io/queue-name": "training",
        "kueue.x-k8s.io/priority-class": "research-low",
    },
    environment=RAY_TASK_ENVIRONMENT,
)
def train_reactive_stage_ray_8(
    shards: List[FlyteDirectory],
    stage: str,
    parent_checkpoint: Optional[FlyteFile] = None,
    gate_metadata: Optional[FlyteFile] = None,
    backbone: str = "swin_v2_tiny",
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    training_seed: int = 149,
    precision: str = "bf16",
    gradient_accumulation_steps: int = 1,
    steps_per_epoch: int = 0,
    shuffle_buffer: int = 1000,
    is_pretrained: bool = True,
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
    corridor_pos_weight: float = 1.0,
) -> ReactiveRayOutput:
    """Run one production-size Reactive DDP stage."""
    if stage == "nuplan_full":
        if gate_metadata is None:
            raise ValueError(
                "eight-rank Stage A requires BEV overfit gate metadata"
            )
        required_gate_dataset = _validated_bev_overfit_gate_dataset(
            gate_metadata
        )
    else:
        if gate_metadata is not None:
            raise ValueError(
                "Stage B cannot consume BEV overfit gate metadata"
            )
        required_gate_dataset = ""
    return _run_reactive_stage_task(
        shards=shards,
        stage=stage,
        num_workers=8,
        parent_checkpoint=parent_checkpoint,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        shuffle_buffer=shuffle_buffer,
        is_pretrained=is_pretrained,
        bev_weight=bev_weight,
        route_weight=route_weight,
        corridor_pos_weight=corridor_pos_weight,
        bev_pos_weight_cap=64.0,
        bev_repeat_frequency_threshold=0.05,
        bev_max_repeat=4,
        bev_min_positive_samples=20,
        bev_min_positive_cells=2000,
        overfit_sample_count=0,
        overfit_shard_limit=0,
        overfit_min_ap=0.9,
        overfit_min_recall=0.9,
        required_gate_dataset_manifest_sha256=required_gate_dataset,
        validation_sample_limit=1024,
    )


@workflow
def wf_ray_ddp_smoke_4(steps: int = 4) -> FlyteFile:
    return ray_ddp_smoke_4(steps=steps).report


@workflow
def wf_overfit_reactive_nuplan_ray_4(
    nuplan_shards: List[FlyteDirectory],
    sample_count: int = BEV_OVERFIT_SAMPLE_COUNT,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    val_fraction: float = 0.2,
    training_seed: int = 149,
) -> ReactiveRayOutput:
    """Require near-perfect memorization before corpus-scale training."""
    return train_reactive_stage_ray_4(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        gate_metadata=None,
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=2,
        training_seed=training_seed,
        precision="bf16",
        gradient_accumulation_steps=1,
        steps_per_epoch=0,
        shuffle_buffer=256,
        is_pretrained=True,
        bev_weight=1.0,
        route_weight=1.0,
        overfit_sample_count=sample_count,
        overfit_min_ap=BEV_OVERFIT_MIN_AP,
        overfit_min_recall=BEV_OVERFIT_MIN_RECALL,
    )


@workflow
def wf_train_reactive_nuplan_ray_4(
    nuplan_shards: List[FlyteDirectory],
    epochs: int = 30,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.2,
    num_loader_workers: int = 2,
    training_seed: int = 149,
    precision: str = "bf16",
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
) -> ReactiveRayOutput:
    """Pass the BEV overfit gate, then train Stage A from initial weights."""
    overfit = train_reactive_stage_ray_4(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        gate_metadata=None,
        epochs=50,
        learning_rate=3e-4,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        steps_per_epoch=0,
        shuffle_buffer=256,
        is_pretrained=True,
        bev_weight=bev_weight,
        route_weight=route_weight,
        overfit_sample_count=BEV_OVERFIT_SAMPLE_COUNT,
        overfit_min_ap=BEV_OVERFIT_MIN_AP,
        overfit_min_recall=BEV_OVERFIT_MIN_RECALL,
    )
    return train_reactive_stage_ray_4(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        gate_metadata=overfit.metadata,
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        steps_per_epoch=0,
        shuffle_buffer=1000,
        is_pretrained=True,
        bev_weight=bev_weight,
        route_weight=route_weight,
    )


@workflow
def wf_train_reactive_nuplan_l2d_ray_8(
    nuplan_shards: List[FlyteDirectory],
    l2d_shards: List[FlyteDirectory],
    stage_a_epochs: int = 3,
    stage_b_epochs: int = 3,
    stage_a_learning_rate: float = 1e-4,
    stage_b_learning_rate: float = 3e-5,
    val_fraction: float = 0.1,
    num_loader_workers: int = 2,
    training_seed: int = 149,
    precision: str = "bf16",
    bev_weight: float = 1.0,
    route_weight: float = 1.0,
) -> ReactiveDistributedProgramOutput:
    """Train Stage A and Stage B as separate eight-rank RayJobs."""
    overfit = train_reactive_stage_ray_4(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        gate_metadata=None,
        epochs=50,
        learning_rate=3e-4,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        steps_per_epoch=0,
        shuffle_buffer=256,
        is_pretrained=True,
        bev_weight=bev_weight,
        route_weight=route_weight,
        overfit_sample_count=BEV_OVERFIT_SAMPLE_COUNT,
        overfit_min_ap=BEV_OVERFIT_MIN_AP,
        overfit_min_recall=BEV_OVERFIT_MIN_RECALL,
    )
    stage_a = train_reactive_stage_ray_8(
        shards=nuplan_shards,
        stage="nuplan_full",
        parent_checkpoint=None,
        gate_metadata=overfit.metadata,
        epochs=stage_a_epochs,
        learning_rate=stage_a_learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        bev_weight=bev_weight,
        route_weight=route_weight,
    )
    stage_b = train_reactive_stage_ray_8(
        shards=l2d_shards,
        stage="l2d_continuation",
        parent_checkpoint=stage_a.checkpoint,
        gate_metadata=None,
        epochs=stage_b_epochs,
        learning_rate=stage_b_learning_rate,
        val_fraction=val_fraction,
        num_loader_workers=num_loader_workers,
        training_seed=training_seed,
        precision=precision,
        bev_weight=0.0,
        route_weight=route_weight,
    )
    return ReactiveDistributedProgramOutput(
        stage_a_checkpoint=stage_a.checkpoint,
        stage_a_metadata=stage_a.metadata,
        stage_a_checkpoint_uri=stage_a.checkpoint_uri,
        stage_a_checkpoint_sha256=stage_a.checkpoint_sha256,
        stage_b_checkpoint=stage_b.checkpoint,
        stage_b_metadata=stage_b.metadata,
        stage_b_checkpoint_uri=stage_b.checkpoint_uri,
        stage_b_checkpoint_sha256=stage_b.checkpoint_sha256,
    )


@workflow
def wf_reactive_multistage_ray_2_canary() -> ReactiveCanaryOutput:
    """Run the real Reactive objectives through two multi-node RayJobs."""
    stage_a_data = build_reactive_canary_dataset(
        stage="nuplan_full"
    )
    stage_b_data = build_reactive_canary_dataset(
        stage="l2d_continuation"
    )
    stage_a = train_reactive_stage_ray_2(
        shards=[stage_a_data],
        stage="nuplan_full",
        parent_checkpoint=None,
        epochs=3,
        learning_rate=3e-4,
        val_fraction=0.5,
        num_loader_workers=1,
        precision="fp32",
        steps_per_epoch=4,
        shuffle_buffer=0,
        is_pretrained=False,
        bev_weight=0.1,
        route_weight=0.1,
    )
    stage_b = train_reactive_stage_ray_2(
        shards=[stage_b_data],
        stage="l2d_continuation",
        parent_checkpoint=stage_a.checkpoint,
        epochs=2,
        learning_rate=1e-4,
        val_fraction=0.5,
        num_loader_workers=1,
        precision="fp32",
        steps_per_epoch=2,
        shuffle_buffer=0,
        is_pretrained=False,
        bev_weight=0.0,
        route_weight=0.1,
    )
    gate_report = verify_reactive_canary_training(
        stage_a_metadata=stage_a.metadata,
        stage_b_metadata=stage_b.metadata,
    )
    return ReactiveCanaryOutput(
        stage_a_checkpoint=stage_a.checkpoint,
        stage_b_checkpoint=stage_b.checkpoint,
        stage_a_metadata=stage_a.metadata,
        stage_b_metadata=stage_b.metadata,
        gate_report=gate_report,
    )
