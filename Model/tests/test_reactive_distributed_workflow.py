"""Flyte wiring for distributed Reactive Stage A and Stage B."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from flytekit.types.directory import FlyteDirectory

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
)
from distributed_training.reactive_stage import (
    BEV_LANE_RANGE_METRIC_PREFIX,
    CAMERA_FEATURE_SCALE_WEIGHT_METRIC_PREFIX,
    MIN_OVERFIT_OPTIMIZER_STEPS,
    OVERFIT_POSITIVE_SAMPLE_SUPPORT_METRIC_PREFIX,
    PEAK_CUDA_ALLOCATED_BYTES_METRIC_PREFIX,
    PEAK_CUDA_RESERVED_BYTES_METRIC_PREFIX,
)
from Platform.pipelines import (
    distributed_training,
    nuplan_dataset,
    workflows,
)


def test_distributed_workflow_import_is_path_order_independent():
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_paths = [
        str(repository_root / "Model"),
        str(repository_root),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import reactive_training_contracts as contracts; "
                "assert 'torch' not in sys.modules; "
                "assert 'numpy' not in sys.modules; "
                "import distributed_training as module; "
                "assert module.BEV_OVERFIT_SAMPLE_COUNT == "
                "contracts.MIN_OVERFIT_SAMPLE_COUNT; "
                "print(module.BEV_OVERFIT_SAMPLE_COUNT)"
            ),
        ],
        cwd=repository_root / "Platform" / "pipelines",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "64"


def test_reactive_pack_cache_tracks_bev_v3_contract():
    assert workflows.PACK_CACHE_VERSION == "pack-v3-v1-v9-v4"


def test_flyte_entrypoints_do_not_use_mutable_defaults():
    pipelines_root = Path(workflows.__file__).parent
    mutable_defaults = []

    source_paths = [
        pipelines_root / "distributed_training.py",
        pipelines_root / "nuplan_dataset.py",
    ]
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            decorators = set()
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(
                    decorator,
                    ast.Call,
                ) else decorator
                if isinstance(target, ast.Name):
                    decorators.add(target.id)
                elif isinstance(target, ast.Attribute):
                    decorators.add(target.attr)
            if not decorators.intersection(
                {"task", "workflow", "dynamic"},
            ):
                continue

            arguments = node.args.posonlyargs + node.args.args
            defaults = [None] * (
                len(arguments) - len(node.args.defaults)
            ) + list(node.args.defaults)
            for argument, default in zip(arguments, defaults):
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    mutable_defaults.append(
                        (
                            source_path.name,
                            node.name,
                            argument.arg,
                        )
                    )

    assert mutable_defaults == []


def test_reviewed_ray_topologies_have_fixed_worker_groups():
    assert (
        distributed_training.RAY_2.worker_node_config[0].replicas
        == 2
    )
    assert (
        distributed_training.RAY_8.worker_node_config[0].replicas
        == 8
    )
    for config in (
        distributed_training.RAY_2,
        distributed_training.RAY_4,
        distributed_training.RAY_REACTIVE_4,
        distributed_training.RAY_8,
    ):
        workers = config.worker_node_config[0]
        assert workers.min_replicas == workers.replicas
        assert workers.max_replicas == workers.replicas
        assert config.enable_autoscaling is False


def test_four_rank_performance_capacity_matches_ray_contract():
    worker = distributed_training.RAY_REACTIVE_4.worker_node_config[0]
    worker_spec = worker.pod_template.pod_spec
    assert worker.replicas == 4
    assert worker.ray_start_params["num-cpus"] == "3"
    assert worker_spec.node_selector == {
        "workload-type": "gpu-performance"
    }
    assert worker_spec.containers[0].resources.requests == {
        "cpu": "3",
        "memory": "12Gi",
        "nvidia.com/gpu": "1",
    }
    assert (
        distributed_training.train_reactive_stage_ray_4.metadata.labels[
            "kueue.x-k8s.io/queue-name"
        ]
        == "gpu-performance"
    )

    platform_root = Path(distributed_training.__file__).parents[1]
    node_classes = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodeclass.yaml"
            ).read_text()
        )
    }
    reserved_class = node_classes[
        "auto-e2e-gpu-performance-reserved"
    ]["spec"]
    assert reserved_class["capacityReservationSelectorTerms"] == [
        {
            "ownerID": "REPLACE_WITH_AWS_ACCOUNT_ID",
            "tags": {"Name": "auto-e2e-gpu-canary"},
        }
    ]
    assert reserved_class["placementGroupSelector"] == {
        "name": "auto-e2e-distributed-training-pg"
    }
    assert "capacityReservationSelectorTerms" not in node_classes[
        "auto-e2e-gpu-performance-ondemand"
    ]["spec"]
    training_class = node_classes["auto-e2e-gpu-training"]["spec"]
    assert training_class["capacityReservationSelectorTerms"] == [{
        "ownerID": "REPLACE_WITH_AWS_ACCOUNT_ID",
        "tags": {"Name": "auto-e2e-distributed-training"},
    }]

    node_pools = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodepool.yaml"
            ).read_text()
        )
    }
    reserved_pool = node_pools["gpu-performance-reserved"]["spec"]
    assert reserved_pool["weight"] == 100
    assert reserved_pool["limits"] == {
        "cpu": "32",
        "memory": "256Gi",
        "nodes": "4",
        "nvidia.com/gpu": "4",
    }
    assert reserved_pool["template"]["metadata"]["labels"] == {
        "workload-type": "gpu-performance"
    }
    requirements = {
        item["key"]: item["values"]
        for item in reserved_pool["template"]["spec"]["requirements"]
    }
    assert requirements["node.kubernetes.io/instance-type"] == [
        "g6.2xlarge"
    ]
    assert requirements["karpenter.sh/capacity-type"] == ["reserved"]
    training_pool = node_pools["gpu-training"]["spec"]
    assert training_pool["template"]["spec"]["nodeClassRef"]["name"] == (
        "auto-e2e-gpu-training"
    )
    training_requirements = {
        item["key"]: item["values"]
        for item in training_pool["template"]["spec"]["requirements"]
    }
    assert training_requirements["karpenter.sh/capacity-type"] == [
        "reserved"
    ]
    assert training_pool["limits"] == {
        "cpu": "64",
        "memory": "512Gi",
        "nodes": "4",
        "nvidia.com/gpu": "4",
    }

    queue_objects = {
        (
            item["kind"],
            item["metadata"]["name"],
            item["metadata"].get("namespace"),
        ): item
        for item in yaml.safe_load_all(
            (
                platform_root
                / "k8s/kueue-config/kueue-objects.yaml"
            ).read_text()
        )
    }
    performance_queue = queue_objects[
        ("ClusterQueue", "gpu-performance-queue", None)
    ]["spec"]
    gpu_group = next(
        group
        for group in performance_queue["resourceGroups"]
        if group["coveredResources"] == ["nvidia.com/gpu"]
    )
    assert gpu_group["flavors"][0]["resources"] == [
        {"name": "nvidia.com/gpu", "nominalQuota": "4"}
    ]
    assert (
        "LocalQueue",
        "gpu-performance",
        "auto-e2e-development",
    ) in queue_objects

    deploy_script = (platform_root / "infra/post-apply.sh").read_text()
    render_index = deploy_script.index(
        "s/REPLACE_WITH_AWS_ACCOUNT_ID/${ACCOUNT}/g"
    )
    node_pool_index = deploy_script.index(
        "karpenter-nodepools/gpu-nodepool.yaml"
    )
    assert render_index < node_pool_index
    assert "kueue-config/kueue-objects.yaml" in deploy_script
    assert "nodepool/gpu-performance-reserved" in deploy_script
    assert re.search(r"\b[0-9]{12}\b", deploy_script) is None
    for relative_path in re.findall(
        r"\.\./k8s/[A-Za-z0-9_./-]+\.yaml",
        deploy_script,
    ):
        assert (
            platform_root
            / "infra"
            / relative_path
        ).resolve().is_file()


def test_reactive_ray_cpu_contract_has_one_source_of_truth():
    for config in (
        distributed_training.RAY_2,
        distributed_training.RAY_REACTIVE_4,
        distributed_training.RAY_8,
    ):
        worker = config.worker_node_config[0]
        cpu = worker.ray_start_params["num-cpus"]
        resources = (
            worker.pod_template.pod_spec
            .containers[0].resources
        )
        assert resources.requests["cpu"] == cpu
        assert resources.limits["cpu"] == cpu
        assert int(cpu) == distributed_training._reactive_worker_cpus(
            worker.replicas
        )


def test_capacity_and_joint_gates_use_distinct_run_names():
    common = {
        "execution_name": "execution.with.unsupported.characters",
        "stage": "nuplan_full",
        "num_workers": 4,
        "overfit_sample_count": 64,
    }

    capacity = distributed_training._reactive_run_name(
        **common,
        overfit_bev_only=True,
    )
    joint = distributed_training._reactive_run_name(
        **common,
        overfit_bev_only=False,
    )

    assert capacity.endswith("-capacity-overfit-64")
    assert joint.endswith("-joint-overfit-64")
    assert capacity != joint
    assert "." not in capacity
    assert "." not in joint


def test_ray_tasks_serialize_the_resolved_storage_path():
    expected_environment = {
        "AWS_DEFAULT_REGION": "us-west-2",
        "AUTO_E2E_RAY_STORAGE_PATH": (
            distributed_training.RAY_STORAGE_PATH
        ),
        "RAY_TRAIN_V2_ENABLED": "1",
    }

    assert distributed_training.ray_ddp_smoke_4.environment == (
        expected_environment
    )
    assert distributed_training.train_reactive_stage_ray_2.environment == (
        expected_environment
    )
    assert distributed_training.train_reactive_stage_ray_4.environment == (
        expected_environment
    )
    assert distributed_training.train_reactive_stage_ray_8.environment == (
        expected_environment
    )


def test_distributed_program_passes_stage_a_checkpoint_to_stage_b():
    capacity, joint, stage_a, stage_b = (
        distributed_training.wf_train_reactive_nuplan_l2d_ray_8.nodes
    )
    assert capacity.flyte_entity.name.endswith(
        "train_reactive_stage_ray_4"
    )
    assert joint.flyte_entity.name.endswith(
        "train_reactive_stage_ray_4"
    )
    assert stage_a.flyte_entity.name.endswith(
        "train_reactive_stage_ray_8"
    )
    assert stage_b.flyte_entity.name.endswith(
        "train_reactive_stage_ray_8"
    )
    capacity_bindings = {
        binding.var: binding.binding
        for binding in capacity.bindings
    }
    joint_bindings = {
        binding.var: binding.binding
        for binding in joint.bindings
    }
    stage_a_bindings = {
        binding.var: binding.binding for binding in stage_a.bindings
    }
    stage_b_bindings = {
        binding.var: binding.binding for binding in stage_b.bindings
    }
    assert capacity_bindings[
        "overfit_bev_only"
    ].scalar.primitive.boolean
    assert capacity_bindings[
        "overfit_fixed_lr"
    ].scalar.primitive.boolean
    assert (
        capacity_bindings["trajectory_weight"].scalar.primitive.float_value
        == 0.0
    )
    assert capacity_bindings["bev_weight"].scalar.primitive.float_value == 1.0
    assert capacity_bindings["route_weight"].scalar.primitive.float_value == 0.0
    capacity_metadata = joint_bindings["gate_metadata"].promise
    assert capacity_metadata.node_id == capacity.id
    assert capacity_metadata.var == "metadata"
    assert joint_bindings["epochs"].scalar.primitive.integer == 10
    assert joint_bindings["steps_per_epoch"].scalar.primitive.integer == 500
    assert joint_bindings["weight_decay"].scalar.primitive.float_value == 0.0
    assert not joint_bindings[
        "overfit_bev_only"
    ].scalar.primitive.boolean
    assert joint_bindings[
        "overfit_fixed_lr"
    ].scalar.primitive.boolean
    for weight in ("trajectory_weight", "bev_weight", "route_weight"):
        assert joint_bindings[weight].promise.var == weight
    assert stage_a_bindings["stage"].scalar.primitive.string_value == (
        "nuplan_full"
    )
    assert stage_b_bindings["stage"].scalar.primitive.string_value == (
        "l2d_continuation"
    )
    gate_promise = stage_a_bindings["gate_metadata"].promise
    assert gate_promise.node_id == joint.id
    assert gate_promise.var == "metadata"
    assert (
        stage_a_bindings[
            "parent_checkpoint"
        ].scalar.union.value.scalar.none_type
        is not None
    )
    parent_promise = stage_b_bindings["parent_checkpoint"].promise
    assert parent_promise.node_id == stage_a.id
    assert parent_promise.var == "checkpoint"


def test_remote_dataset_inputs_are_required():
    remote = FlyteDirectory("s3://datasets/nuplan/reactive")
    assert distributed_training._flyte_remote_uri(remote) == (
        "s3://datasets/nuplan/reactive"
    )

    with pytest.raises(ValueError, match="immutable S3"):
        distributed_training._flyte_remote_uri(
            FlyteDirectory("/tmp/reactive")
        )


def test_distributed_workflow_source_has_no_deployment_account_id():
    source = Path(distributed_training.__file__).read_text()

    assert re.search(r"\b[0-9]{12}\b", source) is None
    assert "cr-" not in source
    assert "pg-" not in source
    assert "bev_pos_weights" not in source
    assert "bev_pos_weights" not in (
        distributed_training.train_reactive_stage_ray_4
        .python_interface.inputs
    )
    for task in (
        distributed_training.train_reactive_stage_ray_2,
        distributed_training.train_reactive_stage_ray_4,
        distributed_training.train_reactive_stage_ray_8,
    ):
        assert {
            "trajectory_weight",
            "overfit_bev_only",
            "overfit_fixed_lr",
        } <= set(task.python_interface.inputs)


def test_four_rank_training_requires_capacity_then_joint_gate():
    capacity, joint, full = (
        distributed_training.wf_train_reactive_nuplan_ray_4.nodes
    )
    capacity_bindings = {
        binding.var: binding.binding
        for binding in capacity.bindings
    }
    joint_bindings = {
        binding.var: binding.binding
        for binding in joint.bindings
    }
    full_bindings = {
        binding.var: binding.binding
        for binding in full.bindings
    }

    capacity_values = {
        "epochs": 10,
        "overfit_sample_count": 64,
        "steps_per_epoch": 500,
    }
    for name, expected in capacity_values.items():
        assert capacity_bindings[name].scalar.primitive.integer == expected
    assert capacity_bindings[
        "overfit_bev_only"
    ].scalar.primitive.boolean
    assert capacity_bindings[
        "overfit_fixed_lr"
    ].scalar.primitive.boolean
    assert (
        capacity_bindings["weight_decay"].scalar.primitive.float_value
        == 0.0
    )
    assert (
        capacity_bindings["trajectory_weight"].scalar.primitive.float_value
        == 0.0
    )
    assert capacity_bindings["bev_weight"].scalar.primitive.float_value == 1.0
    assert capacity_bindings["route_weight"].scalar.primitive.float_value == 0.0

    capacity_metadata = joint_bindings["gate_metadata"].promise
    assert capacity_metadata.node_id == capacity.id
    assert capacity_metadata.var == "metadata"
    assert joint_bindings[
        "overfit_sample_count"
    ].scalar.primitive.integer == 64
    assert joint_bindings["epochs"].scalar.primitive.integer == 10
    assert joint_bindings["steps_per_epoch"].scalar.primitive.integer == 500
    assert joint_bindings["weight_decay"].scalar.primitive.float_value == 0.0
    assert not joint_bindings[
        "overfit_bev_only"
    ].scalar.primitive.boolean
    assert joint_bindings[
        "overfit_fixed_lr"
    ].scalar.primitive.boolean
    for weight in ("trajectory_weight", "bev_weight", "route_weight"):
        assert joint_bindings[weight].promise.var == weight

    assert (
        full_bindings[
            "overfit_sample_count"
        ].scalar.primitive.integer
        == 0
    )
    assert (
        full_bindings[
            "parent_checkpoint"
        ].scalar.union.value.scalar.none_type
        is not None
    )
    gate_metadata = full_bindings["gate_metadata"].promise
    assert gate_metadata.node_id == joint.id
    assert gate_metadata.var == "metadata"
    assert not full_bindings[
        "overfit_bev_only"
    ].scalar.primitive.boolean
    assert not full_bindings[
        "overfit_fixed_lr"
    ].scalar.primitive.boolean
    for weight in ("trajectory_weight", "bev_weight", "route_weight"):
        assert full_bindings[weight].promise.var == weight


def test_standalone_overfit_workflow_is_bev_capacity_probe():
    node, = distributed_training.wf_overfit_reactive_nuplan_ray_4.nodes
    bindings = {
        binding.var: binding.binding for binding in node.bindings
    }

    assert bindings["epochs"].promise.var == "epochs"
    assert bindings["steps_per_epoch"].scalar.primitive.integer == 500
    assert bindings["weight_decay"].scalar.primitive.float_value == 0.0
    assert bindings["trajectory_weight"].scalar.primitive.float_value == 0.0
    assert bindings["bev_weight"].scalar.primitive.float_value == 1.0
    assert bindings["route_weight"].scalar.primitive.float_value == 0.0
    assert bindings["overfit_bev_only"].scalar.primitive.boolean
    assert bindings["overfit_fixed_lr"].scalar.primitive.boolean


def _bev_overfit_gate_metadata(tmp_path, **overrides):
    metrics = {
        "checkpoint_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "bev_weight": 1.0,
        "corridor_pos_weight": 1.0,
        "executed_optimizer_steps": 5000,
        "overfit_bev_only": False,
        "overfit_fixed_lr": True,
        "overfit_gate_pass": 1,
        "overfit_sample_count": 64,
        "overfit_sample_uid_sha256": "c" * 64,
        "overfit_thresholds_pass": 1,
        "route_weight": 1.0,
        "scheduler_identity": "constant_v1",
        "training_seed": 149,
        "trajectory_weight": 1.0,
        "validation_bev_dynamic_macro_average_precision": 0.95,
        "world_size": 4,
    }
    metrics.update({
        f"{CAMERA_FEATURE_SCALE_WEIGHT_METRIC_PREFIX}{index}": 0.25
        for index in range(4)
    })
    for rank in range(4):
        metrics[
            f"{PEAK_CUDA_ALLOCATED_BYTES_METRIC_PREFIX}{rank}"
        ] = 10_000
        metrics[
            f"{PEAK_CUDA_RESERVED_BYTES_METRIC_PREFIX}{rank}"
        ] = 20_000
    metrics.update({
        f"bev_pos_weight_{index}": float(index + 2)
        for index in range(len(BEV_SEGMENTATION_CLASSES))
    })
    metrics.update({
        f"{OVERFIT_POSITIVE_SAMPLE_SUPPORT_METRIC_PREFIX}{class_name}": 8
        for class_name in BEV_SEGMENTATION_CLASSES
    })
    metrics.update({
        f"validation_bev_{class_name}_{suffix}": value
        for class_name in BEV_SEGMENTATION_CLASSES
        for suffix, value in (
            ("average_precision", 0.95),
            ("positive_cells", 10.0),
            ("recall", 0.95),
        )
    })
    metrics.update({
        f"validation_{BEV_LANE_RANGE_METRIC_PREFIX}"
        f"{range_name}_{suffix}": value
        for range_name in ("near", "far")
        for suffix, value in (
            ("average_precision", 0.95),
            ("positive_cells", 10.0),
            ("precision", 0.95),
            ("recall", 0.95),
        )
    })
    metrics.update(overrides)
    path = tmp_path / "bev-overfit-gate.json"
    path.write_text(json.dumps({
        "history": [dict(metrics)],
        "metrics": metrics,
    }))
    return distributed_training.FlyteFile(str(path))


def _validate_bev_overfit_gate(metadata, **overrides):
    expected = {
        "expected_bev_only": False,
        "expected_trajectory_weight": 1.0,
        "expected_bev_weight": 1.0,
        "expected_route_weight": 1.0,
        "expected_corridor_pos_weight": 1.0,
        "expected_training_seed": 149,
    }
    expected.update(overrides)
    return distributed_training._validated_bev_overfit_gate_dataset(
        metadata,
        **expected,
    )


def test_bev_overfit_gate_validates_final_evidence(tmp_path):
    metadata = _bev_overfit_gate_metadata(tmp_path)

    assert _validate_bev_overfit_gate(metadata) == "b" * 64


def test_bev_overfit_gate_accepts_128_sample_evidence(tmp_path):
    metadata = _bev_overfit_gate_metadata(
        tmp_path,
        overfit_sample_count=128,
    )

    assert _validate_bev_overfit_gate(metadata) == "b" * 64


def test_bev_overfit_gate_accepts_dynamic_camera_feature_stages(tmp_path):
    five_stage_weights = {
        f"{CAMERA_FEATURE_SCALE_WEIGHT_METRIC_PREFIX}{index}": 0.2
        for index in range(5)
    }
    metadata = _bev_overfit_gate_metadata(
        tmp_path,
        **five_stage_weights,
    )

    assert _validate_bev_overfit_gate(metadata) == "b" * 64


@pytest.mark.parametrize(
    ("override_name", "override_value", "match"),
    [
        (
            "validation_bev_lane_boundary_average_precision",
            0.89,
            "average precision",
        ),
        (
            "validation_bev_vehicle_recall",
            0.89,
            "vehicle",
        ),
        (
            "validation_bev_vehicle_positive_cells",
            0.0,
            "no positives",
        ),
    ],
)
def test_bev_overfit_gate_rejects_weak_evidence(
    tmp_path,
    override_name,
    override_value,
    match,
):
    metadata = _bev_overfit_gate_metadata(
        tmp_path,
        **{override_name: override_value},
    )

    with pytest.raises(ValueError, match=match):
        _validate_bev_overfit_gate(metadata)


@pytest.mark.parametrize(
    ("override_name", "override_value", "match"),
    [
        (
            "overfit_positive_sample_support_vehicle",
            7,
            "subset support",
        ),
        (
            "camera_feature_scale_weight_0",
            0.5,
            "do not sum to one",
        ),
        (
            "camera_feature_scale_weight_0",
            "invalid",
            "invalid camera feature scale weights",
        ),
        (
            "peak_cuda_allocated_bytes_rank_2",
            0,
            "CUDA memory evidence",
        ),
        (
            "validation_bev_lane_boundary_far_positive_cells",
            0.0,
            "far lane positives",
        ),
    ],
)
def test_bev_overfit_gate_rejects_missing_diagnostics(
    tmp_path,
    override_name,
    override_value,
    match,
):
    metadata = _bev_overfit_gate_metadata(
        tmp_path,
        **{override_name: override_value},
    )

    with pytest.raises(ValueError, match=match):
        _validate_bev_overfit_gate(metadata)


def test_bev_overfit_gate_rejects_wrong_mode_and_objective(tmp_path):
    capacity_metadata = _bev_overfit_gate_metadata(
        tmp_path,
        overfit_bev_only=True,
        trajectory_weight=0.0,
        route_weight=0.0,
    )
    assert _validate_bev_overfit_gate(
        capacity_metadata,
        expected_bev_only=True,
        expected_trajectory_weight=0.0,
        expected_route_weight=0.0,
    ) == "b" * 64
    with pytest.raises(ValueError, match="wrong mode"):
        _validate_bev_overfit_gate(capacity_metadata)

    joint_metadata = _bev_overfit_gate_metadata(tmp_path)
    with pytest.raises(ValueError, match="wrong bev_weight"):
        _validate_bev_overfit_gate(
            joint_metadata,
            expected_bev_weight=0.5,
        )


def test_bev_overfit_gate_rejects_incomplete_optimizer_budget(tmp_path):
    metadata = _bev_overfit_gate_metadata(
        tmp_path,
        executed_optimizer_steps=MIN_OVERFIT_OPTIMIZER_STEPS - 1,
    )

    with pytest.raises(
        ValueError,
        match=f"fewer than {MIN_OVERFIT_OPTIMIZER_STEPS}",
    ):
        _validate_bev_overfit_gate(metadata)


def test_bev_overfit_gate_rejects_unit_weights_and_history_tampering(
    tmp_path,
):
    unit_weights = {
        f"bev_pos_weight_{index}": 1.0
        for index in range(len(BEV_SEGMENTATION_CLASSES))
    }
    metadata = _bev_overfit_gate_metadata(tmp_path, **unit_weights)
    with pytest.raises(ValueError, match="unit pos weights"):
        _validate_bev_overfit_gate(metadata)

    metadata = _bev_overfit_gate_metadata(tmp_path)
    path = Path(metadata.path)
    payload = json.loads(path.read_text())
    payload["history"][-1]["dataset_manifest_sha256"] = "d" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="history disagrees"):
        _validate_bev_overfit_gate(metadata)

    metadata = _bev_overfit_gate_metadata(tmp_path)
    path = Path(metadata.path)
    payload = json.loads(path.read_text())
    payload["history"][-1]["route_weight"] = 0.5
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="history has wrong route_weight"):
        _validate_bev_overfit_gate(metadata)


def test_four_rank_full_task_rejects_direct_ungated_call():
    with pytest.raises(ValueError, match="requires joint gate"):
        distributed_training.train_reactive_stage_ray_4.task_function(
            shards=[],
            stage="nuplan_full",
        )


def test_eight_rank_stage_a_rejects_direct_ungated_call():
    with pytest.raises(ValueError, match="requires BEV overfit gate"):
        distributed_training.train_reactive_stage_ray_8.task_function(
            shards=[],
            stage="nuplan_full",
        )


def test_canary_launcher_is_idempotent_and_retries_flyte_admin():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-distributed-canary.yml"
    ).read_text()

    assert "remote.fetch_execution(" in buildspec
    assert "remote.sync_execution(" in buildspec
    assert "FlyteEntityAlreadyExistsException" in buildspec
    assert "FlyteEntityNotExistException" in buildspec
    assert "grpc.StatusCode.UNAVAILABLE" in buildspec
    assert "FLYTE_ADMIN_TRANSIENT_RETRY=" in buildspec
    assert "remote.wait(" not in buildspec


def test_nuplan_acquisition_launcher_uses_private_manifest_and_retries_admin():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-nuplan-acquisition.yml"
    ).read_text()

    assert '"source_manifest": FlyteFile(' in buildspec
    assert 'os.environ["SOURCE_MANIFEST_URI"]' in buildspec
    assert '"datasets_bucket": os.environ["DATASETS_BUCKET"]' in buildspec
    assert "remote.fetch_execution(" in buildspec
    assert "remote.sync_execution(" in buildspec
    assert "FlyteEntityAlreadyExistsException" in buildspec
    assert "FlyteEntityNotExistException" in buildspec
    assert "grpc.StatusCode.UNAVAILABLE" in buildspec
    assert 'WAIT_FOR_COMPLETION: "true"' in buildspec
    assert 'os.environ["WAIT_FOR_COMPLETION"] == "true"' in buildspec
    assert "FLYTE_EXECUTION_DETACHED=true" in buildspec
    assert "remote.wait(" not in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None
    workflow_source = Path(nuplan_dataset.__file__).read_text()
    assert "authorized HTTPS source returned" in workflow_source
    assert "authorized HTTPS source connection failed" in workflow_source
    assert "copy_s3_object_multipart(" in workflow_source
    assert "source_s3.get_object(" not in workflow_source


def test_nuplan_acquisition_workflow_binds_one_dynamic_import_program():
    node, = nuplan_dataset.wf_acquire_nuplan_raw_snapshot.nodes

    assert node.flyte_entity.name.endswith(
        "_acquire_nuplan_raw_snapshot"
    )
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }
    assert bindings["source_manifest"].promise.var == "source_manifest"
    assert bindings["datasets_bucket"].promise.var == "datasets_bucket"
    assert bindings["concurrency"].promise.var == "concurrency"


def test_nuplan_snapshot_pack_uses_bev_v3_cache_and_full_default():
    node, = nuplan_dataset.wf_pack_nuplan_snapshot_reactive_dataset.nodes
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith(
        "pack_nuplan_snapshot_reactive_dataset"
    )
    assert node.flyte_entity.metadata.cache_version == (
        "nuplan-snapshot-pack-v5-parallel"
    )
    assert node.flyte_entity.metadata.retries == 1
    assert (
        bindings["limit_total_scenarios"].promise.var
        == "limit_total_scenarios"
    )
    buildspec = (
        Path(nuplan_dataset.__file__).parents[1]
        / "buildspec-launch-nuplan-pack.yml"
    ).read_text()
    assert 'LIMIT_TOTAL_SCENARIOS: "0"' in buildspec
    assert "Platform.pipelines.nuplan_dataset." in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None


def test_nuplan_pack_worker_count_caps_full_and_serializes_limited():
    assert nuplan_dataset._nuplan_pack_worker_count(2, 0) == 2
    assert nuplan_dataset._nuplan_pack_worker_count(20, 0) == 8
    assert nuplan_dataset._nuplan_pack_worker_count(20, 64) == 1


def test_two_rank_canary_wires_both_stages_and_gate():
    nodes = distributed_training.wf_reactive_multistage_ray_2_canary.nodes

    assert len(nodes) == 5
    stage_a = nodes[2]
    stage_b = nodes[3]
    gate = nodes[4]
    stage_b_bindings = {
        binding.var: binding.binding
        for binding in stage_b.bindings
    }
    assert stage_b_bindings["parent_checkpoint"].promise.node_id == (
        stage_a.id
    )
    assert {node.id for node in gate.upstream_nodes} == {
        stage_a.id,
        stage_b.id,
    }


def test_canary_gate_requires_loss_decrease_and_stage_b_bev_off(tmp_path):
    def metadata(history, name):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"history": history}))
        return distributed_training.FlyteFile(str(path))

    common = {
        "train_route_reconstruction": 0.2,
        "train_trajectory": 1.0,
        "validation_ade_6p4s_m": 2.0,
        "validation_selection_score": 0.4,
    }
    stage_a_metrics = {
        **{
            f"bev_pos_weight_{index}": float(index + 2)
            for index in range(len(BEV_SEGMENTATION_CLASSES))
        },
        **{
            f"validation_bev_{class_name}_{suffix}": value
            for class_name in BEV_SEGMENTATION_CLASSES
            for suffix, value in (
                ("average_precision", 0.5),
                ("positive_cells", 10.0),
                ("recall", 0.5),
            )
        },
    }
    stage_a = metadata(
        [
            {
                **common,
                **stage_a_metrics,
                "train_bev_segmentation": 0.5,
                "train_bev_segmentation_bce": 0.6,
                "train_bev_segmentation_dice": 0.4,
                "train_total": 1.7,
            },
            {
                **common,
                **stage_a_metrics,
                "train_bev_segmentation": 0.4,
                "train_bev_segmentation_bce": 0.5,
                "train_bev_segmentation_dice": 0.3,
                "train_total": 1.5,
            },
        ],
        "stage-a",
    )
    stage_b = metadata(
        [
            {
                **common,
                "train_bev_segmentation": 0.0,
                "train_bev_segmentation_bce": 0.0,
                "train_bev_segmentation_dice": 0.0,
                "train_total": 1.2,
            },
            {
                **common,
                "train_bev_segmentation": 0.0,
                "train_bev_segmentation_bce": 0.0,
                "train_bev_segmentation_dice": 0.0,
                "train_total": 1.1,
            },
        ],
        "stage-b",
    )

    report = (
        distributed_training.verify_reactive_canary_training.task_function(
            stage_a_metadata=stage_a,
            stage_b_metadata=stage_b,
        )
    )

    assert json.loads(Path(report.path).read_text())["thresholds_pass"]
