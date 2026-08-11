"""Flyte wiring for distributed Reactive Stage A and Stage B."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("flytekit")

from flytekit.types.directory import FlyteDirectory

from Platform.pipelines import distributed_training, workflows


def test_flyte_entrypoints_do_not_use_mutable_defaults():
    pipelines_root = Path(workflows.__file__).parent
    mutable_defaults = []

    for source_path in sorted(pipelines_root.glob("*.py")):
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
        distributed_training.RAY_8,
    ):
        workers = config.worker_node_config[0]
        assert workers.min_replicas == workers.replicas
        assert workers.max_replicas == workers.replicas
        assert config.enable_autoscaling is False


def test_two_rank_canary_targets_dedicated_gpu_placement_pool():
    canary_spec = (
        distributed_training.RAY_2.worker_node_config[0]
        .pod_template.pod_spec
    )
    assert canary_spec.node_selector == {
        "workload-type": "gpu-canary"
    }
    assert [
        (item.key, item.operator, item.effect)
        for item in canary_spec.tolerations
    ] == [("nvidia.com/gpu", "Exists", "NoSchedule")]
    canary_container = canary_spec.containers[0]
    assert canary_container.resources.requests == {
        "cpu": "3",
        "memory": "12Gi",
        "nvidia.com/gpu": "1",
    }
    assert canary_container.resources.limits == {
        "cpu": "3",
        "memory": "12Gi",
        "nvidia.com/gpu": "1",
    }
    assert canary_spec.volumes[0].empty_dir.size_limit == "4Gi"
    assert (
        distributed_training.RAY_2.worker_node_config[0]
        .ray_start_params["num-cpus"]
        == "3"
    )

    for config in (
        distributed_training.RAY_4,
        distributed_training.RAY_8,
    ):
        worker_spec = (
            config.worker_node_config[0].pod_template.pod_spec
        )
        assert worker_spec.node_selector == {
            "workload-type": "gpu-training"
        }
        worker_container = worker_spec.containers[0]
        assert worker_container.resources.requests == {
            "cpu": "4",
            "memory": "16Gi",
            "nvidia.com/gpu": "1",
        }
        assert worker_spec.volumes[0].empty_dir.size_limit == "8Gi"

    assert (
        distributed_training.train_reactive_stage_ray_2.metadata.labels[
            "kueue.x-k8s.io/queue-name"
        ]
        == "gpu-canary"
    )
    assert (
        distributed_training.train_reactive_stage_ray_8.metadata.labels[
            "kueue.x-k8s.io/queue-name"
        ]
        == "training"
    )


def test_gpu_canary_infrastructure_is_bounded_and_placement_backed():
    platform_root = Path(distributed_training.__file__).parents[1]
    node_classes = {
        document["metadata"]["name"]: document
        for document in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodeclass.yaml"
            ).read_text()
        )
    }
    canary_class = node_classes["auto-e2e-gpu-canary"]["spec"]
    assert "capacityReservationSelectorTerms" not in canary_class
    assert canary_class["placementGroupSelector"] == {
        "name": "auto-e2e-distributed-training-pg"
    }
    assert canary_class["subnetSelectorTerms"] == [
        {
            "tags": {
                "Name": "auto-e2e-platform-private-us-west-2a"
            }
        }
    ]

    node_pools = {
        document["metadata"]["name"]: document
        for document in yaml.safe_load_all(
            (
                platform_root
                / "k8s/karpenter-nodepools/gpu-nodepool.yaml"
            ).read_text()
        )
    }
    canary_pool = node_pools["gpu-canary"]["spec"]
    assert canary_pool["limits"] == {
        "cpu": "16",
        "memory": "128Gi",
        "nodes": "2",
        "nvidia.com/gpu": "2",
    }
    requirements = {
        item["key"]: item["values"]
        for item in canary_pool["template"]["spec"]["requirements"]
    }
    assert requirements["node.kubernetes.io/instance-type"] == [
        "g6.xlarge",
        "g6.2xlarge",
        "g6e.xlarge",
        "g6e.2xlarge",
        "g7.2xlarge",
        "g7e.2xlarge",
    ]
    assert not {
        "g6f.xlarge",
        "g6f.2xlarge",
    }.intersection(requirements["node.kubernetes.io/instance-type"])
    assert requirements["karpenter.sh/capacity-type"] == ["on-demand"]
    assert requirements["topology.kubernetes.io/zone"] == [
        "us-west-2a"
    ]


def test_gpu_canary_has_an_isolated_kueue_flavor_and_quota():
    platform_root = Path(distributed_training.__file__).parents[1]
    objects = list(
        yaml.safe_load_all(
            (
                platform_root / "k8s/kueue-config/kueue-objects.yaml"
            ).read_text()
        )
    )
    by_kind_and_name = {
        (item["kind"], item["metadata"]["name"]): item
        for item in objects
    }

    flavor = by_kind_and_name[
        ("ResourceFlavor", "gpu-canary-flavor")
    ]["spec"]
    assert flavor["nodeLabels"] == {"workload-type": "gpu-canary"}

    cluster_queue = by_kind_and_name[
        ("ClusterQueue", "gpu-canary-queue")
    ]["spec"]
    gpu_group = next(
        group
        for group in cluster_queue["resourceGroups"]
        if group["coveredResources"] == ["nvidia.com/gpu"]
    )
    assert gpu_group["flavors"] == [
        {
            "name": "gpu-canary-flavor",
            "resources": [
                {"name": "nvidia.com/gpu", "nominalQuota": "2"}
            ],
        }
    ]

    local_queue = by_kind_and_name[
        ("LocalQueue", "gpu-canary")
    ]
    assert local_queue["metadata"]["namespace"] == (
        "auto-e2e-development"
    )
    assert local_queue["spec"]["clusterQueue"] == "gpu-canary-queue"


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
    assert distributed_training.train_reactive_stage_ray_8.environment == (
        expected_environment
    )


def test_distributed_program_passes_stage_a_checkpoint_to_stage_b():
    stage_a, stage_b = (
        distributed_training.wf_train_reactive_nuplan_l2d_ray_8.nodes
    )
    assert stage_a.flyte_entity.name.endswith(
        "train_reactive_stage_ray_8"
    )
    assert stage_b.flyte_entity.name.endswith(
        "train_reactive_stage_ray_8"
    )
    stage_a_bindings = {
        binding.var: binding.binding
        for binding in stage_a.bindings
    }
    stage_b_bindings = {
        binding.var: binding.binding
        for binding in stage_b.bindings
    }
    assert stage_a_bindings["stage"].scalar.primitive.string_value == (
        "nuplan_full"
    )
    assert stage_b_bindings["stage"].scalar.primitive.string_value == (
        "l2d_continuation"
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


def test_l2d_mini_launcher_uses_explicit_range_and_remote_osm():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-l2d-reactive-mini.yml"
    ).read_text()

    assert '"source_pbf": FlyteFile(os.environ["SOURCE_PBF_URI"])' in (
        buildspec
    )
    assert '"start_ep": int(os.environ["START_EP"])' in buildspec
    assert '"end_ep": int(os.environ["END_EP"])' in buildspec
    assert "remote.fetch_execution(" in buildspec
    assert "remote.sync_execution(" in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None


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
    workflow_source = Path(workflows.__file__).read_text()
    assert "authorized HTTPS source returned" in workflow_source
    assert "authorized HTTPS source connection failed" in workflow_source
    assert "copy_s3_object_multipart(" in workflow_source
    assert "source_s3.get_object(" not in workflow_source


def test_nuplan_acquisition_workflow_binds_one_dynamic_import_program():
    node, = workflows.wf_acquire_nuplan_raw_snapshot.nodes

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


def test_nuplan_snapshot_pack_workflow_binds_materialization_contract():
    node, = workflows.wf_pack_nuplan_snapshot_reactive_dataset.nodes

    assert node.flyte_entity.name.endswith(
        "pack_nuplan_snapshot_reactive_dataset"
    )
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }
    assert (
        bindings["snapshot_manifest"].promise.var
        == "snapshot_manifest"
    )
    assert bindings["datasets_bucket"].promise.var == "datasets_bucket"
    assert bindings["archive_ids"].promise.var == "archive_ids"
    assert (
        bindings["limit_total_scenarios"].promise.var
        == "limit_total_scenarios"
    )


def test_nuplan_snapshot_pack_launcher_is_detached_and_bounded():
    buildspec = (
        Path(distributed_training.__file__).parents[1]
        / "buildspec-launch-nuplan-pack.yml"
    ).read_text()

    assert (
        "wf_pack_nuplan_snapshot_reactive_dataset"
        in buildspec
    )
    assert '"snapshot_manifest": FlyteFile(' in buildspec
    assert '"archive_ids": archive_ids or None' in buildspec
    assert 'WAIT_FOR_COMPLETION: "false"' in buildspec
    assert "FLYTE_EXECUTION_DETACHED=true" in buildspec
    assert "MAX_REJECTION_FRACTION" in buildspec
    assert re.search(r"\b[0-9]{12}\b", buildspec) is None
    workflow_source = Path(workflows.__file__).read_text()
    assert 'ephemeral_storage="500Gi"' in workflow_source
    assert 'cache_version="nuplan-snapshot-pack-v1"' in workflow_source


def test_l2d_reactive_pack_workflow_binds_osm_and_target_contract():
    node, = workflows.wf_pack_l2d_reactive_dataset.nodes
    assert node.flyte_entity.name.endswith("wf_create_dataset_sharded")
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }
    assert bindings["dataset"].scalar.primitive.string_value == (
        workflows.Dataset.L2D.value
    )
    assert bindings[
        "reactive_targets"
    ].scalar.primitive.boolean is True
    assert bindings["osm_graph_snapshot"].promise.var == (
        "osm_graph_snapshot"
    )
    assert bindings["start_ep"].promise.var == "start_ep"
    assert bindings["end_ep"].promise.var == "end_ep"


def test_l2d_osm_builder_workflow_is_one_offline_task():
    node, = workflows.wf_build_l2d_osm_graph_artifact.nodes

    assert node.flyte_entity.name.endswith(
        "build_l2d_osm_graph_artifact"
    )
    assert node.bindings[0].binding.promise is not None


def test_l2d_reactive_prepare_workflow_passes_built_osm_to_pack():
    build_osm, pack = workflows.wf_prepare_l2d_reactive_dataset.nodes

    assert build_osm.flyte_entity.name.endswith(
        "build_l2d_osm_graph_artifact"
    )
    assert pack.flyte_entity.name.endswith(
        "wf_pack_l2d_reactive_dataset"
    )
    bindings = {
        binding.var: binding.binding
        for binding in pack.bindings
    }
    assert bindings["osm_graph_snapshot"].promise.node_id == build_osm.id
    assert bindings["start_ep"].promise.var == "start_ep"
    assert bindings["end_ep"].promise.var == "end_ep"


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
    }
    stage_a = metadata(
        [
            {
                **common,
                "train_bev_segmentation": 0.5,
                "train_total": 1.7,
            },
            {
                **common,
                "train_bev_segmentation": 0.4,
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
                "train_total": 1.2,
            },
            {
                **common,
                "train_bev_segmentation": 0.0,
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
