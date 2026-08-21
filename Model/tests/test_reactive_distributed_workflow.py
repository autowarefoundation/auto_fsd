"""Flyte wiring for distributed Reactive Stage A and Stage B."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("flytekit")

from flytekit.types.directory import FlyteDirectory

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_CLASSES,
)
from Platform.pipelines import (
    distributed_training,
    nuplan_dataset,
    workflows,
)


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
        distributed_training.RAY_8,
    ):
        workers = config.worker_node_config[0]
        assert workers.min_replicas == workers.replicas
        assert workers.max_replicas == workers.replicas
        assert config.enable_autoscaling is False


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
    assert "bev_pos_weights" not in source
    assert "bev_pos_weights" not in (
        distributed_training.train_reactive_stage_ray_4
        .python_interface.inputs
    )


def test_four_rank_full_training_depends_on_gate_but_not_its_weights():
    overfit, full = (
        distributed_training.wf_train_reactive_nuplan_ray_4.nodes
    )
    overfit_bindings = {
        binding.var: binding.binding
        for binding in overfit.bindings
    }
    full_bindings = {
        binding.var: binding.binding
        for binding in full.bindings
    }

    assert (
        overfit_bindings[
            "overfit_sample_count"
        ].scalar.primitive.integer
        == 64
    )
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
    assert gate_metadata.node_id == overfit.id
    assert gate_metadata.var == "metadata"


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


def test_nuplan_snapshot_pack_uses_bev_v2_cache_and_full_default():
    node, = nuplan_dataset.wf_pack_nuplan_snapshot_reactive_dataset.nodes
    bindings = {
        binding.var: binding.binding
        for binding in node.bindings
    }

    assert node.flyte_entity.name.endswith(
        "pack_nuplan_snapshot_reactive_dataset"
    )
    assert node.flyte_entity.metadata.cache_version == (
        "nuplan-snapshot-pack-v2"
    )
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
                "train_total": 1.7,
            },
            {
                **common,
                **stage_a_metrics,
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
