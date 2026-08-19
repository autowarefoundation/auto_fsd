"""Action-conditioned JEPA is wired through AutoE2E / train_il, not just the API."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from evaluation.action_jepa_ab import ACTION_DIM, run_action_jepa_ab


def test_autoe2e_teacher_forces_actions_into_jepa(build_mock_model):
    """trajectory_target must reach predict_future when action_dim is set."""
    model = build_mock_model(
        num_views=6, device=torch.device("cpu"),
        enable_world_model=True,
        world_model_kwargs={"feature_channels": 768, "action_dim": ACTION_DIM},
    )
    b, v = 2, 6
    visual = torch.randn(b, v, 3, 256, 256)
    mp = torch.randn(b, 3, 256, 256)
    vh = torch.zeros(b, 896)
    ego = torch.randn(b, 256)
    target = torch.randn(b, ACTION_DIM)
    hist = torch.randn(b, 4, v, 3, 256, 256)
    fut = torch.randn(b, 4, v, 3, 256, 256)

    _, aux = model(
        visual, mp, vh, ego, mode="train", trajectory_target=target,
        history_frames=hist, future_frames=fut,
    )
    jepa = model.World_Action_Model_E2E.jepa_loss(
        aux["future_state_pred"], aux["future_frames"])
    jepa.backward()
    grads = [
        p.grad for n, p in model.World_Action_Model_E2E.named_parameters()
        if "action_proj" in n and p.grad is not None
    ]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_default_combined_does_not_build_action_proj(build_mock_model):
    model = build_mock_model(
        num_views=6, device=torch.device("cpu"), enable_world_model=True,
    )
    assert model.World_Action_Model_E2E.action_dim is None
    assert not hasattr(model.World_Action_Model_E2E.future_predictor, "action_proj")


def test_action_jepa_ab_reports_containment_then_sensitivity(build_mock_model):
    torch.manual_seed(0)
    report = run_action_jepa_ab(build_mock_model, steps=4, seed=0)
    # Zero-init: swapping actions at init must not change the forecast.
    assert report["action"]["pred_l1_init_matched_vs_shuffled"] < 1e-5
    assert abs(
        report["action"]["jepa_init_matched"] - report["action"]["jepa_init_shuffled"]
    ) < 1e-5
    # After a few steps the projection must leave zero and the forecast
    # must become action-sensitive.
    assert report["action"]["action_proj_l2"] > 0
    assert report["action"]["pred_l1_trained_matched_vs_shuffled"] > 1e-5
    assert report["history"]["jepa"] > 0
    assert report["action"]["jepa"] > 0


def test_train_il_forwards_action_dim_when_flag_set():
    pytest.importorskip("flytekit")
    from Platform.pipelines import workflows

    params = inspect.signature(workflows.train_il.task_function).parameters
    assert params["action_conditioned_jepa"].default is False
    source = inspect.getsource(workflows.train_il.task_function)
    tree = ast.parse(source)
    call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AutoE2E"
        ):
            call = node
            break
    assert call is not None
    keys = {kw.arg for kw in call.keywords}
    assert "world_model_kwargs" in keys
    assert "action_conditioned_jepa=action_conditioned_jepa" in inspect.getsource(
        workflows.wf_train_il
    )
