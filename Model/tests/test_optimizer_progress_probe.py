"""The optimizer-progress check must not hinge on one arbitrary parameter.

`train_il` verifies the optimizer actually moved the planner. It used to clone
whichever parameter `named_parameters()` yielded first for TrajectoryPlanner —
which is `BezierPlanner.visual_history_proj.weight`, a parameter that is
provably immovable under the default configuration. A healthy run then raised
at the very end, after every epoch had been paid for.

These tests pin the three facts that combine into that failure, so a future
refactor that reintroduces single-parameter probing fails here instead of after
a night of training.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from model_components.trajectory_planning.bezier_planner import BezierPlanner


def test_zero_init_weight_with_zero_input_never_moves():
    """The mechanism: no gradient, and AdamW's decay scales rather than shifts.

    This is why probing a single zero-init parameter is unsound — not a
    hypothetical, this is exactly what `visual_history_proj.weight` does when
    the World Model is off.
    """
    layer = nn.Linear(8, 8)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    before = layer.weight.detach().clone()
    # AdamW with decoupled weight decay, as train_il configures it.
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4, weight_decay=1e-2)

    for _ in range(50):
        optimizer.zero_grad()
        # All-zero input, which is what the packed shards carry for
        # visual_history when world_model=False.
        layer(torch.zeros(4, 8)).sum().backward()
        optimizer.step()

    assert torch.equal(layer.weight.detach(), before)
    assert float(layer.weight.detach().norm()) == 0.0


def test_a_parameter_that_receives_gradient_does_move():
    """Control: the same optimizer moves a parameter that is actually fed."""
    layer = nn.Linear(8, 8)
    nn.init.zeros_(layer.weight)
    before = layer.weight.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4, weight_decay=1e-2)

    for _ in range(50):
        optimizer.zero_grad()
        layer(torch.randn(4, 8)).sum().backward()
        optimizer.step()

    assert not torch.equal(layer.weight.detach(), before)


def test_bezier_planner_zero_inits_the_visual_history_projection():
    """The premise holds in the real model, not just in the toy above."""
    planner = BezierPlanner()

    assert float(planner.visual_history_proj.weight.detach().abs().max()) == 0.0
    assert float(planner.visual_history_proj.bias.detach().abs().max()) == 0.0


def test_visual_history_projection_is_the_first_planner_parameter():
    """Guards the ordering that made a single-parameter probe pick it.

    If a refactor moves a gradient-receiving parameter back in front of it, the
    old probe would start passing again by luck rather than by design. This test
    failing is not a defect — it means the coincidence is gone and this file's
    reason for existing should be re-read.
    """
    planner = BezierPlanner()
    first_name = next(
        name for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    )

    assert first_name.startswith("visual_history_proj")


def test_train_il_probes_more_than_one_planner_parameter():
    """The fix itself: the probe collects every trainable planner parameter."""
    pytest.importorskip("flytekit")
    from Platform.pipelines import workflows

    source = inspect.getsource(workflows.train_il.task_function)
    probe_start = source.index("optimizer_probe")
    probe = source[probe_start:probe_start + 400]

    assert "TrajectoryPlanner" in probe, "probe no longer scopes to the planner"
    # The distinguishing property: the probe binds a COLLECTION, not the single
    # element `next()` pulls off the generator. `next(` is what the bug was.
    assert "= next(" not in probe, (
        "optimizer progress is being probed on a single parameter again — "
        "a zero-init planner parameter fed zero input can never move, so this "
        "fails a healthy run at the end of training"
    )
    assert "optimizer_probe = [" in probe
