"""Unit tests for the fidelity-aware reward prototype (#123)."""

from __future__ import annotations

import torch

from training.losses.fidelity_aware_reward import (
    FIDELITY_AWARE_REWARD_VERSION,
    consequence_alignment_reward,
    fidelity_aware_reward,
    soft_advantage_from_reward,
    world_model_fidelity,
)


def test_version_pin() -> None:
    assert FIDELITY_AWARE_REWARD_VERSION == "v1_safety_comfort_progress_gated_wm"


def test_perfect_wm_fidelity_is_one() -> None:
    x = torch.randn(4, 8, 8)
    fid = world_model_fidelity(x, x.clone())
    assert torch.allclose(fid, torch.ones_like(fid))


def test_error_lowers_fidelity() -> None:
    pred = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    fid = world_model_fidelity(pred, target, temperature=1.0)
    assert torch.all(fid < 1.0)
    assert torch.all(fid > 0.0)


def test_fidelity_does_not_zero_handcrafted_base() -> None:
    """#123: a broken WM must not wipe safety/comfort/progress."""
    base = torch.tensor([1.0, 1.0])
    wm_pred = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    wm_tgt = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    out = fidelity_aware_reward(
        base,
        wm_prediction=wm_pred,
        wm_target=wm_tgt,
        fidelity_temperature=1.0,
    )
    assert torch.allclose(out.reward, base)
    assert out.fidelity[0].item() == 1.0
    assert out.fidelity[1].item() < out.fidelity[0].item()


def test_consequence_term_requires_external_preference() -> None:
    base = torch.zeros(2)
    wm = torch.zeros(2, 3)
    pred_fut = torch.zeros(2, 3)
    pref = torch.ones(2, 3)
    out = fidelity_aware_reward(
        base,
        wm_prediction=wm,
        wm_target=wm.clone(),
        predicted_future=pred_fut,
        preferred_future=pref,
        consequence_weight=1.0,
        consequence_scale=1.0,
    )
    # Perfect WM fidelity=1, consequence = -mse = -1 → reward -1.
    assert torch.allclose(out.reward, torch.tensor([-1.0, -1.0]))
    assert out.consequence_reward is not None


def test_consequence_alignment_reward_shapes() -> None:
    r = consequence_alignment_reward(torch.zeros(3, 2), torch.ones(3, 2), scale=2.0)
    assert r.shape == (3,)
    assert torch.all(r < 0)


def test_soft_advantage_zero_mean_without_baseline() -> None:
    reward = torch.tensor([1.0, 3.0])
    adv = soft_advantage_from_reward(reward)
    assert abs(adv.mean().item()) < 1e-6


def test_expert_outranks_jerky_on_v1_base() -> None:
    from evaluation.fidelity_reward_experiment import run_fidelity_reward_experiment

    report = run_fidelity_reward_experiment()
    assert report["base"]["expert_minus_jerky"] > 0
    assert report["noise_wm"]["ranking_matches_base"] is True
    assert report["noise_wm"]["fidelity_mean"] < 0.05
    assert report["faithful_wm"]["fidelity_mean"] > 0.99

