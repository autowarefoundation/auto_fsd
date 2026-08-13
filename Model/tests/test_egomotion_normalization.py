"""Contracts for egomotion feature normalization."""

import pytest
import torch

from model_components.auto_e2e import normalize_egomotion_history


def _scale(device):
    return torch.tensor(
        [33.0, 8.0, 1.0, 1.0],
        device=device,
    )


def test_normalization_is_non_mutating_for_2d_input(device):
    values = torch.tensor(
        [[33.0, 8.0, 2.0, 3.0] * 2],
        device=device,
    )
    source = values.clone()

    normalized = normalize_egomotion_history(values)

    expected = source / _scale(device).repeat(2)
    assert torch.allclose(normalized, expected)
    assert torch.equal(values, source)
    assert normalized.data_ptr() != values.data_ptr()


def test_normalization_supports_3d_input_and_gradients(device):
    values = torch.randn(
        2,
        5,
        8,
        device=device,
        requires_grad=True,
    )

    normalized = normalize_egomotion_history(values)
    normalized.square().mean().backward()

    assert normalized.shape == values.shape
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(
    "values,error_type,error_message",
    [
        (torch.ones(8, dtype=torch.int64), TypeError, "floating-point"),
        (torch.ones(8), ValueError, "shape"),
        (torch.ones(2, 6), ValueError, "divisible by 4"),
    ],
)
def test_normalization_rejects_invalid_inputs(
    values,
    error_type,
    error_message,
):
    with pytest.raises(error_type, match=error_message):
        normalize_egomotion_history(values)
