"""Contracts for the Bezier fused-feature pooling boundary."""

import pytest
import torch

from model_components.fused_feature_pooling import FusedFeaturePooling


@pytest.mark.parametrize("height,width", [(8, 8), (45, 30), (450, 300)])
def test_pooling_produces_fixed_size_vector(device, height, width):
    pooling = FusedFeaturePooling(embed_dim=256).to(device)
    features = torch.randn(2, 256, height, width, device=device)

    pooled = pooling(features)

    assert pooled.shape == (2, 75 * 50)
    assert torch.isfinite(pooled).all()


def test_pooling_rejects_invalid_rank(device):
    pooling = FusedFeaturePooling(embed_dim=256).to(device)
    with pytest.raises(ValueError, match="\\[B,C,H,W\\]"):
        pooling(torch.randn(2, 256, 64, device=device))


def test_pooling_rejects_invalid_channel_count(device):
    pooling = FusedFeaturePooling(embed_dim=256).to(device)
    with pytest.raises(ValueError, match="must have 256 channels"):
        pooling(torch.randn(2, 128, 8, 8, device=device))
