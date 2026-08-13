import torch
import torch.nn as nn


class FusedFeaturePooling(nn.Module):
    """Pool Fused image and map BEV feature maps into a feature vector
    which will be consumed by the trajectory planner.
    """

    output_dim = 75 * 50

    def __init__(self, embed_dim=256):
        super(FusedFeaturePooling, self).__init__()
        self.embed_dim = int(embed_dim)

        # Reduce image/map BEV features from [batch, C, H, W] to
        # [batch, 1, H, W] to capture most salient feature activation at each BEV
        # grid location
        self.reduce_channels = nn.Conv2d(self.embed_dim, 1, 3, 1, 1)

        # Reduce image/map BEV spatial resolution to a coarse grid of size
        # 75 x 50 such that the resulting tensor has dimensions [batch, C, 75, 50]
        self.reduce_bev = nn.AdaptiveAvgPool2d((75, 50))


    def forward(self, fused_features):
        # fused_features: [B, C, H, W]
        if fused_features.ndim != 4:
            raise ValueError(
                "fused_features must have shape [B,C,H,W], "
                f"got {tuple(fused_features.shape)}"
            )
        if fused_features.shape[1] != self.embed_dim:
            raise ValueError(
                f"fused_features must have {self.embed_dim} channels, "
                f"got {fused_features.shape[1]}"
            )

        # Reduce fused BEV features to [batch, 1, 75, 50]
        features_channel_compressed = self.reduce_channels(fused_features)
        features_bev_compressed = self.reduce_bev(features_channel_compressed)

        # Flatten reduced BEV features to a vector of dim [batch x 3750]
        reduced_features = torch.flatten(features_bev_compressed, start_dim=1)

        return reduced_features
