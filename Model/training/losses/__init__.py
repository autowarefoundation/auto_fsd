"""Training loss modules for AutoE2E (kept outside the model per Zain's criterion)."""

from .horizon_reasoning_loss import HorizonReasoningLoss
from .control_rollout import (
    ROLLOUT_POLICY_VERSION,
    integrate_controls_torch,
)
from .rollout_aligned_loss import (
    ROLLOUT_ALIGNED_LOSS_VERSION,
    RolloutAlignedLoss,
)
from .route_consistency_loss import (
    RouteConsistencyLoss,
    RouteConsistencyWeights,
    ego_points_to_grid,
)
from .fidelity_aware_reward import (
    FIDELITY_AWARE_REWARD_VERSION,
    FidelityAwareRewardResult,
    V1_WEIGHTS,
    consequence_alignment_reward,
    fidelity_aware_reward,
    soft_advantage_from_reward,
    v1_handcrafted_reward,
    world_model_fidelity,
)

__all__ = [
    "FIDELITY_AWARE_REWARD_VERSION",
    "FidelityAwareRewardResult",
    "HorizonReasoningLoss",
    "ROLLOUT_ALIGNED_LOSS_VERSION",
    "ROLLOUT_POLICY_VERSION",
    "RouteConsistencyLoss",
    "RouteConsistencyWeights",
    "RolloutAlignedLoss",
    "V1_WEIGHTS",
    "consequence_alignment_reward",
    "ego_points_to_grid",
    "fidelity_aware_reward",
    "integrate_controls_torch",
    "soft_advantage_from_reward",
    "v1_handcrafted_reward",
    "world_model_fidelity",
]
