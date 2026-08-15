import abc
from typing import Dict, Any
import torch

class BaseReward(abc.ABC):
    """Abstract base class for all reward functions in the AutoE2E RL loop."""

    @abc.abstractmethod
    def compute(
        self,
        ego_pose: tuple[float, float, float],
        trajectory_xy: torch.Tensor,
        headings: torch.Tensor,
        navigation_map: Any,
        speed: float,
        acceleration: float,
        yaw_rate: float,
        reasoning_faithfulness_gate: float = 0.0,
    ) -> float:
        """Compute the scalar reward component.

        Args:
            ego_pose: (x, y, yaw) in map frame.
            trajectory_xy: Predicted trajectory [T, 2].
            headings: Predicted headings [T].
            navigation_map: C++ NativeNavigationRasterizer map object or similar context.
            speed: Ego vehicle speed (m/s).
            acceleration: Ego vehicle acceleration (m/s^2).
            yaw_rate: Ego vehicle yaw rate (rad/s).
            reasoning_faithfulness_gate: The 'g' causal coupling scalar [0, 1].

        Returns:
            Scalar reward.
        """
        pass

class SafetyReward(BaseReward):
    """Handcrafted penalty for collisions, off-road driving, and TTC violations (R_safety)."""
    def compute(self, **kwargs) -> float:
        # TODO: Implement safety penalties using DRIVABLE_AREA intersection
        return 0.0

class ProgressReward(BaseReward):
    """Reward for route progress, treated as a binding term to prevent stalling (R_progress)."""
    def compute(self, **kwargs) -> float:
        # TODO: Implement longitudinal progress along the route
        return 0.0

class ComfortReward(BaseReward):
    """Negative penalty for jerk and high lateral acceleration (R_comfort)."""
    def compute(self, **kwargs) -> float:
        # TODO: Implement jerk / lateral acc penalty
        return 0.0

class ReasoningReward(BaseReward):
    """Reasoning-shaped reward gated by the faithfulness scalar (R_reason)."""
    def compute(self, **kwargs) -> float:
        # Extract the faithfulness gate directly and apply it to the logic
        g = kwargs.get("reasoning_faithfulness_gate", 0.0)
        # TODO: Implement reasoning compliance
        reason_reward = 0.0
        return g * reason_reward

class ImitationAnchor(BaseReward):
    """Imitation learning regularization anchor (D(pi || pi_IL))."""
    def compute(self, **kwargs) -> float:
        # TODO: Implement KL divergence / MSE anchor against the IL expert policy
        # Note: the subtraction is handled by configuring the weight as a negative number (-lambda_il).
        return 0.0

class RewardRegistry:
    """Manages a collection of BaseReward functions and their scaling weights."""

    def __init__(self, config_weights: Dict[str, float]) -> None:
        """Initialize the registry with specific weights.
        
        Args:
            config_weights: A dictionary mapping reward names to their weights.
                e.g., {'w_safe': 1.0, 'w_prog': 1.0, 'w_comf': 0.1, 'w_reason': 0.5, 'lambda_il': -0.05}
        """
        self.weights = config_weights
        
        # Instantiate the active rewards
        self.rewards: Dict[str, BaseReward] = {}
        if "w_safe" in self.weights:
            self.rewards["w_safe"] = SafetyReward()
        if "w_prog" in self.weights:
            self.rewards["w_prog"] = ProgressReward()
        if "w_comf" in self.weights:
            self.rewards["w_comf"] = ComfortReward()
        if "w_reason" in self.weights:
            self.rewards["w_reason"] = ReasoningReward()
        if "lambda_il" in self.weights:
            self.rewards["lambda_il"] = ImitationAnchor()

    def compute_total_reward(self, **kwargs) -> tuple[float, Dict[str, float]]:
        """Compute the weighted sum of all registered rewards.
        
        Returns:
            A tuple containing:
                - The total scalar reward.
                - A dictionary of the unweighted individual components.
        """
        components: Dict[str, float] = {}
        total_reward = 0.0
        
        for name, reward_func in self.rewards.items():
            val = reward_func.compute(**kwargs)
            components[name] = val
            total_reward += self.weights[name] * val
            
        return total_reward, components
