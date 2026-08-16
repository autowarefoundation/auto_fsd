import abc
from typing import Dict, Any
import torch
import numpy as np

try:
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree
except ImportError:
    Point, Polygon, STRtree = None, None, None

class BaseReward(abc.ABC):
    """Abstract base class for all reward functions in the AutoE2E RL loop."""

    @abc.abstractmethod
    def compute(
        self,
        ego_pose: tuple[float, float, float] | None = None,
        trajectory_xy: torch.Tensor | np.ndarray | None = None,
        headings: torch.Tensor | np.ndarray | None = None,
        navigation_map: Any = None,
        speed: float = 0.0,
        acceleration: float = 0.0,
        yaw_rate: float = 0.0,
        info: Dict[str, Any] | None = None,
        reasoning_faithfulness_gate: float = 0.0,
        **kwargs: Any,
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
            info: Extra simulation / agent info dictionary.
            reasoning_faithfulness_gate: The 'g' causal coupling scalar [0, 1].
            **kwargs: Extra keyword arguments.

        Returns:
            Scalar reward.
        """
        pass

class SafetyReward(BaseReward):
    """Handcrafted penalty for collisions, off-road driving, and TTC violations (R_safety)."""
    
    def __init__(self) -> None:
        self._cached_map_version = None
        self._drivable_tree = None

    def compute(
        self,
        ego_pose: tuple[float, float, float] | None = None,
        trajectory_xy: torch.Tensor | np.ndarray | None = None,
        headings: torch.Tensor | np.ndarray | None = None,
        navigation_map: Any = None,
        speed: float = 0.0,
        acceleration: float = 0.0,
        yaw_rate: float = 0.0,
        info: Dict[str, Any] | None = None,
        reasoning_faithfulness_gate: float = 0.0,
        **kwargs: Any,
    ) -> float:
        if Point is None:
            raise ImportError("shapely is required for SafetyReward polygon intersection checks.")
            
        # 1. Extract required inputs
        ego_pose = ego_pose if ego_pose is not None else kwargs.get("ego_pose")
        trajectory_xy = trajectory_xy if trajectory_xy is not None else kwargs.get("trajectory_xy")
        headings = headings if headings is not None else kwargs.get("headings")
        nav_map = navigation_map if navigation_map is not None else kwargs.get("navigation_map")
        info_dict = info if info is not None else kwargs.get("info", {})
        
        if not ego_pose or trajectory_xy is None or not nav_map or headings is None:
            raise ValueError("ego_pose, trajectory_xy, headings or nav_map was not passed for reward computation")
            
        # 2. Build or retrieve the spatial index for the drivable area polygons
        if self._cached_map_version != nav_map.map_version or self._drivable_tree is None:
            polygons = []
            for poly_primitive in nav_map.drivable_polygons:
                pts = poly_primitive.points_enu_m[:, :2] # Take (X, Y)
                if len(pts) >= 3:
                    polygons.append(Polygon(pts))
            self._drivable_tree = STRtree(polygons) if polygons else None
            self._cached_map_version = nav_map.map_version
            
        if self._drivable_tree is None:
            raise ValueError("No drivable area defined")
            
        # 3. Transform trajectory from ego-centric to map frame (ENU)
        if isinstance(trajectory_xy, torch.Tensor):
            traj_np = trajectory_xy.detach().cpu().numpy()
        else:
            traj_np = np.asarray(trajectory_xy)
            
        if isinstance(headings, torch.Tensor):
            headings_np = headings.detach().cpu().numpy()
        else:
            headings_np = np.asarray(headings)
            
        if len(traj_np) == 0:
            return 0.0

        c, s = np.cos(ego_pose[2]), np.sin(ego_pose[2])
        rot_mat = np.array([[c, -s], [s, c]])
        traj_global = (traj_np @ rot_mat.T) + np.array([ego_pose[0], ego_pose[1]])
        headings_global = headings_np + ego_pose[2]
        
        off_road_penalty = 0.0
        ttc_penalty = 0.0
        
        dt = 0.1  # 10Hz planning rate
        
        # Ego bounding box half-dimensions (approx. 4.7m x 2.0m)
        ego_half_l, ego_half_w = 2.35, 1.0
        
        dynamic_agents = info_dict.get("dynamic_agents", [])
        
        for i, ((x, y), yaw) in enumerate(zip(traj_global, headings_global)):
            pt = Point(x, y)
            
            # --- Off-road check ---
            possible_matches = self._drivable_tree.query(pt)
            if not possible_matches.size:
                 off_road_penalty -= 1.0
            else:
                is_on_road = False
                for idx in possible_matches:
                    if self._drivable_tree.geometries[idx].covers(pt):
                        is_on_road = True
                        break
                if not is_on_road:
                    off_road_penalty -= 1.0
                    
            '''
            --- TTC & Collision check ---

            This part is temporary and will be dependent on how the bounding boxes are passed.
            
            It will likely have to be modified before deployment
            
            '''
            if dynamic_agents:
                # Construct ego bounding box polygon at timestep i
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                dx_l, dy_l = ego_half_l * cos_y, ego_half_l * sin_y
                dx_w, dy_w = -ego_half_w * sin_y, ego_half_w * cos_y
                ego_poly = Polygon([
                    (x + dx_l + dx_w, y + dy_l + dy_w),
                    (x + dx_l - dx_w, y + dy_l - dy_w),
                    (x - dx_l - dx_w, y - dy_l - dy_w),
                    (x - dx_l + dx_w, y - dy_l + dy_w)
                ])
                
                t_sec = i * dt
                
                for agent in dynamic_agents:
                    ax, ay = agent["position"]
                    vx, vy = agent.get("velocity", (0.0, 0.0))
                    al, aw = agent.get("bbox_size", (4.0, 1.8)) # This is just an assumed car size, might be possible to handle this better later
                    a_yaw = agent.get("yaw", 0.0)
                    
                    # Project agent position to time t_sec
                    proj_ax = ax + vx * t_sec
                    proj_ay = ay + vy * t_sec
                    
                    # Construct agent bounding box polygon
                    cos_a, sin_a = np.cos(a_yaw), np.sin(a_yaw)
                    hl, hw = al / 2.0, aw / 2.0
                    adx_l, ady_l = hl * cos_a, hl * sin_a
                    adx_w, ady_w = -hw * sin_a, hw * cos_a
                    
                    agent_poly = Polygon([
                        (proj_ax + adx_l + adx_w, proj_ay + ady_l + ady_w),
                        (proj_ax + adx_l - adx_w, proj_ay + ady_l - ady_w),
                        (proj_ax - adx_l - adx_w, proj_ay - ady_l - ady_w),
                        (proj_ax - adx_l + adx_w, proj_ay - ady_l + ady_w)
                    ])
                    
                    if ego_poly.intersects(agent_poly):
                        # Collision at time t_sec!
                        if t_sec < 2.0:
                            ttc_penalty -= 5.0 # Massive penalty for immediate collision
                        else:
                            ttc_penalty -= (5.0 / t_sec) # Scales inversely with time
                        break # Only penalize once per timestep for collisions
                
        num_steps = len(traj_global)
        if num_steps > 0:
            avg_off_road = off_road_penalty / num_steps
            avg_ttc = ttc_penalty / num_steps
            return float(avg_off_road + avg_ttc)
        return 0.0

class ProgressReward(BaseReward):
    """Reward for route progress, treated as a binding term to prevent stalling (R_progress)."""
    def compute(
        self,
        ego_pose: tuple[float, float, float] | None = None,
        trajectory_xy: torch.Tensor | np.ndarray | None = None,
        headings: torch.Tensor | np.ndarray | None = None,
        navigation_map: Any = None,
        speed: float = 0.0,
        acceleration: float = 0.0,
        yaw_rate: float = 0.0,
        info: Dict[str, Any] | None = None,
        reasoning_faithfulness_gate: float = 0.0,
        **kwargs: Any,
    ) -> float:
        # TODO: Implement longitudinal progress along the route
        return 0.0

class ComfortReward(BaseReward):
    """Negative penalty for jerk and high lateral acceleration (R_comfort)."""
    def compute(
        self,
        ego_pose: tuple[float, float, float] | None = None,
        trajectory_xy: torch.Tensor | np.ndarray | None = None,
        headings: torch.Tensor | np.ndarray | None = None,
        navigation_map: Any = None,
        speed: float = 0.0,
        acceleration: float = 0.0,
        yaw_rate: float = 0.0,
        info: Dict[str, Any] | None = None,
        reasoning_faithfulness_gate: float = 0.0,
        **kwargs: Any,
    ) -> float:
        # TODO: Implement jerk / lateral acc penalty
        return 0.0

class ReasoningReward(BaseReward):
    """Reasoning-shaped reward gated by the faithfulness scalar (R_reason)."""
    def compute(
        self,
        ego_pose: tuple[float, float, float] | None = None,
        trajectory_xy: torch.Tensor | np.ndarray | None = None,
        headings: torch.Tensor | np.ndarray | None = None,
        navigation_map: Any = None,
        speed: float = 0.0,
        acceleration: float = 0.0,
        yaw_rate: float = 0.0,
        info: Dict[str, Any] | None = None,
        reasoning_faithfulness_gate: float = 0.0,
        **kwargs: Any,
    ) -> float:
        # Extract the faithfulness gate directly and apply it to the logic
        g = reasoning_faithfulness_gate if reasoning_faithfulness_gate != 0.0 else kwargs.get("reasoning_faithfulness_gate", 0.0)
        # TODO: Implement reasoning compliance
        reason_reward = 0.0
        return float(g * reason_reward)

class ImitationAnchor(BaseReward):
    """Imitation learning regularization anchor (D(pi || pi_IL))."""
    def compute(
        self,
        ego_pose: tuple[float, float, float] | None = None,
        trajectory_xy: torch.Tensor | np.ndarray | None = None,
        headings: torch.Tensor | np.ndarray | None = None,
        navigation_map: Any = None,
        speed: float = 0.0,
        acceleration: float = 0.0,
        yaw_rate: float = 0.0,
        info: Dict[str, Any] | None = None,
        reasoning_faithfulness_gate: float = 0.0,
        **kwargs: Any,
    ) -> float:
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
