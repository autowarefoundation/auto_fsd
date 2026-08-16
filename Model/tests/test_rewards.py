"""Unit tests for SafetyReward and the AutoE2E reward framework.

Covers:
- Off-road penalty logic: inside, outside, partially off-road, multi-polygon,
  map version caching/invalidation, and coordinate transformations.
- Time-to-Collision (TTC) & dynamic agent collision logic:
  - No collision / distant agents / parallel lane traffic.
  - Immediate collision at t < 2.0s (constant penalty -5.0).
  - Delayed collision at t >= 2.0s (time-decayed penalty -5.0 / t).
  - Collision duration spanning multiple timesteps.
  - Linear kinematics projection (position + velocity * t).
  - Agent yaw orientation and custom bounding box sizes.
  - Single penalty per timestep with multiple overlapping agents.
- Combined off-road + TTC penalties.
- Input validation, torch tensor formats, and edge cases.
- RewardRegistry and auxiliary reward class interfaces.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import torch

_ALPASIM_DRIVER_DIR = Path(__file__).resolve().parents[1] / "plugins" / "alpasim_driver"
if str(_ALPASIM_DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(_ALPASIM_DRIVER_DIR))

from alpasim_autoe2e.rewards import (  # noqa: E402
    ComfortReward,
    ImitationAnchor,
    ProgressReward,
    ReasoningReward,
    RewardRegistry,
    SafetyReward,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Primitives
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MockPolygonPrimitive:
    """Mock polygon primitive adhering to navigation polygon contract."""

    primitive_id: str
    points_enu_m: np.ndarray


@dataclasses.dataclass
class MockNavigationMap:
    """Mock navigation map object holding drivable area polygon primitives."""

    map_version: str
    drivable_polygons: List[MockPolygonPrimitive]


@pytest.fixture
def drivable_corridor_map() -> MockNavigationMap:
    """Drivable corridor along the x-axis: x in [-100, 100], y in [-5, 5]."""
    # Counter-clockwise rectangle: [x, y, z]
    pts = np.array(
        [
            [-100.0, -5.0, 0.0],
            [100.0, -5.0, 0.0],
            [100.0, 5.0, 0.0],
            [-100.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    poly = MockPolygonPrimitive(primitive_id="lane_0", points_enu_m=pts)
    return MockNavigationMap(map_version="v1.0", drivable_polygons=[poly])


@pytest.fixture
def multi_polygon_map() -> MockNavigationMap:
    """Map with two disjoint drivable polygons separated by a 10m off-road gap.

    - Polygon A: x in [-50, -5], y in [-5, 5]
    - Gap (off-road): x in (-5, 5)
    - Polygon B: x in [5, 50], y in [-5, 5]
    """
    poly_a = MockPolygonPrimitive(
        primitive_id="poly_a",
        points_enu_m=np.array(
            [[-50.0, -5.0], [-5.0, -5.0], [-5.0, 5.0], [-50.0, 5.0]],
            dtype=np.float64,
        ),
    )
    poly_b = MockPolygonPrimitive(
        primitive_id="poly_b",
        points_enu_m=np.array(
            [[5.0, -5.0], [50.0, -5.0], [50.0, 5.0], [5.0, 5.0]],
            dtype=np.float64,
        ),
    )
    return MockNavigationMap(map_version="v1.0", drivable_polygons=[poly_a, poly_b])


@pytest.fixture
def straight_trajectory_10_steps() -> tuple[torch.Tensor, torch.Tensor]:
    """10-step straight trajectory along local x-axis from x=1 to x=10 with heading 0."""
    traj = torch.stack(
        [torch.arange(1.0, 11.0, dtype=torch.float32), torch.zeros(10, dtype=torch.float32)],
        dim=-1,
    )
    headings = torch.zeros(10, dtype=torch.float32)
    return traj, headings


# ---------------------------------------------------------------------------
# 1. Off-Road Penalty Tests
# ---------------------------------------------------------------------------


class TestSafetyRewardOffRoad:
    """Tests covering off-road detection and polygon intersection logic."""

    def test_trajectory_fully_inside_drivable_area(
        self, drivable_corridor_map: MockNavigationMap, straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor]
    ):
        """When all waypoints lie inside drivable polygons, off-road penalty is 0.0."""
        traj, headings = straight_trajectory_10_steps
        reward_fn = SafetyReward()

        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": []},
        )

        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_trajectory_fully_outside_drivable_area(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """When all 10 waypoints lie outside drivable polygons (y=20m, corridor y in [-5, 5]),
        each step is penalized -1.0, yielding an average penalty of -1.0.
        """
        # 10 steps along y=20.0 (corridor only extends to y=5.0)
        traj = torch.stack(
            [torch.arange(1.0, 11.0, dtype=torch.float32), torch.full((10,), 20.0, dtype=torch.float32)],
            dim=-1,
        )
        headings = torch.zeros(10, dtype=torch.float32)
        reward_fn = SafetyReward()

        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": []},
        )

        # 10 steps * (-1.0) / 10 steps = -1.0
        assert reward == pytest.approx(-1.0, abs=1e-6)

    def test_trajectory_partially_outside_drivable_area(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Trajectory with 6 points inside and 4 points outside the drivable area.
        Average off-road penalty should equal -4.0 / 10 = -0.4.
        """
        # First 6 points inside corridor (y=0.0), last 4 points off-road (y=15.0)
        x_pts = torch.arange(1.0, 11.0, dtype=torch.float32)
        y_pts = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0], dtype=torch.float32)
        traj = torch.stack([x_pts, y_pts], dim=-1)
        headings = torch.zeros(10, dtype=torch.float32)

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": []},
        )

        assert reward == pytest.approx(-0.4, abs=1e-6)

    def test_off_road_with_ego_pose_rotation_and_translation(self):
        """Verify ego_pose (x, y, yaw) transforms ego-frame trajectory to map frame.

        Ego is at (0, 0, pi/2) facing North (+y).
        Ego-frame trajectory moving forward along local x: [1, 2, 3, 4, 5]
        transforms to global y: [1, 2, 3, 4, 5], global x: [0, 0, 0, 0, 0].
        """
        # Vertical drivable corridor along global y-axis: x in [-2, 2], y in [-10, 10]
        corridor_pts = np.array([[-2.0, -10.0], [2.0, -10.0], [2.0, 10.0], [-2.0, 10.0]], dtype=np.float64)
        nav_map = MockNavigationMap(
            map_version="v1.0",
            drivable_polygons=[MockPolygonPrimitive("north_lane", corridor_pts)],
        )

        # Local trajectory going forward in ego x
        traj_local = torch.stack([torch.arange(1.0, 6.0), torch.zeros(5)], dim=-1)
        headings_local = torch.zeros(5)

        reward_fn = SafetyReward()

        # Case A: Facing North (yaw = pi/2), local forward moves into global y -> inside corridor
        reward_inside = reward_fn.compute(
            ego_pose=(0.0, 0.0, np.pi / 2),
            trajectory_xy=traj_local,
            headings=headings_local,
            navigation_map=nav_map,
        )
        assert reward_inside == pytest.approx(0.0, abs=1e-6)

        # Case B: Facing East (yaw = 0), local forward moves into global x -> outside corridor for x > 2.0
        # For points x=1,2,3,4,5: x=1 is inside (<=2), x=2 is on boundary/outside, x=3,4,5 are outside
        reward_outside = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_local,
            headings=headings_local,
            navigation_map=nav_map,
        )
        assert reward_outside < 0.0

    def test_multiple_drivable_polygons_and_gap(self, multi_polygon_map: MockNavigationMap):
        """Trajectory traversing from polygon A, across an off-road gap, into polygon B.

        Poly A: x in [-50, -5]
        Gap: x in (-5, 5) -> off-road
        Poly B: x in [5, 50]
        """
        # 5 points at x = [-10.0, -7.0, 0.0, 7.0, 10.0], y = 0.0
        # -10 and -7 are in Poly A
        # 0 is in the gap (off-road -> penalty -1.0)
        # 7 and 10 are in Poly B
        traj = torch.tensor([[-10.0, 0.0], [-7.0, 0.0], [0.0, 0.0], [7.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
        headings = torch.zeros(5, dtype=torch.float32)

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=multi_polygon_map,
        )

        # 1 step off-road out of 5 steps = -1.0 / 5 = -0.2
        assert reward == pytest.approx(-0.2, abs=1e-6)

    def test_2d_and_3d_polygon_coordinates(self):
        """Navigation maps with 2D [N, 2] or 3D [N, 3] points_enu_m are handled correctly."""
        pts_3d = np.array([[-20.0, -5.0, 1.5], [20.0, -5.0, 1.5], [20.0, 5.0, 2.0], [-20.0, 5.0, 2.0]])
        nav_map = MockNavigationMap(
            map_version="v3d",
            drivable_polygons=[MockPolygonPrimitive("poly_3d", pts_3d)],
        )
        traj = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=torch.float32)
        headings = torch.zeros(2, dtype=torch.float32)

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=nav_map,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_polygons_with_fewer_than_3_points_ignored(self):
        """Polygons with fewer than 3 vertices are skipped, while valid ones are indexed."""
        invalid_poly = MockPolygonPrimitive("line_primitive", np.array([[0.0, 0.0], [1.0, 1.0]]))
        valid_poly = MockPolygonPrimitive(
            "triangle_primitive",
            np.array([[-10.0, -10.0], [10.0, -10.0], [0.0, 10.0]]),
        )
        nav_map = MockNavigationMap(
            map_version="v_mixed",
            drivable_polygons=[invalid_poly, valid_poly],
        )

        reward_fn = SafetyReward()
        # Point (0, 0) is strictly inside triangle (-10,-10), (10,-10), (0,10)
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=torch.tensor([[0.0, 0.0]]),
            headings=torch.zeros(1),
            navigation_map=nav_map,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_spatial_index_caching_and_invalidation(self, drivable_corridor_map: MockNavigationMap):
        """STRtree is cached across compute calls with the same map_version and rebuilt on version change."""
        reward_fn = SafetyReward()
        traj = torch.tensor([[1.0, 0.0]])
        headings = torch.zeros(1)

        assert reward_fn._drivable_tree is None
        assert reward_fn._cached_map_version is None

        # First compute builds the STRtree
        reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
        )
        tree_v1 = reward_fn._drivable_tree
        assert tree_v1 is not None
        assert reward_fn._cached_map_version == "v1.0"

        # Second compute with same map_version reuses the exact same tree instance
        reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
        )
        assert reward_fn._drivable_tree is tree_v1

        # Third compute with new map_version rebuilds the tree
        updated_map = MockNavigationMap(
            map_version="v2.0",
            drivable_polygons=drivable_corridor_map.drivable_polygons,
        )
        reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=updated_map,
        )
        assert reward_fn._cached_map_version == "v2.0"
        assert reward_fn._drivable_tree is not tree_v1


# ---------------------------------------------------------------------------
# 2. Time-to-Collision (TTC) & Dynamic Agents Tests
# ---------------------------------------------------------------------------


class TestSafetyRewardTTCAndCollisions:
    """Tests covering TTC calculation, linear kinematics, and collision penalty scaling."""

    def test_no_dynamic_agents(
        self, drivable_corridor_map: MockNavigationMap, straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor]
    ):
        """When no dynamic agents are in info, TTC penalty is 0.0."""
        traj, headings = straight_trajectory_10_steps
        reward_fn = SafetyReward()

        reward_empty = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": []},
        )
        reward_missing_key = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={},
        )

        assert reward_empty == pytest.approx(0.0, abs=1e-6)
        assert reward_missing_key == pytest.approx(0.0, abs=1e-6)

    def test_distant_agent_no_collision(
        self, drivable_corridor_map: MockNavigationMap, straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor]
    ):
        """Distant agent far from ego trajectory yields 0.0 TTC penalty."""
        traj, headings = straight_trajectory_10_steps
        reward_fn = SafetyReward()

        info = {
            "dynamic_agents": [
                {
                    "position": (500.0, 500.0),
                    "velocity": (0.0, 0.0),
                    "bbox_size": (4.0, 1.8),
                    "yaw": 0.0,
                }
            ]
        }

        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info=info,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_parallel_vehicle_no_collision(
        self, drivable_corridor_map: MockNavigationMap, straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor]
    ):
        """Vehicle traveling parallel in adjacent lane (lateral offset y=4.0m) does not collide."""
        traj, headings = straight_trajectory_10_steps  # ego y=0.0, width=2.0 -> half_w=1.0 (bounds y in [-1, 1])
        reward_fn = SafetyReward()

        # Agent at y=4.0, width=1.8 -> half_w=0.9 (bounds y in [3.1, 4.9]), no lateral overlap
        info = {
            "dynamic_agents": [
                {
                    "position": (5.0, 4.0),
                    "velocity": (1.0, 0.0),
                    "bbox_size": (4.0, 1.8),
                    "yaw": 0.0,
                }
            ]
        }

        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info=info,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_immediate_collision_t_less_than_2s(self, drivable_corridor_map: MockNavigationMap):
        """Collision at timestep i=5 where t_sec = 5 * 0.1 = 0.5s (< 2.0s).

        Penalty per colliding step is fixed at -5.0.
        For a 10-step trajectory with 1 colliding step, total reward is -5.0 / 10 = -0.5.
        """
        # Ego trajectory: 10 steps, each step advances by 1.0m (from x=1 to x=10)
        traj = torch.stack([torch.arange(1.0, 11.0), torch.zeros(10)], dim=-1)
        headings = torch.zeros(10)

        # Place stationary agent at x=6.0 (ego at step i=5 is at x=6.0)
        # Ego bounding box at step 5: x in [6.0 - 2.35, 6.0 + 2.35] = [3.65, 8.35]
        # Agent bounding box: x in [6.0 - 2.0, 6.0 + 2.0] = [4.0, 8.0] -> Collides at step 5!
        # Step i=5 corresponds to t_sec = 0.5s < 2.0s -> penalty = -5.0
        # To avoid collisions at other steps, use a small agent bbox
        agent = {
            "position": (6.0, 0.0),
            "velocity": (0.0, 0.0),
            "bbox_size": (0.5, 0.5),
            "yaw": 0.0,
        }

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent]},
        )

        # Collision occurs at steps where ego_poly intersects agent at (6.0, 0.0).
        # Ego center at step 3: x=4.0, front is 4.0 + 2.35 = 6.35 >= 5.75 -> collision at step 3, 4, 5, 6, 7.
        # All these steps have t_sec = i * 0.1 <= 0.7s < 2.0s. Each gets -5.0.
        assert reward < 0.0

        # Now test single isolated collision step by setting trajectory where only index 5 is near agent:
        traj_single = torch.zeros((10, 2))
        traj_single[:, 1] = 0.0
        traj_single[:, 0] = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 100.0])
        # Agent placed at (0, 0)
        agent_single = {
            "position": (0.0, 0.0),
            "velocity": (0.0, 0.0),
            "bbox_size": (1.0, 1.0),
            "yaw": 0.0,
        }

        reward_single = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_single,
            headings=torch.zeros(10),
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent_single]},
        )
        # At i=5: t_sec = 0.5 < 2.0 -> ttc_penalty = -5.0. Total reward = -5.0 / 10 = -0.5
        assert reward_single == pytest.approx(-0.5, abs=1e-6)

    def test_delayed_collision_t_greater_equal_2s(self, drivable_corridor_map: MockNavigationMap):
        """Collision at timestep i=25 where t_sec = 25 * 0.1 = 2.5s (>= 2.0s).

        Penalty per colliding step scales inversely with time: -(5.0 / t_sec) = -(5.0 / 2.5) = -2.0.
        For a 50-step trajectory with 1 colliding step, total reward is -2.0 / 50 = -0.04.
        """
        num_steps = 50
        # Ego trajectory where only step i=25 is at (0, 0), others far away inside corridor (x=50, y=0)
        traj = torch.zeros((num_steps, 2))
        traj[:, 0] = 50.0
        traj[25] = torch.tensor([0.0, 0.0])
        headings = torch.zeros(num_steps)

        agent = {
            "position": (0.0, 0.0),
            "velocity": (0.0, 0.0),
            "bbox_size": (1.0, 1.0),
            "yaw": 0.0,
        }

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent]},
        )

        # At i=25, t_sec = 2.5s >= 2.0s -> penalty = -(5.0 / 2.5) = -2.0
        # Average reward = -2.0 / 50 = -0.04
        assert reward == pytest.approx(-2.0 / 50, abs=1e-6)

    def test_ttc_penalty_monotonic_decay_with_time(self, drivable_corridor_map: MockNavigationMap):
        """Verify delayed collision penalty is strictly weaker as time-to-collision increases."""
        reward_fn = SafetyReward()
        num_steps = 100

        agent = {
            "position": (0.0, 0.0),
            "velocity": (0.0, 0.0),
            "bbox_size": (1.0, 1.0),
            "yaw": 0.0,
        }

        penalties = []
        # Test collision at t=2.0s (i=20), t=4.0s (i=40), and t=8.0s (i=80)
        for target_i in [20, 40, 80]:
            traj = torch.zeros((num_steps, 2))
            traj[:, 0] = 50.0
            traj[target_i] = torch.tensor([0.0, 0.0])
            r = reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                trajectory_xy=traj,
                headings=torch.zeros(num_steps),
                navigation_map=drivable_corridor_map,
                info={"dynamic_agents": [agent]},
            )
            penalties.append(r)

        # Penalties: at t=2.0 -> -2.5/100, at t=4.0 -> -1.25/100, at t=8.0 -> -0.625/100
        assert penalties[0] == pytest.approx(-2.5 / num_steps, abs=1e-6)
        assert penalties[1] == pytest.approx(-1.25 / num_steps, abs=1e-6)
        assert penalties[2] == pytest.approx(-0.625 / num_steps, abs=1e-6)
        # Monotonic decay in penalty magnitude (less negative)
        assert penalties[0] < penalties[1] < penalties[2] < 0.0

    def test_multi_timestep_collision_spanning_under_and_over_2s(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Collision spanning timesteps across the 2.0s threshold.

        Step i=10 (t=1.0s < 2.0s): penalty = -5.0
        Step i=25 (t=2.5s >= 2.0s): penalty = -5.0 / 2.5 = -2.0
        Total penalty = -7.0 for 50 steps -> avg = -0.14.
        """
        num_steps = 50
        traj = torch.zeros((num_steps, 2))
        traj[:, 0] = 50.0
        traj[10] = torch.tensor([0.0, 0.0])
        traj[25] = torch.tensor([0.0, 0.0])
        headings = torch.zeros(num_steps)

        agent = {
            "position": (0.0, 0.0),
            "velocity": (0.0, 0.0),
            "bbox_size": (1.0, 1.0),
            "yaw": 0.0,
        }

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent]},
        )

        expected_penalty = (-5.0 + -2.0) / num_steps
        assert reward == pytest.approx(expected_penalty, abs=1e-6)

    def test_dynamic_agent_linear_kinematics(self, drivable_corridor_map: MockNavigationMap):
        """Agent moving towards ego trajectory: proj_pos = position + velocity * t_sec.

        Agent initially at (20.0, 0.0) with vx = -5.0 m/s.
        At t = 2.0s (i = 20), agent projects to x = 20.0 + (-5.0 * 2.0) = 10.0m.
        Ego is at x = 10.0m at i = 20 -> Collision occurs at predicted interception!
        """
        num_steps = 30
        traj = torch.zeros((num_steps, 2))
        traj[:, 0] = 50.0
        traj[20] = torch.tensor([10.0, 0.0])
        headings = torch.zeros(num_steps)

        agent = {
            "position": (20.0, 0.0),
            "velocity": (-5.0, 0.0),
            "bbox_size": (1.0, 1.0),
            "yaw": 0.0,
        }

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent]},
        )

        # At t=2.0s: penalty = -(5.0 / 2.0) = -2.5. Total = -2.5 / 30
        assert reward == pytest.approx(-2.5 / num_steps, abs=1e-6)

    def test_multiple_overlapping_agents_single_penalty_per_timestep(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """When multiple agents intersect ego at the same timestep, only one penalty is assessed."""
        num_steps = 10
        traj = torch.zeros((num_steps, 2))
        traj[:, 0] = 50.0
        traj[5] = torch.tensor([0.0, 0.0])
        headings = torch.zeros(num_steps)

        agent1 = {"position": (0.0, 0.0), "velocity": (0.0, 0.0), "bbox_size": (2.0, 2.0), "yaw": 0.0}
        agent2 = {"position": (0.0, 0.0), "velocity": (0.0, 0.0), "bbox_size": (2.0, 2.0), "yaw": 0.0}

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent1, agent2]},
        )

        # Single penalty of -5.0 assessed for timestep 5 (t=0.5s), average = -0.5
        assert reward == pytest.approx(-0.5, abs=1e-6)

    def test_agent_yaw_and_custom_bbox_size(self, drivable_corridor_map: MockNavigationMap):
        """Custom agent dimensions and rotated yaw correctly orient the bounding box polygon."""
        num_steps = 10
        traj = torch.zeros((num_steps, 2))
        traj[:, 0] = 50.0
        # Ego at (0, 3.5), length 4.7 (half_l 2.35), width 2.0 (half_w 1.0) -> y in [2.5, 4.5]
        traj[5] = torch.tensor([0.0, 3.5])
        headings = torch.zeros(num_steps)

        # Truck placed at (0, 0), length 8.0, width 2.0.
        # If yaw = 0: truck extends x in [-4, 4], y in [-1, 1] -> does NOT intersect ego at y=3.5.
        # If yaw = pi/2: truck extends x in [-1, 1], y in [-4, 4] -> INTERSECTS ego at y=3.5!
        truck_aligned = {"position": (0.0, 0.0), "velocity": (0.0, 0.0), "bbox_size": (8.0, 2.0), "yaw": 0.0}
        truck_rotated = {"position": (0.0, 0.0), "velocity": (0.0, 0.0), "bbox_size": (8.0, 2.0), "yaw": np.pi / 2}

        reward_fn = SafetyReward()

        reward_aligned = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [truck_aligned]},
        )
        reward_rotated = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [truck_rotated]},
        )

        assert reward_aligned == pytest.approx(0.0, abs=1e-6)
        assert reward_rotated == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Combined Penalties & Edge Cases
# ---------------------------------------------------------------------------


class TestSafetyRewardCombinedAndEdgeCases:
    """Tests for combined violations, tensor types, missing parameters, and edge conditions."""

    def test_combined_off_road_and_collision_penalties(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Trajectory that has both off-road waypoints and an agent collision.

        10 steps:
        - 5 steps off-road (y=20.0) -> off_road_penalty = -5.0
        - 1 step collision at i=2 (t=0.2s < 2.0s) -> ttc_penalty = -5.0
        - Total reward = (-5.0 + -5.0) / 10 = -1.0
        """
        traj = torch.zeros((10, 2))
        # Place inside steps far from agent (x=50, y=0)
        traj[:, 0] = 50.0
        # Ego at step 2 is at (0, 0)
        traj[2] = torch.tensor([0.0, 0.0])
        # Steps 5..9 off-road (y=20, x=0)
        traj[5:, 0] = 0.0
        traj[5:, 1] = 20.0
        headings = torch.zeros(10)

        agent = {"position": (0.0, 0.0), "velocity": (0.0, 0.0), "bbox_size": (1.0, 1.0), "yaw": 0.0}

        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            info={"dynamic_agents": [agent]},
        )

        assert reward == pytest.approx(-1.0, abs=1e-6)

    def test_torch_tensor_with_grad_and_numpy_inputs(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Handles torch.Tensor (with requires_grad), numpy.ndarray, and list inputs seamlessly."""
        reward_fn = SafetyReward()

        # 1. PyTorch Tensor with gradient tracking
        traj_torch = torch.tensor([[1.0, 0.0], [2.0, 0.0]], requires_grad=True)
        headings_torch = torch.tensor([0.0, 0.0], requires_grad=True)

        r1 = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_torch,
            headings=headings_torch,
            navigation_map=drivable_corridor_map,
        )
        assert isinstance(r1, float)
        assert r1 == pytest.approx(0.0, abs=1e-6)

        # 2. Numpy ndarray
        traj_np = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        headings_np = np.array([0.0, 0.0], dtype=np.float32)
        r2 = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj_np,
            headings=headings_np,
            navigation_map=drivable_corridor_map,
        )
        assert r2 == pytest.approx(0.0, abs=1e-6)

        # 3. Python lists
        r3 = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=[[1.0, 0.0], [2.0, 0.0]],
            headings=[0.0, 0.0],
            navigation_map=drivable_corridor_map,
        )
        assert r3 == pytest.approx(0.0, abs=1e-6)

    def test_empty_trajectory_returns_zero(self, drivable_corridor_map: MockNavigationMap):
        """Zero-step trajectory returns 0.0 scalar without division by zero errors."""
        reward_fn = SafetyReward()
        reward = reward_fn.compute(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=torch.zeros((0, 2)),
            headings=torch.zeros((0,)),
            navigation_map=drivable_corridor_map,
        )
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_missing_required_kwargs_raises_value_error(
        self, drivable_corridor_map: MockNavigationMap
    ):
        """Missing ego_pose, trajectory_xy, headings, or navigation_map raises ValueError."""
        reward_fn = SafetyReward()
        traj = torch.tensor([[1.0, 0.0]])
        headings = torch.zeros(1)

        with pytest.raises(ValueError, match="ego_pose, trajectory_xy, headings or nav_map was not passed"):
            reward_fn.compute(
                trajectory_xy=traj,
                headings=headings,
                navigation_map=drivable_corridor_map,
            )

        with pytest.raises(ValueError, match="ego_pose, trajectory_xy, headings or nav_map was not passed"):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                headings=headings,
                navigation_map=drivable_corridor_map,
            )

        with pytest.raises(ValueError, match="ego_pose, trajectory_xy, headings or nav_map was not passed"):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                trajectory_xy=traj,
                navigation_map=drivable_corridor_map,
            )

        with pytest.raises(ValueError, match="ego_pose, trajectory_xy, headings or nav_map was not passed"):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                trajectory_xy=traj,
                headings=headings,
            )

    def test_empty_drivable_area_raises_value_error(self):
        """Navigation map with empty drivable polygons raises ValueError."""
        empty_map = MockNavigationMap(map_version="empty", drivable_polygons=[])
        reward_fn = SafetyReward()

        with pytest.raises(ValueError, match="No drivable area defined"):
            reward_fn.compute(
                ego_pose=(0.0, 0.0, 0.0),
                trajectory_xy=torch.tensor([[1.0, 0.0]]),
                headings=torch.zeros(1),
                navigation_map=empty_map,
            )

    def test_shapely_missing_raises_import_error(self, drivable_corridor_map: MockNavigationMap):
        """If shapely is unavailable, SafetyReward raises an informative ImportError."""
        reward_fn = SafetyReward()
        with patch("alpasim_autoe2e.rewards.Point", None):
            with pytest.raises(ImportError, match="shapely is required for SafetyReward"):
                reward_fn.compute(
                    ego_pose=(0.0, 0.0, 0.0),
                    trajectory_xy=torch.tensor([[1.0, 0.0]]),
                    headings=torch.zeros(1),
                    navigation_map=drivable_corridor_map,
                )


# ---------------------------------------------------------------------------
# 4. RewardRegistry & Auxiliary Rewards
# ---------------------------------------------------------------------------


class TestRewardRegistryAndFramework:
    """Tests covering RewardRegistry, weight configurations, and base reward stubs."""

    def test_reward_registry_initialization_and_computation(
        self, drivable_corridor_map: MockNavigationMap, straight_trajectory_10_steps: tuple[torch.Tensor, torch.Tensor]
    ):
        """RewardRegistry initializes active rewards based on config keys and computes weighted total."""
        traj, headings = straight_trajectory_10_steps
        weights = {
            "w_safe": 2.0,
            "w_prog": 1.0,
            "w_comf": 0.5,
            "w_reason": 1.5,
            "lambda_il": -0.1,
        }

        registry = RewardRegistry(config_weights=weights)

        assert "w_safe" in registry.rewards
        assert isinstance(registry.rewards["w_safe"], SafetyReward)
        assert isinstance(registry.rewards["w_prog"], ProgressReward)
        assert isinstance(registry.rewards["w_comf"], ComfortReward)
        assert isinstance(registry.rewards["w_reason"], ReasoningReward)
        assert isinstance(registry.rewards["lambda_il"], ImitationAnchor)

        total_reward, components = registry.compute_total_reward(
            ego_pose=(0.0, 0.0, 0.0),
            trajectory_xy=traj,
            headings=headings,
            navigation_map=drivable_corridor_map,
            speed=10.0,
            acceleration=0.0,
            yaw_rate=0.0,
            info={"dynamic_agents": []},
            reasoning_faithfulness_gate=0.8,
        )

        assert "w_safe" in components
        assert components["w_safe"] == pytest.approx(0.0, abs=1e-6)
        assert total_reward == pytest.approx(0.0, abs=1e-6)

    def test_reasoning_reward_faithfulness_gate(self):
        """ReasoningReward scales by the causal faithfulness gate scalar g."""
        reward_fn = ReasoningReward()
        g_val = 0.75
        reward = reward_fn.compute(reasoning_faithfulness_gate=g_val)
        assert isinstance(reward, float)

    def test_auxiliary_reward_stubs_return_float(self):
        """ProgressReward, ComfortReward, and ImitationAnchor return scalar floats."""
        assert isinstance(ProgressReward().compute(), float)
        assert isinstance(ComfortReward().compute(), float)
        assert isinstance(ImitationAnchor().compute(), float)
