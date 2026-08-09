"""Egomotion derivation for the KIT Scenes Multimodal dataset.

KIT Scenes does not store velocity/acceleration/
curvature columns. It provides only 6-DOF ego poses in ``poses.txt`` (TUM
format, UTM frame) at the 10 Hz reference rate. The four model signals are
therefore *derived* from the pose sequence by finite differencing:

    egomotion_history  (256,) — 64 timesteps before the sample point x 4 signals
                                [speed, acceleration, yaw_rate, curvature]

    trajectory_target  (128,) — 64 timesteps after the sample point x 2 signals
                                [acceleration, curvature]

The pose stream is already at 10 Hz, so there is no downsampling step.
"""

from __future__ import annotations

import numpy as np
import torch
from kitscenes.schema import EgoPose
from scipy.spatial.transform import Rotation

from .temporal_contract import (
    KITSCENES_ABI_FUTURE_STEPS,
    KITSCENES_ABI_HISTORY_STEPS,
)

_HISTORY_TIMESTEPS = KITSCENES_ABI_HISTORY_STEPS
_FUTURE_TIMESTEPS = KITSCENES_ABI_FUTURE_STEPS
_NUM_HISTORY_SIGNALS = 4        # speed, acceleration, yaw_rate, curvature
_NUM_TARGET_SIGNALS = 2         # acceleration, curvature

EGOMOTION_DIM = _HISTORY_TIMESTEPS * _NUM_HISTORY_SIGNALS   # 256
TRAJECTORY_DIM = _FUTURE_TIMESTEPS * _NUM_TARGET_SIGNALS    # 128

# Minimum number of ego poses for a scene to yield at least one valid sample.
MIN_ROWS = _HISTORY_TIMESTEPS + _FUTURE_TIMESTEPS + 1  # 129

# Indices for selecting target signals.
_ACCELERATION_IDX = 1
_CURVATURE_IDX = 3

# Target signals selected from the derived array: [acceleration, curvature].
_TARGET_IDX = [_ACCELERATION_IDX, _CURVATURE_IDX]

# Speed floor (m/s) guarding the curvature = yaw_rate / speed division.
_MIN_SPEED = 0.1


def pose_yaws(poses: tuple[EgoPose, ...]) -> np.ndarray:
    """Return absolute Z-up yaw angles from pose quaternions in radians."""
    quats = np.array([p.rotation for p in poses], dtype=np.float64)
    return Rotation.from_quat(quats).as_euler("ZYX")[:, 0]


def poses_to_arrays(
    poses: tuple[EgoPose, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive egomotion quantities and extract UTM translations from ego poses.

    ``np.gradient`` uses the real (possibly uneven) timestamps and falls back
    to one-sided differences at the boundaries.

    Args:
        poses: Ego poses for one scene, ordered along the reference timeline.

    Returns:
        egomotion: Float32 array of shape (T, 4):
            [speed, acceleration, yaw_rate, curvature].
            - speed: ground-plane (XY) world velocity magnitude, m/s.
            - acceleration: time-derivative of speed (longitudinal), m/s^2.
            - yaw_rate: d(yaw)/dt from the pose quaternion, rad/s (matches L2D/NVIDIA).
            - curvature: yaw_rate / speed, 1/m, with speed floored at
              _MIN_SPEED.
        translations_local: Float64 array of shape (T, 2): scene-local frame
            coordinates [easting, northing] in metres per timestep. These match the
            coordinate frame of the Lanelet2 HD map and the ``get_lanelets_in_roi``
            query.
    """
    t_s = np.array([p.timestamp_ns for p in poses], dtype=np.float64) / 1e9
    translations = np.array([p.translation for p in poses], dtype=np.float64)  # (T, 3)
    velocity_xy = np.gradient(translations[:, :2], t_s, axis=0)  # (T, 2)
    speed = np.linalg.norm(velocity_xy, axis=1)                   # (T,)

    acceleration = np.gradient(speed, t_s)                        # (T,)

    yaw_angle = pose_yaws(poses)
    yaw_rate = np.gradient(np.unwrap(yaw_angle), t_s)             # (T,)
    speed_safe = np.where(speed < _MIN_SPEED, _MIN_SPEED, speed)
    curvature = yaw_rate / speed_safe                              # (T,)

    # Channel 2 is YAW_RATE (rad/s), matching L2D and NVIDIA so the merged
    # multi-dataset loader feeds one consistent physical quantity into the shared
    # ego encoder (yaw_rate is already computed above for curvature).
    egomotion = np.stack(
        [speed, acceleration, yaw_rate, curvature], axis=1
    ).astype(np.float32)

    translations_local = translations[:, :2].astype(np.float64)     # (T, 2) easting, northing
    return egomotion, translations_local


def load_egomotion(
    egomotion_arr: np.ndarray,
    frame_idx: int,
    *,
    history_steps: int = _HISTORY_TIMESTEPS,
    future_steps: int = _FUTURE_TIMESTEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice egomotion around a sample point and pad to the fixed model ABI.

    Training uses the default 64 past and 64 future rows. The KITScenes paper
    approximation uses 40 past and 50 future rows while retaining the same
    tensor shapes: history is left-zero-padded and targets are right-zero-padded.

    ``frame_idx`` must satisfy the requested sampling margins.

    Args:
        egomotion_arr: Derived (T, 4) array from ``poses_to_arrays``.
        frame_idx: Reference-timeline index treated as the current moment.
        history_steps: Real history rows to retain, at most 64.
        future_steps: Real future target rows to retain, at most 64.

    Returns:
        egomotion_history: Float tensor of shape ``(256,)``.
        trajectory_target: Float tensor of shape ``(128,)``.
    """
    if not 1 <= history_steps <= _HISTORY_TIMESTEPS:
        raise ValueError(
            "history_steps must be in "
            f"[1,{_HISTORY_TIMESTEPS}], got {history_steps}"
        )
    if not 1 <= future_steps <= _FUTURE_TIMESTEPS:
        raise ValueError(
            "future_steps must be in "
            f"[1,{_FUTURE_TIMESTEPS}], got {future_steps}"
        )

    min_idx = history_steps
    max_idx = len(egomotion_arr) - future_steps - 1
    if not (min_idx <= frame_idx <= max_idx):
        raise ValueError(
            f"frame_idx {frame_idx} out of valid range [{min_idx}, {max_idx}] "
            f"for a {len(egomotion_arr)}-pose scene."
        )

    history = np.zeros(
        (_HISTORY_TIMESTEPS, _NUM_HISTORY_SIGNALS),
        dtype=np.float32,
    )
    history[-history_steps:] = egomotion_arr[
        frame_idx - history_steps:frame_idx
    ]
    future = egomotion_arr[
        frame_idx + 1:frame_idx + 1 + future_steps
    ]
    target = np.zeros(
        (_FUTURE_TIMESTEPS, _NUM_TARGET_SIGNALS),
        dtype=np.float32,
    )
    target[:future_steps] = future[:, _TARGET_IDX]

    return (
        torch.from_numpy(history.flatten()),
        torch.from_numpy(target.flatten()),
    )
