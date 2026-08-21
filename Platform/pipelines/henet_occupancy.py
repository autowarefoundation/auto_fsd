"""Pure HENet BEV segmentation adaptation for ASOC publications.

HENet publishes three NuScenes BEV segmentation probabilities on a 200 x 200
grid with 0.5 metre cells. This module preserves their physical coordinates
when producing the Dashboard's 450 x 300 ASOC geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
)

HENET_REPOSITORY = "https://github.com/VDIGPKU/HENet"
HENET_REVISION = "29ca81dd109cabe0a0c53ee354c4a74ad1559740"
HENET_CONFIG_NAME = "henet_det_bevseg.py"
HENET_WEIGHT_SOURCE_URL = (
    "https://drive.google.com/drive/folders/"
    "1AYajSnL1JLrFOTHcpjX7WlRRLXUxQhwV"
)
HENET_CODE_LICENSE_SPDX = "LicenseRef-HENet-Research-Only"
HENET_WEIGHT_LICENSE_SPDX = "LicenseRef-HENet-Research-Only"
HENET_TRAINING_DATA_LICENSE_SPDX = "NOASSERTION"
HENET_ARTIFACT_KIND = "native-semantic-occupancy"
HENET_HEAD_VERSION = "henet-det-bevseg-v1"
HENET_CAMERA_ORDER = (1, 0, 2, 4, 3, 5)
HENET_CAMERA_COUNT = len(HENET_CAMERA_ORDER)
HENET_SOURCE_HZ = 10
HENET_MODEL_HZ = 2
HENET_SHORT_FRAME_OFFSETS = (0, -5, -10)
HENET_LONG_FRAME_OFFSETS = tuple(range(0, -45, -5))
HENET_INPUT_HEIGHT = 640
HENET_INPUT_WIDTH = 1152
HENET_LONGTERM_INPUT_HEIGHT = 256
HENET_LONGTERM_INPUT_WIDTH = 704
HENET_SOURCE_X_MIN_M = -50.0
HENET_SOURCE_X_MAX_M = 50.0
HENET_SOURCE_Y_MIN_M = -50.0
HENET_SOURCE_Y_MAX_M = 50.0
HENET_SOURCE_METERS_PER_CELL = 0.5
HENET_SOURCE_HEIGHT = 200
HENET_SOURCE_WIDTH = 200
HENET_SEGMENTATION_CLASS_NAMES = (
    "vehicle",
    "drivable_area",
    "divider",
)

SEMANTIC_OCCUPANCY_CLASS_NAMES = (
    "drivable_area",
    "lane_area",
    "intersection",
    "crosswalk",
    "stop_line",
    "vehicle",
    "vulnerable_road_user",
    "other_obstacle",
)


def _validate_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef"
        for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _rq_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular and orthonormal factors for a 3 x 3 matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError("projection matrix must be a finite [3,3] matrix")

    # NumPy has QR but not RQ. Reverse the axes around QR to obtain M = K @ R.
    orthonormal, upper = np.linalg.qr(np.flipud(values).T)
    intrinsic = np.flipud(upper.T)
    intrinsic = np.fliplr(intrinsic)
    rotation = orthonormal.T
    rotation = np.flipud(rotation)
    return intrinsic, rotation


def decompose_pinhole_projection(
    projection_ref_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover packed-image intrinsics and camera-to-ego from ``K[R|t]``.

    The packed KITScenes calibration uses the top-lidar FLU frame as its
    reference. HENet's ``sensor2ego`` tensor is therefore camera-to-top-lidar.
    """
    projection = np.asarray(projection_ref_to_camera, dtype=np.float64)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("projection must be a finite [3,4] matrix")

    intrinsic, rotation_ref_to_camera = _rq_decomposition(projection[:, :3])
    if abs(intrinsic[2, 2]) < 1e-12:
        raise ValueError("projection intrinsic scale is singular")

    signs = np.where(np.diag(intrinsic) < 0.0, -1.0, 1.0)
    sign_matrix = np.diag(signs)
    intrinsic = intrinsic @ sign_matrix
    rotation_ref_to_camera = sign_matrix @ rotation_ref_to_camera
    if np.linalg.det(rotation_ref_to_camera) < 0.0:
        intrinsic[:, 2] *= -1.0
        rotation_ref_to_camera[2, :] *= -1.0
    if not np.allclose(
        rotation_ref_to_camera @ rotation_ref_to_camera.T,
        np.eye(3),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("projection rotation is not orthonormal")

    intrinsic /= intrinsic[2, 2]
    translation_ref_to_camera = np.linalg.solve(
        intrinsic,
        projection[:, 3],
    )
    camera_to_ref = np.eye(4, dtype=np.float64)
    camera_to_ref[:3, :3] = rotation_ref_to_camera.T
    camera_to_ref[:3, 3] = (
        -rotation_ref_to_camera.T @ translation_ref_to_camera
    )
    return intrinsic, camera_to_ref


def _source_coordinates(
    x_forward_m: np.ndarray,
    y_left_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_row = (
        (x_forward_m - HENET_SOURCE_X_MIN_M)
        / HENET_SOURCE_METERS_PER_CELL
        - 0.5
    )
    source_col = (
        (y_left_m - HENET_SOURCE_Y_MIN_M)
        / HENET_SOURCE_METERS_PER_CELL
        - 0.5
    )
    valid = (
        (source_row >= 0.0)
        & (source_row <= HENET_SOURCE_HEIGHT - 1)
        & (source_col >= 0.0)
        & (source_col <= HENET_SOURCE_WIDTH - 1)
    )
    return source_row, source_col, valid


def _sample_henet_probability(
    probability: np.ndarray,
    *,
    x_forward_m: np.ndarray,
    y_left_m: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample HENet cells at Dashboard pixel centres."""
    source_row, source_col, valid = _source_coordinates(
        x_forward_m,
        y_left_m,
    )
    row0 = np.floor(source_row).astype(np.int64)
    col0 = np.floor(source_col).astype(np.int64)
    row1 = np.minimum(row0 + 1, HENET_SOURCE_HEIGHT - 1)
    col1 = np.minimum(col0 + 1, HENET_SOURCE_WIDTH - 1)
    row0 = np.clip(row0, 0, HENET_SOURCE_HEIGHT - 1)
    col0 = np.clip(col0, 0, HENET_SOURCE_WIDTH - 1)
    row_fraction = source_row - np.floor(source_row)
    col_fraction = source_col - np.floor(source_col)

    sampled = np.zeros(
        (probability.shape[0], *x_forward_m.shape),
        dtype=np.float32,
    )
    for class_index, source in enumerate(probability):
        top = (
            source[row0, col0] * (1.0 - col_fraction)
            + source[row0, col1] * col_fraction
        )
        bottom = (
            source[row1, col0] * (1.0 - col_fraction)
            + source[row1, col1] * col_fraction
        )
        sampled[class_index] = np.where(
            valid,
            top * (1.0 - row_fraction) + bottom * row_fraction,
            0.0,
        )
    return sampled


def adapt_henet_segmentation(
    probability: np.ndarray,
    *,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
) -> np.ndarray:
    """Map official HENet probabilities to the fixed ASOC taxonomy and grid.

    HENet's third output is a road divider rather than a lane area. It is
    retained in ASOC's lane channel only to make the three published semantic
    outputs inspectable in one fixed Dashboard taxonomy.
    """
    values = np.asarray(probability, dtype=np.float32)
    expected_shape = (
        len(HENET_SEGMENTATION_CLASS_NAMES),
        HENET_SOURCE_HEIGHT,
        HENET_SOURCE_WIDTH,
    )
    if values.shape != expected_shape:
        raise ValueError(
            "HENet probability must have shape "
            f"{list(expected_shape)}, got {list(values.shape)}"
        )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(
        values > 1.0
    ):
        raise ValueError("HENet probability must be finite in [0,1]")

    x_forward_m, y_left_m = geometry.pixel_center_grids()
    sampled = _sample_henet_probability(
        values,
        x_forward_m=x_forward_m,
        y_left_m=y_left_m,
    )
    output = np.zeros(
        (
            len(SEMANTIC_OCCUPANCY_CLASS_NAMES),
            geometry.height_px,
            geometry.width_px,
        ),
        dtype=np.float32,
    )
    output[SEMANTIC_OCCUPANCY_CLASS_NAMES.index("drivable_area")] = sampled[
        HENET_SEGMENTATION_CLASS_NAMES.index("drivable_area")
    ]
    output[SEMANTIC_OCCUPANCY_CLASS_NAMES.index("lane_area")] = sampled[
        HENET_SEGMENTATION_CLASS_NAMES.index("divider")
    ]
    output[SEMANTIC_OCCUPANCY_CLASS_NAMES.index("vehicle")] = sampled[
        HENET_SEGMENTATION_CLASS_NAMES.index("vehicle")
    ]
    return output


def provenance(checkpoint_sha256: str) -> Mapping[str, object]:
    """Return the immutable scientific provenance for one HENet checkpoint."""
    return {
        "artifact_kind": HENET_ARTIFACT_KIND,
        "config": HENET_CONFIG_NAME,
        "head_version": HENET_HEAD_VERSION,
        "repository": HENET_REPOSITORY,
        "repository_revision": HENET_REVISION,
        "weight_sha256": _validate_sha256(
            checkpoint_sha256,
            "checkpoint_sha256",
        ),
        "weight_source_url": HENET_WEIGHT_SOURCE_URL,
        "code_license_spdx": HENET_CODE_LICENSE_SPDX,
        "weight_license_spdx": HENET_WEIGHT_LICENSE_SPDX,
        "training_data_license_spdx": HENET_TRAINING_DATA_LICENSE_SPDX,
        "supported_semantic_classes": [
            "drivable_area",
            "lane_area",
            "vehicle",
        ],
        "teacher_available": False,
        "limitations": [
            (
                "The official HENet repository and supplied checkpoint are "
                "free only for academic research; commercial use requires "
                "authorization from the authors."
            ),
            (
                "The lane channel contains HENet road-divider probability, "
                "not a lane-area segmentation."
            ),
            (
                "KITScenes 256-square camera images are upsampled into the "
                "official HENet input sizes and therefore have lower source "
                "resolution than official nuScenes evaluation."
            ),
            (
                "Teacher and Error views are unavailable because KITScenes "
                "packed shards do not contain perception ground truth."
            ),
        ],
    }
