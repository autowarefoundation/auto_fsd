"""Pure KITScenes adaptation helpers for BEVFormer V2 detections.

The published BEVFormer V2 checkpoint is a 3-D detector, not an occupancy
segmentation model. This module therefore rasterizes only the ground-plane
footprints of predicted boxes. It never invents road, map, or teacher labels.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np

from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
)

BEVFORMER_V2_REPOSITORY = "https://github.com/fundamentalvision/BEVFormer"
BEVFORMER_V2_REVISION = "66b65f3a1f58caf0507cb2a971b9c0e7f842376c"
BEVFORMER_V2_CONFIG_NAME = "bevformerv2-r50-t8-24ep.py"
BEVFORMER_V2_WEIGHT_SHA256 = (
    "5585bc4d3ff8b396928cb92d91f773a2c57a81258f83cab0c668ebb2eb9d3307"
)
BEVFORMER_V2_WEIGHT_SOURCE_URL = (
    "https://drive.google.com/drive/folders/1Ml_usx5BNx43CFH1Di2OTazuzSyAlBto"
)
BEVFORMER_V2_CODE_LICENSE_SPDX = "Apache-2.0"
BEVFORMER_V2_WEIGHT_LICENSE_SPDX = "NOASSERTION"
BEVFORMER_V2_TRAINING_DATA_LICENSE_SPDX = "CC-BY-NC-SA-4.0"
BEVFORMER_V2_HEAD_VERSION = "bevformer-v2-r50-t8-box-raster-v1"
BEVFORMER_V2_ARTIFACT_KIND = "detection-derived-occupancy"
BEVFORMER_V2_FRAMES = (-7, -6, -5, -4, -3, -2, -1, 0)
BEVFORMER_V2_SOURCE_HZ = 2
KITSCENES_SOURCE_HZ = 10

SEMANTIC_CLASS_NAMES = (
    "drivable_area",
    "lane_area",
    "intersection",
    "crosswalk",
    "stop_line",
    "vehicle",
    "vulnerable_road_user",
    "other_obstacle",
)

_BEVFORMER_CLASS_TO_SEMANTIC = {
    "barrier": "other_obstacle",
    "bicycle": "vulnerable_road_user",
    "bus": "vehicle",
    "car": "vehicle",
    "construction_vehicle": "vehicle",
    "motorcycle": "vulnerable_road_user",
    "pedestrian": "vulnerable_road_user",
    "traffic_cone": "other_obstacle",
    "trailer": "vehicle",
    "truck": "vehicle",
}

BEVFORMER_V2_SUPPORTED_SEMANTIC_CLASSES = tuple(
    sorted(set(_BEVFORMER_CLASS_TO_SEMANTIC.values()))
)


@dataclasses.dataclass(frozen=True)
class DetectionBox:
    """One BEVFormer box in the current KITScenes top-lidar FLU frame."""

    class_name: str
    score: float
    center_x_m: float
    center_y_m: float
    length_m: float
    width_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        numeric = (
            self.score,
            self.center_x_m,
            self.center_y_m,
            self.length_m,
            self.width_m,
            self.yaw_rad,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("detection box values must be finite")
        if self.class_name not in _BEVFORMER_CLASS_TO_SEMANTIC:
            raise ValueError(f"unsupported BEVFormer class {self.class_name!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("detection score must be in [0,1]")
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("detection dimensions must be positive")


def semantic_class_for_detection(class_name: str) -> str:
    """Return the disclosed semantic class for one official detector class."""
    try:
        return _BEVFORMER_CLASS_TO_SEMANTIC[class_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported BEVFormer class {class_name!r}"
        ) from error


def temporal_frame_indices(
    current_frame_index: int,
    *,
    scene_start_index: int = 0,
    source_hz: int = KITSCENES_SOURCE_HZ,
    model_hz: int = BEVFORMER_V2_SOURCE_HZ,
    model_frames: Sequence[int] = BEVFORMER_V2_FRAMES,
) -> dict[int, int]:
    """Map BEVFormer temporal offsets to KITScenes 10 Hz frame indices.

    Missing early history is omitted. BEVFormer V2 fills missing BEV features
    from the nearest available frame during fusion, matching its official
    dataset path.
    """
    if current_frame_index < scene_start_index:
        raise ValueError("current frame precedes the scene start")
    if source_hz <= 0 or model_hz <= 0 or source_hz % model_hz:
        raise ValueError("source_hz must be a positive multiple of model_hz")
    if not model_frames or 0 not in model_frames:
        raise ValueError("model_frames must include the current frame")
    if len(set(model_frames)) != len(model_frames):
        raise ValueError("model_frames must be unique")
    if any(frame > 0 for frame in model_frames):
        raise ValueError("future frames are not valid detector inputs")

    stride = source_hz // model_hz
    selected = {
        int(offset): current_frame_index + int(offset) * stride
        for offset in sorted(model_frames)
        if current_frame_index + int(offset) * stride >= scene_start_index
    }
    if selected.get(0) != current_frame_index:
        raise ValueError("current frame selection is invalid")
    return selected


def pose_to_world_from_top_lidar(
    *,
    latitude_deg: float,
    longitude_deg: float,
    heading_deg_cw_from_north: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> np.ndarray:
    """Approximate a local ENU transform for one packed KITScenes pose.

    The Console snapshot carries WGS84 position and heading rather than the raw
    six-degree-of-freedom pose. The small-area equirectangular conversion keeps
    translation in metres; roll and pitch are intentionally unavailable.
    """
    values = (
        latitude_deg,
        longitude_deg,
        heading_deg_cw_from_north,
        origin_latitude_deg,
        origin_longitude_deg,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose coordinates must be finite")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude is outside WGS84 bounds")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("longitude is outside WGS84 bounds")

    earth_radius_m = 6_378_137.0
    origin_latitude_rad = math.radians(origin_latitude_deg)
    east_m = (
        math.radians(longitude_deg - origin_longitude_deg)
        * earth_radius_m
        * math.cos(origin_latitude_rad)
    )
    north_m = (
        math.radians(latitude_deg - origin_latitude_deg)
        * earth_radius_m
    )
    heading_rad = math.radians(heading_deg_cw_from_north)
    sin_heading = math.sin(heading_rad)
    cos_heading = math.cos(heading_rad)

    transform = np.eye(4, dtype=np.float64)
    # Columns are the FLU forward, left, and up axes expressed in local ENU.
    transform[:3, :3] = np.asarray(
        [
            [sin_heading, -cos_heading, 0.0],
            [cos_heading, sin_heading, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = [east_m, north_m, 0.0]
    return transform


def align_history_projection_to_current(
    projection_ref_to_camera: np.ndarray,
    *,
    history_to_world: np.ndarray,
    current_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a history camera projection in the current top-lidar frame."""
    projection = np.asarray(projection_ref_to_camera, dtype=np.float64)
    history_pose = np.asarray(history_to_world, dtype=np.float64)
    current_pose = np.asarray(current_to_world, dtype=np.float64)
    if projection.shape != (3, 4):
        raise ValueError("projection must have shape [3,4]")
    if history_pose.shape != (4, 4) or current_pose.shape != (4, 4):
        raise ValueError("poses must have shape [4,4]")
    if not (
        np.isfinite(projection).all()
        and np.isfinite(history_pose).all()
        and np.isfinite(current_pose).all()
    ):
        raise ValueError("projection transforms must be finite")

    history_to_current = np.linalg.inv(current_pose) @ history_pose
    current_to_history = np.linalg.inv(history_to_current)
    return projection @ current_to_history, history_to_current


def scale_packed_projection(
    projection_ref_to_camera: np.ndarray,
    *,
    packed_width: int = 256,
    packed_height: int = 256,
    model_width: int = 640,
    model_height: int = 256,
) -> np.ndarray:
    """Scale packed-image intrinsics to the BEVFormer evaluation tensor."""
    if min(packed_width, packed_height, model_width, model_height) <= 0:
        raise ValueError("image dimensions must be positive")
    projection = np.asarray(projection_ref_to_camera, dtype=np.float64)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("projection must be a finite [3,4] matrix")
    image_scale = np.diag(
        [
            model_width / packed_width,
            model_height / packed_height,
            1.0,
        ]
    )
    return image_scale @ projection


def rasterize_detection_boxes(
    detections: Sequence[DetectionBox],
    *,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    score_threshold: float = 0.2,
    max_detections: int = 300,
) -> np.ndarray:
    """Rasterize box footprints into `[8,H,W]` semantic probabilities.

    A cell is occupied when its physical centre lies within the exact predicted
    oriented rectangle. Overlapping predictions use maximum confidence.
    """
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0,1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    probability = np.zeros(
        (
            len(SEMANTIC_CLASS_NAMES),
            geometry.height_px,
            geometry.width_px,
        ),
        dtype=np.float32,
    )
    selected = sorted(
        (
            detection
            for detection in detections
            if detection.score >= score_threshold
        ),
        key=lambda detection: detection.score,
        reverse=True,
    )[:max_detections]
    if not selected:
        return probability

    x_grid, y_grid = geometry.pixel_center_grids()
    for detection in selected:
        semantic_name = semantic_class_for_detection(detection.class_name)
        class_index = SEMANTIC_CLASS_NAMES.index(semantic_name)
        cos_yaw = math.cos(detection.yaw_rad)
        sin_yaw = math.sin(detection.yaw_rad)
        delta_x = x_grid - detection.center_x_m
        delta_y = y_grid - detection.center_y_m
        # Match mmdet3d v0.17.1 LiDARInstance3DBoxes.corners. Its row-vector
        # local-to-LiDAR rotation is clockwise, so this is the inverse map.
        box_forward = delta_x * cos_yaw - delta_y * sin_yaw
        box_left = delta_x * sin_yaw + delta_y * cos_yaw
        occupied = (
            (np.abs(box_forward) <= detection.length_m * 0.5)
            & (np.abs(box_left) <= detection.width_m * 0.5)
        )
        probability[class_index, occupied] = np.maximum(
            probability[class_index, occupied],
            detection.score,
        )
    return probability


def provenance() -> Mapping[str, object]:
    """Return public scientific provenance for the dedicated Dashboard."""
    return {
        "artifact_kind": BEVFORMER_V2_ARTIFACT_KIND,
        "config": BEVFORMER_V2_CONFIG_NAME,
        "head_version": BEVFORMER_V2_HEAD_VERSION,
        "repository": BEVFORMER_V2_REPOSITORY,
        "repository_revision": BEVFORMER_V2_REVISION,
        "weight_sha256": BEVFORMER_V2_WEIGHT_SHA256,
        "weight_source_url": BEVFORMER_V2_WEIGHT_SOURCE_URL,
        "code_license_spdx": BEVFORMER_V2_CODE_LICENSE_SPDX,
        "weight_license_spdx": BEVFORMER_V2_WEIGHT_LICENSE_SPDX,
        "training_data_license_spdx": (
            BEVFORMER_V2_TRAINING_DATA_LICENSE_SPDX
        ),
        "supported_semantic_classes": list(
            BEVFORMER_V2_SUPPORTED_SEMANTIC_CLASSES
        ),
        "teacher_available": False,
        "limitations": [
            "Official BEVFormer V2 publishes 3-D detection, not BEV segmentation.",
            "Object occupancy is derived only from predicted box footprints.",
            "Road and map classes are unsupported and remain empty.",
            (
                "KITScenes packed square images are stretched to the official "
                "2.5:1 input aspect; this changes the source-camera aspect "
                "ratio and uses 2.5x lower linear resolution than official "
                "nuScenes evaluation."
            ),
            "Teacher and Error views are unavailable without perception labels.",
            (
                "The public weight has no separately stated license; its "
                "nuScenes training data is CC-BY-NC-SA-4.0."
            ),
        ],
    }
