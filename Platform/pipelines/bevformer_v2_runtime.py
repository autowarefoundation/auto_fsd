"""Official BEVFormer V2 inference over packed KITScenes samples.

OpenMMLab and BEVFormer imports stay behind the model-loading boundary so the
packing, geometry, and rasterization contracts remain unit-testable without
the legacy CUDA runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import tarfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.geospatial import decode_pose
from Platform.pipelines.bevformer_v2_occupancy import (
    BEVFORMER_V2_CONFIG_NAME,
    BEVFORMER_V2_FRAMES,
    BEVFORMER_V2_REVISION,
    BEVFORMER_V2_WEIGHT_SHA256,
    DetectionBox,
    align_history_projection_to_current,
    pose_to_world_from_top_lidar,
    rasterize_detection_boxes,
    scale_packed_projection,
    temporal_frame_indices,
)

BEVFORMER_V2_CAMERA_COUNT = 6
BEVFORMER_V2_IMAGE_HEIGHT = 256
BEVFORMER_V2_IMAGE_WIDTH = 640
BEVFORMER_V2_IMAGE_MEAN_BGR = (103.53, 116.28, 123.675)
BEVFORMER_V2_CLASS_NAMES = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
)


@dataclass(frozen=True)
class PackedBEVFormerFrame:
    """One packed KITScenes frame before detector-specific preprocessing."""

    sample_uid: str
    episode_id: str
    frame_index: int
    timestamp_ns: int
    image_payloads: tuple[bytes, ...]
    projection_ref_to_camera: np.ndarray
    pose: Mapping[str, float | int]

    def __post_init__(self) -> None:
        if not self.sample_uid or not self.episode_id:
            raise ValueError("packed frame identity must not be empty")
        if self.frame_index < 0:
            raise ValueError("packed frame index must be non-negative")
        if len(self.image_payloads) != BEVFORMER_V2_CAMERA_COUNT:
            raise ValueError("BEVFormer V2 requires six camera payloads")
        if any(not payload for payload in self.image_payloads):
            raise ValueError("packed camera payloads must not be empty")
        projection = np.asarray(self.projection_ref_to_camera)
        if projection.shape != (BEVFORMER_V2_CAMERA_COUNT, 3, 4):
            raise ValueError("packed projection must have shape [6,3,4]")
        if not np.isfinite(projection).all():
            raise ValueError("packed projection must be finite")


def _sample_from_members(
    sample_uid: str,
    members: Mapping[str, bytes],
) -> PackedBEVFormerFrame:
    required = {"meta.json", "calib.json", "pose.npy"}
    required.update(
        f"cam_{camera}.jpg"
        for camera in range(BEVFORMER_V2_CAMERA_COUNT)
    )
    missing = required - set(members)
    if missing:
        raise ValueError(
            f"packed sample {sample_uid!r} is missing {sorted(missing)}"
        )
    metadata = json.loads(members["meta.json"])
    calibration = json.loads(members["calib.json"])
    if not isinstance(metadata, Mapping) or not isinstance(
        calibration,
        Mapping,
    ):
        raise ValueError("packed metadata must be JSON objects")
    if metadata.get("sample_uid") not in (None, sample_uid):
        raise ValueError("packed sample UID differs from its tar key")
    projection_spec = calibration.get("projection")
    if (
        not isinstance(projection_spec, Mapping)
        or projection_spec.get("type") != "pinhole"
        or projection_spec.get("reference_frame") != "top_lidar_flu"
    ):
        raise ValueError(
            "BEVFormer V2 requires top_lidar_flu pinhole calibration"
        )
    pose = decode_pose(members["pose.npy"])
    timestamp_ns = int(pose["timestamp_ns"])
    frame_index = metadata.get("frame_idx")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("packed frame_idx must be an integer")
    episode_id = metadata.get("split_group_uid")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("packed sample has no scene identity")
    projection = np.asarray(
        projection_spec.get("matrix"),
        dtype=np.float64,
    )
    return PackedBEVFormerFrame(
        sample_uid=sample_uid,
        episode_id=episode_id,
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        image_payloads=tuple(
            members[f"cam_{camera}.jpg"]
            for camera in range(BEVFORMER_V2_CAMERA_COUNT)
        ),
        projection_ref_to_camera=projection,
        pose=pose,
    )


def iter_packed_bevformer_frames(
    tar_path: str | Path,
) -> Iterable[PackedBEVFormerFrame]:
    """Yield contiguous WebDataset samples without decoding unrelated members."""
    current_key: str | None = None
    current_members: dict[str, bytes] = {}
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or "." not in member.name:
                continue
            sample_uid, suffix = member.name.split(".", 1)
            if current_key is not None and sample_uid != current_key:
                yield _sample_from_members(current_key, current_members)
                current_members = {}
            current_key = sample_uid
            if suffix not in {
                "meta.json",
                "calib.json",
                "pose.npy",
                *(
                    f"cam_{camera}.jpg"
                    for camera in range(BEVFORMER_V2_CAMERA_COUNT)
                ),
            }:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"could not read tar member {member.name!r}")
            current_members[suffix] = stream.read()
    if current_key is not None:
        yield _sample_from_members(current_key, current_members)


def temporal_frames_for(
    current: PackedBEVFormerFrame,
    history: Mapping[int, PackedBEVFormerFrame],
) -> OrderedDict[int, PackedBEVFormerFrame]:
    """Select exact 2 Hz history, omitting unavailable early packed frames."""
    selected = temporal_frame_indices(current.frame_index)
    frames: OrderedDict[int, PackedBEVFormerFrame] = OrderedDict()
    for offset, frame_index in selected.items():
        frame = current if offset == 0 else history.get(frame_index)
        if frame is None:
            continue
        if frame.episode_id != current.episode_id:
            raise ValueError("temporal history crossed a KITScenes scene")
        frames[offset] = frame
    if frames.get(0) is not current:
        raise ValueError("temporal selection omitted the current frame")
    return frames


def _world_pose(
    frame: PackedBEVFormerFrame,
    *,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> np.ndarray:
    return pose_to_world_from_top_lidar(
        latitude_deg=float(frame.pose["latitude_deg"]),
        longitude_deg=float(frame.pose["longitude_deg"]),
        heading_deg_cw_from_north=float(
            frame.pose["heading_deg_cw_from_north"]
        ),
        origin_latitude_deg=origin_latitude_deg,
        origin_longitude_deg=origin_longitude_deg,
    )


def bevformer_metadata_for(
    frames: Mapping[int, PackedBEVFormerFrame],
    *,
    box_type_3d: Any,
) -> OrderedDict[int, dict[str, Any]]:
    """Build the official V2 metadata map in current top-lidar coordinates."""
    current = frames.get(0)
    if current is None:
        raise ValueError("BEVFormer metadata requires offset zero")
    origin_latitude = float(current.pose["latitude_deg"])
    origin_longitude = float(current.pose["longitude_deg"])
    current_to_world = _world_pose(
        current,
        origin_latitude_deg=origin_latitude,
        origin_longitude_deg=origin_longitude,
    )
    output: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for offset, frame in sorted(frames.items()):
        frame_to_world = _world_pose(
            frame,
            origin_latitude_deg=origin_latitude,
            origin_longitude_deg=origin_longitude,
        )
        projections = []
        frame_to_current: np.ndarray | None = None
        for packed_projection in frame.projection_ref_to_camera:
            if offset == 0:
                aligned = packed_projection
            else:
                aligned, frame_to_current = (
                    align_history_projection_to_current(
                        packed_projection,
                        history_to_world=frame_to_world,
                        current_to_world=current_to_world,
                    )
                )
            scaled = scale_packed_projection(aligned)
            homogeneous = np.eye(4, dtype=np.float32)
            homogeneous[:3, :] = scaled.astype(np.float32)
            projections.append(homogeneous)
        image_shape = (
            BEVFORMER_V2_IMAGE_HEIGHT,
            BEVFORMER_V2_IMAGE_WIDTH,
            3,
        )
        output[offset] = {
            "box_mode_3d": None,
            "box_type_3d": box_type_3d,
            "filename": [frame.sample_uid] * BEVFORMER_V2_CAMERA_COUNT,
            "flip": False,
            "img_norm_cfg": {
                "mean": np.asarray(
                    BEVFORMER_V2_IMAGE_MEAN_BGR,
                    dtype=np.float32,
                ),
                "std": np.ones(3, dtype=np.float32),
                "to_rgb": False,
            },
            "img_shape": [image_shape] * BEVFORMER_V2_CAMERA_COUNT,
            "lidar2img": projections,
            "lidaradj2lidarcurr": frame_to_current,
            "ori_shape": [image_shape] * BEVFORMER_V2_CAMERA_COUNT,
            "pad_shape": [image_shape] * BEVFORMER_V2_CAMERA_COUNT,
            "sample_idx": frame.sample_uid,
            "scale_factor": np.asarray(
                [2.5, 1.0, 2.5, 1.0],
                dtype=np.float32,
            ),
            "scene_token": frame.episode_id,
            "timestamp": frame.timestamp_ns / 1_000_000.0,
        }
    return output


def preprocess_packed_images(
    frame: PackedBEVFormerFrame,
) -> Any:
    """Return `[6,3,256,640]` BGR float images for the official backbone."""
    import torch
    from PIL import Image

    images = []
    mean = np.asarray(
        BEVFORMER_V2_IMAGE_MEAN_BGR,
        dtype=np.float32,
    )
    for payload in frame.image_payloads:
        with Image.open(io.BytesIO(payload)) as source:
            rgb = source.convert("RGB").resize(
                (
                    BEVFORMER_V2_IMAGE_WIDTH,
                    BEVFORMER_V2_IMAGE_HEIGHT,
                ),
                resample=Image.Resampling.BILINEAR,
            )
            values = np.asarray(rgb, dtype=np.float32)
        bgr = np.ascontiguousarray(values[:, :, ::-1])
        bgr -= mean
        images.append(torch.from_numpy(bgr).permute(2, 0, 1))
    return torch.stack(images)


def bevformer_batch_for(
    frames: Mapping[int, PackedBEVFormerFrame],
    *,
    box_type_3d: Any,
    device: Any,
) -> tuple[Any, list[list[OrderedDict[int, dict[str, Any]]]]]:
    """Build the exact image and metadata nesting expected by forward_test."""
    import torch

    ordered = OrderedDict(sorted(frames.items()))
    images = torch.stack(
        [preprocess_packed_images(frame) for frame in ordered.values()]
    ).unsqueeze(0)
    images = images.to(device=device, non_blocking=True)
    metadata = bevformer_metadata_for(
        ordered,
        box_type_3d=box_type_3d,
    )
    return images, [[metadata]]


def detections_from_bevformer_result(
    result: Any,
) -> list[DetectionBox]:
    """Convert one official mmdet3d result without changing box geometry."""
    if (
        not isinstance(result, Sequence)
        or len(result) != 1
        or not isinstance(result[0], Mapping)
        or not isinstance(result[0].get("pts_bbox"), Mapping)
    ):
        raise ValueError("BEVFormer result has an unexpected envelope")
    boxes = result[0]["pts_bbox"]
    box_tensor = boxes.get("boxes_3d")
    scores = boxes.get("scores_3d")
    labels = boxes.get("labels_3d")
    if box_tensor is None or scores is None or labels is None:
        raise ValueError("BEVFormer result is missing 3-D boxes")
    tensor = np.asarray(box_tensor.tensor.detach().cpu(), dtype=np.float64)
    score_values = np.asarray(scores.detach().cpu(), dtype=np.float64)
    label_values = np.asarray(labels.detach().cpu(), dtype=np.int64)
    if (
        tensor.ndim != 2
        or tensor.shape[1] < 7
        or score_values.shape != (tensor.shape[0],)
        or label_values.shape != (tensor.shape[0],)
    ):
        raise ValueError("BEVFormer result tensor shapes are invalid")
    detections = []
    for box, score, label in zip(tensor, score_values, label_values):
        if label < 0 or label >= len(BEVFORMER_V2_CLASS_NAMES):
            raise ValueError("BEVFormer returned an unknown class index")
        detections.append(
            DetectionBox(
                class_name=BEVFORMER_V2_CLASS_NAMES[int(label)],
                score=float(score),
                center_x_m=float(box[0]),
                center_y_m=float(box[1]),
                length_m=float(box[3]),
                width_m=float(box[4]),
                yaw_rad=float(box[6]),
            )
        )
    return detections


def infer_bevformer_frame(
    model: Any,
    current: PackedBEVFormerFrame,
    history: Mapping[int, PackedBEVFormerFrame],
    *,
    box_type_3d: Any,
    device: Any,
    score_threshold: float = 0.2,
) -> np.ndarray:
    """Run official inference and return one uncorrected `[8,450,300]` raster."""
    import torch

    frames = temporal_frames_for(current, history)
    images, metadata = bevformer_batch_for(
        frames,
        box_type_3d=box_type_3d,
        device=device,
    )
    with torch.no_grad():
        result = model(
            return_loss=False,
            img=[images],
            img_metas=metadata,
            rescale=True,
        )
    return rasterize_detection_boxes(
        detections_from_bevformer_result(result),
        score_threshold=score_threshold,
    )


def remember_history_frame(
    history: MutableMapping[int, PackedBEVFormerFrame],
    frame: PackedBEVFormerFrame,
) -> None:
    """Retain only the exact 3.5 second history needed by the t8 model."""
    stale_before = frame.frame_index + min(BEVFORMER_V2_FRAMES) * 5
    for frame_index in list(history):
        if frame_index < stale_before:
            del history[frame_index]
    history[frame.frame_index] = frame


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_bevformer_v2(
    *,
    repository_path: str | Path,
    checkpoint_path: str | Path,
    device: Any,
) -> tuple[Any, Any]:
    """Load only the pinned Apache-2.0 BEVFormer revision and checkpoint."""
    repository = Path(repository_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if sha256_file(checkpoint) != BEVFORMER_V2_WEIGHT_SHA256:
        raise ValueError("BEVFormer V2 weight digest does not match provenance")
    git_head = repository / ".git" / "HEAD"
    if not git_head.exists():
        raise ValueError("BEVFormer repository has no revision metadata")
    import subprocess

    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != BEVFORMER_V2_REVISION:
        raise ValueError("BEVFormer repository revision is not pinned")

    sys.path.insert(0, str(repository))
    try:
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from mmdet3d.models import build_model

        importlib.import_module("projects.mmdet3d_plugin")
        config_path = (
            repository
            / "projects"
            / "configs"
            / "bevformerv2"
            / BEVFORMER_V2_CONFIG_NAME
        )
        config = Config.fromfile(str(config_path))
        config.model.pretrained = None
        config.model.train_cfg = None
        model = build_model(
            config.model,
            test_cfg=config.get("test_cfg"),
        )
        loaded = load_checkpoint(
            model,
            str(checkpoint),
            map_location="cpu",
        )
        metadata = loaded.get("meta", {}) if isinstance(loaded, Mapping) else {}
        model.CLASSES = metadata.get(
            "CLASSES",
            BEVFORMER_V2_CLASS_NAMES,
        )
        model.to(device)
        model.eval()
        return model, LiDARInstance3DBoxes
    finally:
        if sys.path and sys.path[0] == str(repository):
            sys.path.pop(0)
