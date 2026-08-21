"""Official HENet inference over calibrated packed KITScenes samples."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import subprocess
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
    pose_to_world_from_top_lidar,
)
from Platform.pipelines.henet_occupancy import (
    HENET_CAMERA_COUNT,
    HENET_CAMERA_ORDER,
    HENET_CONFIG_NAME,
    HENET_INPUT_HEIGHT,
    HENET_INPUT_WIDTH,
    HENET_LONGTERM_INPUT_HEIGHT,
    HENET_LONGTERM_INPUT_WIDTH,
    HENET_LONG_FRAME_OFFSETS,
    HENET_REVISION,
    HENET_SEGMENTATION_CLASS_NAMES,
    HENET_SHORT_FRAME_OFFSETS,
    adapt_henet_segmentation,
    decompose_pinhole_projection,
)

HENET_IMAGE_MEAN = (123.675, 116.28, 103.53)
HENET_IMAGE_STD = (58.395, 57.12, 57.375)


@dataclass(frozen=True)
class PackedHENetFrame:
    """One six-camera KITScenes frame in HENet's official camera order."""

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
        if len(self.image_payloads) != HENET_CAMERA_COUNT:
            raise ValueError("HENet requires six camera payloads")
        if any(not payload for payload in self.image_payloads):
            raise ValueError("packed camera payloads must not be empty")
        projection = np.asarray(self.projection_ref_to_camera)
        if projection.shape != (HENET_CAMERA_COUNT, 3, 4):
            raise ValueError("packed projection must have shape [6,3,4]")
        if not np.isfinite(projection).all():
            raise ValueError("packed projection must be finite")


def _sample_from_members(
    sample_uid: str,
    members: Mapping[str, bytes],
) -> PackedHENetFrame:
    required = {"meta.json", "calib.json", "pose.npy"}
    required.update(
        f"cam_{camera}.jpg"
        for camera in range(HENET_CAMERA_COUNT)
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
        raise ValueError("HENet requires top_lidar_flu pinhole calibration")
    frame_index = metadata.get("frame_idx")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("packed frame_idx must be an integer")
    episode_id = metadata.get("split_group_uid")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("packed sample has no scene identity")
    pose = decode_pose(members["pose.npy"])
    projection = np.asarray(
        projection_spec.get("matrix"),
        dtype=np.float64,
    )
    if projection.shape != (HENET_CAMERA_COUNT, 3, 4):
        raise ValueError("packed projection must have shape [6,3,4]")
    camera_order = np.asarray(HENET_CAMERA_ORDER)
    return PackedHENetFrame(
        sample_uid=sample_uid,
        episode_id=episode_id,
        frame_index=frame_index,
        timestamp_ns=int(pose["timestamp_ns"]),
        image_payloads=tuple(
            members[f"cam_{camera}.jpg"]
            for camera in HENET_CAMERA_ORDER
        ),
        projection_ref_to_camera=projection[camera_order],
        pose=pose,
    )


def iter_packed_henet_frames(
    tar_path: str | Path,
) -> Iterable[PackedHENetFrame]:
    """Yield packed KITScenes samples without decoding unrelated members."""
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
                    for camera in range(HENET_CAMERA_COUNT)
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
    current: PackedHENetFrame,
    history: Mapping[int, PackedHENetFrame],
    *,
    frame_offsets: Sequence[int],
) -> OrderedDict[int, PackedHENetFrame]:
    """Select fixed official HENet inputs, repeating current at boundaries."""
    offsets = tuple(frame_offsets)
    if not offsets or offsets[0] != 0 or any(offset > 0 for offset in offsets):
        raise ValueError("HENet offsets must start at zero and use past frames")
    if len(set(offsets)) != len(offsets):
        raise ValueError("HENet frame offsets must be unique")

    selected: OrderedDict[int, PackedHENetFrame] = OrderedDict()
    for offset in offsets:
        candidate = (
            current
            if offset == 0
            else history.get(current.frame_index + offset)
        )
        if candidate is not None and candidate.episode_id != current.episode_id:
            raise ValueError("temporal history crossed a KITScenes scene")
        # The official NuScenes loader substitutes the key frame when a
        # requested predecessor crosses a scene boundary.
        selected[offset] = candidate if candidate is not None else current
    return selected


def temporal_substitution_count(
    current: PackedHENetFrame,
    history: Mapping[int, PackedHENetFrame],
    *,
    frame_offsets: Sequence[int],
) -> int:
    """Return how many fixed HENet temporal slots repeat the key frame."""
    return sum(
        1
        for offset in frame_offsets
        if offset != 0 and current.frame_index + offset not in history
    )


def _world_pose(
    frame: PackedHENetFrame,
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


def _preprocess_image(
    payload: bytes,
    *,
    output_height: int,
    output_width: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Apply HENet's deterministic test resize, crop, and normalization."""
    import torch
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as source:
        source = source.convert("RGB")
        source_width, source_height = source.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("packed camera dimensions must be positive")
        resize = output_width / source_width
        resized_height = int(source_height * resize)
        if resized_height < output_height:
            raise ValueError("HENet test crop exceeds resized image height")
        crop_top = resized_height - output_height
        image = source.resize(
            (output_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        ).crop((0, crop_top, output_width, resized_height))

    values = np.asarray(image, dtype=np.float32)
    # HENet's mmlabNormalize receives a PIL RGB image with to_rgb=True,
    # which reverses channels before applying the official RGB statistics.
    values = values[:, :, ::-1]
    values = (values - np.asarray(HENET_IMAGE_MEAN)) / np.asarray(
        HENET_IMAGE_STD
    )
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(values)
    ).permute(2, 0, 1)
    post_rotation = np.eye(3, dtype=np.float32)
    post_rotation[0, 0] = resize
    post_rotation[1, 1] = resize
    post_translation = np.asarray([0.0, -crop_top, 0.0], dtype=np.float32)
    return image_tensor, post_rotation, post_translation


def _image_inputs_for(
    frames: Mapping[int, PackedHENetFrame],
    *,
    output_height: int,
    output_width: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Build a HENet img_inputs tuple with official tensor ordering."""
    import torch

    ordered = tuple(frames.values())
    if not ordered:
        raise ValueError("HENet inference requires at least one frame")
    current = ordered[0]
    origin_latitude = float(current.pose["latitude_deg"])
    origin_longitude = float(current.pose["longitude_deg"])

    per_frame_images = []
    sensor_to_ego = []
    ego_to_global = []
    intrinsics = []
    post_rotations = []
    post_translations = []
    for frame in ordered:
        frame_images = []
        frame_sensor_to_ego = []
        frame_intrinsics = []
        frame_post_rotations = []
        frame_post_translations = []
        for payload, projection in zip(
            frame.image_payloads,
            frame.projection_ref_to_camera,
        ):
            image, post_rotation, post_translation = _preprocess_image(
                payload,
                output_height=output_height,
                output_width=output_width,
            )
            intrinsic, camera_to_ego = decompose_pinhole_projection(
                projection
            )
            frame_images.append(image)
            frame_sensor_to_ego.append(camera_to_ego.astype(np.float32))
            frame_intrinsics.append(intrinsic.astype(np.float32))
            frame_post_rotations.append(post_rotation)
            frame_post_translations.append(post_translation)
        if len(frame_images) != HENET_CAMERA_COUNT:
            raise ValueError("HENet frame has an unexpected camera count")
        per_frame_images.append(torch.stack(frame_images))
        sensor_to_ego.extend(frame_sensor_to_ego)
        intrinsics.extend(frame_intrinsics)
        post_rotations.extend(frame_post_rotations)
        post_translations.extend(frame_post_translations)
        pose = _world_pose(
            frame,
            origin_latitude_deg=origin_latitude,
            origin_longitude_deg=origin_longitude,
        ).astype(np.float32)
        ego_to_global.extend([pose] * HENET_CAMERA_COUNT)

    # HENet's image tensor is camera-major while its calibration tensors are
    # frame-major. Both orders are required by BEVDet4D.prepare_inputs.
    images = torch.stack(per_frame_images, dim=1).reshape(
        HENET_CAMERA_COUNT * len(ordered),
        3,
        output_height,
        output_width,
    ).unsqueeze(0)
    sensor_to_ego_tensor = torch.from_numpy(
        np.stack(sensor_to_ego)
    ).unsqueeze(0)
    ego_to_global_tensor = torch.from_numpy(
        np.stack(ego_to_global)
    ).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(np.stack(intrinsics)).unsqueeze(0)
    post_rotations_tensor = torch.from_numpy(
        np.stack(post_rotations)
    ).unsqueeze(0)
    post_translations_tensor = torch.from_numpy(
        np.stack(post_translations)
    ).unsqueeze(0)
    bda = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    return tuple(
        tensor.to(device=device, non_blocking=True)
        for tensor in (
            images,
            sensor_to_ego_tensor,
            ego_to_global_tensor,
            intrinsics_tensor,
            post_rotations_tensor,
            post_translations_tensor,
            bda,
        )
    )


def henet_inputs_for(
    current: PackedHENetFrame,
    history: Mapping[int, PackedHENetFrame],
    *,
    device: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build the short-term and long-term official HENet img_inputs tuples."""
    short_frames = temporal_frames_for(
        current,
        history,
        frame_offsets=HENET_SHORT_FRAME_OFFSETS,
    )
    long_frames = temporal_frames_for(
        current,
        history,
        frame_offsets=HENET_LONG_FRAME_OFFSETS,
    )
    return (
        _image_inputs_for(
            short_frames,
            output_height=HENET_INPUT_HEIGHT,
            output_width=HENET_INPUT_WIDTH,
            device=device,
        ),
        _image_inputs_for(
            long_frames,
            output_height=HENET_LONGTERM_INPUT_HEIGHT,
            output_width=HENET_LONGTERM_INPUT_WIDTH,
            device=device,
        ),
    )


def henet_segmentation_from_result(result: Any) -> np.ndarray:
    """Extract one official HENet sigmoid segmentation tensor."""
    if (
        not isinstance(result, Sequence)
        or len(result) != 1
        or not isinstance(result[0], Mapping)
    ):
        raise ValueError("HENet result has an unexpected envelope")
    segmentation = result[0].get("pts_seg")
    if segmentation is None or not hasattr(segmentation, "detach"):
        raise ValueError("HENet result is missing BEV segmentation")
    values = np.asarray(segmentation.detach().cpu(), dtype=np.float32)
    expected_shape = (len(HENET_SEGMENTATION_CLASS_NAMES), 200, 200)
    if values.shape != expected_shape:
        raise ValueError(
            "HENet BEV segmentation has shape "
            f"{list(values.shape)}, expected {list(expected_shape)}"
        )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(
        values > 1.0
    ):
        raise ValueError("HENet BEV segmentation must be finite in [0,1]")
    return values


def infer_henet_frame(
    model: Any,
    current: PackedHENetFrame,
    history: Mapping[int, PackedHENetFrame],
    *,
    device: Any,
) -> np.ndarray:
    """Run official HENet and return one ASOC `[8,450,300]` probability grid."""
    import torch

    short_inputs, long_inputs = henet_inputs_for(
        current,
        history,
        device=device,
    )
    placeholder = torch.zeros(
        (
            len(HENET_SEGMENTATION_CLASS_NAMES),
            200,
            200,
        ),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        result = model.simple_test(
            None,
            [{}],
            img=short_inputs,
            img_lt=long_inputs,
            gt_masks_bev=[placeholder],
        )
    return adapt_henet_segmentation(henet_segmentation_from_result(result))


def remember_history_frame(
    history: MutableMapping[int, PackedHENetFrame],
    frame: PackedHENetFrame,
) -> None:
    """Retain only the 4 second history consumed by HENet long-term input."""
    stale_before = frame.frame_index + min(HENET_LONG_FRAME_OFFSETS)
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


def load_official_henet(
    *,
    repository_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    device: Any,
) -> Any:
    """Load only the pinned HENet source revision and immutable checkpoint."""
    repository = Path(repository_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("HENet weight digest does not match provenance")
    git_head = repository / ".git" / "HEAD"
    if not git_head.exists():
        raise ValueError("HENet repository has no revision metadata")
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != HENET_REVISION:
        raise ValueError("HENet repository revision is not pinned")

    sys.path.insert(0, str(repository))
    try:
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model

        importlib.import_module("mmdet3d.models")
        config = Config.fromfile(
            str(repository / "configs" / "henet" / HENET_CONFIG_NAME)
        )
        config.model.pretrained = None
        config.model.train_cfg = None
        model = build_model(
            config.model,
            test_cfg=config.get("test_cfg"),
        )
        load_checkpoint(
            model,
            str(checkpoint),
            map_location="cpu",
        )
        # The Dashboard only publishes the native segmentation head. Avoid
        # detector metadata requirements and unnecessary detector computation.
        model.pts_bbox_head = None
        model.to(device)
        model.eval()
        return model
    finally:
        if sys.path and sys.path[0] == str(repository):
            sys.path.pop(0)
