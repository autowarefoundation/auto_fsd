import io
import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from data_processing.geospatial import encode_pose
from Platform.pipelines.bevformer_v2_runtime import (
    BEVFORMER_V2_IMAGE_HEIGHT,
    BEVFORMER_V2_IMAGE_WIDTH,
    PackedBEVFormerFrame,
    bevformer_metadata_for,
    detections_from_bevformer_result,
    iter_packed_bevformer_frames,
    preprocess_packed_images,
    remember_history_frame,
    temporal_frames_for,
)


def _projection() -> np.ndarray:
    matrix = np.repeat(np.eye(3, 4)[None], 6, axis=0)
    matrix[:, 0, 2] = 128.0
    matrix[:, 1, 2] = 128.0
    return matrix


def _image_bytes(color=(10, 20, 30), size=(256, 256)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _frame(
    frame_index: int,
    *,
    episode_id: str = "scene-a",
    longitude: float = 139.0,
) -> PackedBEVFormerFrame:
    return PackedBEVFormerFrame(
        sample_uid=f"sample-{frame_index}",
        episode_id=episode_id,
        frame_index=frame_index,
        timestamp_ns=frame_index * 100_000_000,
        image_payloads=(_image_bytes(),) * 6,
        projection_ref_to_camera=_projection(),
        pose={
            "latitude_deg": 35.0,
            "longitude_deg": longitude,
            "heading_deg_cw_from_north": 0.0,
            "timestamp_ns": frame_index * 100_000_000,
            "gps_accuracy_m": float("nan"),
        },
    )


def _add_tar_member(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_packed_tar_reader_preserves_six_camera_and_pose_contract(tmp_path):
    tar_path = tmp_path / "train-000000.tar"
    projection = {
        "type": "pinhole",
        "matrix": _projection().tolist(),
        "reference_frame": "top_lidar_flu",
        "ground_z_m": 0.0,
    }
    with tarfile.open(tar_path, "w") as archive:
        for frame_index in (64, 65):
            uid = f"sample-{frame_index}"
            members = {
                **{
                    f"cam_{camera}.jpg": _image_bytes(
                        (camera, frame_index % 255, 30)
                    )
                    for camera in range(6)
                },
                "meta.json": json.dumps(
                    {
                        "sample_uid": uid,
                        "split_group_uid": "scene-a",
                        "frame_idx": frame_index,
                    }
                ).encode(),
                "calib.json": json.dumps(
                    {
                        "dataset": "kitscenes",
                        "geometry_type": "pinhole",
                        "projection": projection,
                    }
                ).encode(),
                "pose.npy": encode_pose(
                    {
                        "latitude_deg": 35.0,
                        "longitude_deg": 139.0,
                        "heading_deg_cw_from_north": 0.0,
                        "timestamp_ns": frame_index * 100_000_000,
                    }
                ),
            }
            for suffix, payload in members.items():
                _add_tar_member(archive, f"{uid}.{suffix}", payload)

    frames = list(iter_packed_bevformer_frames(tar_path))

    assert [frame.frame_index for frame in frames] == [64, 65]
    assert frames[0].episode_id == "scene-a"
    assert frames[1].timestamp_ns == 6_500_000_000
    assert len(frames[0].image_payloads) == 6
    assert [
        Image.open(io.BytesIO(payload)).getpixel((0, 0))[0]
        for payload in frames[0].image_payloads
    ] == [0, 2, 1, 3, 4, 5]
    np.testing.assert_allclose(
        frames[0].projection_ref_to_camera,
        _projection()[[0, 2, 1, 3, 4, 5]],
    )


def test_temporal_selection_uses_exact_two_hz_rows_without_substitution():
    current = _frame(100)
    history = {
        65: _frame(65),
        66: _frame(66),
        70: _frame(70),
        95: _frame(95),
    }

    selected = temporal_frames_for(current, history)

    assert list(selected) == [-7, -6, -1, 0]
    assert 66 not in [frame.frame_index for frame in selected.values()]


def test_temporal_selection_rejects_cross_scene_history():
    with pytest.raises(ValueError, match="crossed"):
        temporal_frames_for(
            _frame(100),
            {95: _frame(95, episode_id="scene-b")},
        )


def test_metadata_scales_intrinsics_and_aligns_history_to_current():
    current = _frame(100, longitude=139.0001)
    history = _frame(95, longitude=139.0)

    metadata = bevformer_metadata_for(
        {-1: history, 0: current},
        box_type_3d=object,
    )

    history_projection = metadata[-1]["lidar2img"][0]
    current_projection = metadata[0]["lidar2img"][0]
    assert history_projection.shape == (4, 4)
    assert current_projection[0, 0] == pytest.approx(2.5)
    assert current_projection[0, 2] == pytest.approx(320.0)
    assert current_projection[1, 2] == pytest.approx(128.0)
    assert metadata[-1]["lidaradj2lidarcurr"] is not None
    assert metadata[0]["lidaradj2lidarcurr"] is None
    assert metadata[0]["timestamp"] == pytest.approx(10.0)


def test_metadata_uses_decoded_packed_image_dimensions():
    frame = replace(
        _frame(10),
        image_payloads=(_image_bytes(size=(8, 4)),) * 6,
    )

    metadata = bevformer_metadata_for(
        {0: frame},
        box_type_3d=object,
    )[0]

    assert metadata["lidar2img"][0][0, 0] == pytest.approx(80.0)
    assert metadata["lidar2img"][0][1, 1] == pytest.approx(64.0)
    np.testing.assert_allclose(
        metadata["scale_factor"],
        [80.0, 64.0, 80.0, 64.0],
    )


def test_metadata_rejects_mixed_camera_dimensions():
    frame = replace(
        _frame(10),
        image_payloads=(
            _image_bytes(size=(8, 4)),
            *(_image_bytes(size=(4, 4)) for _ in range(5)),
        ),
    )

    with pytest.raises(ValueError, match="equal packed camera dimensions"):
        bevformer_metadata_for({0: frame}, box_type_3d=object)


def test_preprocessing_is_bgr_mean_subtracted_at_official_shape():
    frame = _frame(64)

    images = preprocess_packed_images(frame)

    assert tuple(images.shape) == (
        6,
        3,
        BEVFORMER_V2_IMAGE_HEIGHT,
        BEVFORMER_V2_IMAGE_WIDTH,
    )
    assert images[0, 0, 0, 0].item() == pytest.approx(30.0 - 103.53)
    assert images[0, 1, 0, 0].item() == pytest.approx(20.0 - 116.28)
    assert images[0, 2, 0, 0].item() == pytest.approx(10.0 - 123.675)


def test_official_result_conversion_keeps_box_dimensions_and_yaw():
    class Boxes:
        tensor = torch.tensor(
            [[4.0, -2.0, 0.5, 4.8, 2.1, 1.7, 0.25]],
            dtype=torch.float32,
        )

    detections = detections_from_bevformer_result(
        [
            {
                "pts_bbox": {
                    "boxes_3d": Boxes(),
                    "scores_3d": torch.tensor([0.8]),
                    "labels_3d": torch.tensor([3]),
                }
            }
        ]
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "car"
    assert detection.center_x_m == pytest.approx(4.0)
    assert detection.center_y_m == pytest.approx(-2.0)
    assert detection.length_m == pytest.approx(4.8)
    assert detection.width_m == pytest.approx(2.1)
    assert detection.yaw_rad == pytest.approx(0.25)


def test_history_cache_is_bounded_to_the_t8_window():
    history = {64: _frame(64), 65: _frame(65)}

    remember_history_frame(history, _frame(100))

    assert 64 not in history
    assert set(history) == {65, 100}
