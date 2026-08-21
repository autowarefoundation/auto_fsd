import io
import json
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from data_processing.geospatial import encode_pose
from Platform.pipelines.henet_runtime import (
    HENET_INPUT_HEIGHT,
    HENET_INPUT_WIDTH,
    HENET_LONGTERM_INPUT_HEIGHT,
    HENET_LONGTERM_INPUT_WIDTH,
    PackedHENetFrame,
    henet_inputs_for,
    henet_segmentation_from_result,
    iter_packed_henet_frames,
    remember_history_frame,
    temporal_frames_for,
    temporal_substitution_count,
)


def _projection() -> np.ndarray:
    matrix = np.repeat(np.eye(3, 4)[None], 6, axis=0)
    matrix[:, 0, 0] = 800.0
    matrix[:, 1, 1] = 810.0
    matrix[:, 0, 2] = 128.0
    matrix[:, 1, 2] = 129.0
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
) -> PackedHENetFrame:
    return PackedHENetFrame(
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


def test_packed_tar_reader_reorders_kitscenes_cameras_for_henet(tmp_path):
    tar_path = tmp_path / "train-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        uid = "sample-64"
        members = {
            **{
                f"cam_{camera}.jpg": _image_bytes((camera, 20, 30))
                for camera in range(6)
            },
            "meta.json": json.dumps(
                {
                    "sample_uid": uid,
                    "split_group_uid": "scene-a",
                    "frame_idx": 64,
                }
            ).encode(),
            "calib.json": json.dumps(
                {
                    "dataset": "kitscenes",
                    "projection": {
                        "type": "pinhole",
                        "matrix": _projection().tolist(),
                        "reference_frame": "top_lidar_flu",
                    },
                }
            ).encode(),
            "pose.npy": encode_pose(
                {
                    "latitude_deg": 35.0,
                    "longitude_deg": 139.0,
                    "heading_deg_cw_from_north": 0.0,
                    "timestamp_ns": 6_400_000_000,
                }
            ),
        }
        for suffix, payload in members.items():
            _add_tar_member(archive, f"{uid}.{suffix}", payload)

    frames = list(iter_packed_henet_frames(tar_path))

    assert len(frames) == 1
    assert [
        Image.open(io.BytesIO(payload)).getpixel((0, 0))[0]
        for payload in frames[0].image_payloads
    ] == [1, 0, 2, 4, 3, 5]
    np.testing.assert_allclose(
        frames[0].projection_ref_to_camera,
        _projection()[[1, 0, 2, 4, 3, 5]],
    )


def test_temporal_selection_uses_exact_two_hz_history_and_key_fallback():
    current = _frame(100)
    history = {
        60: _frame(60),
        90: _frame(90),
        95: _frame(95),
    }

    selected = temporal_frames_for(
        current,
        history,
        frame_offsets=(0, -5, -10, -15),
    )

    assert [
        frame.frame_index
        for frame in selected.values()
    ] == [100, 95, 90, 100]
    assert temporal_substitution_count(
        current,
        history,
        frame_offsets=(0, -5, -10, -15),
    ) == 1


def test_temporal_selection_rejects_cross_scene_history():
    with pytest.raises(ValueError, match="crossed"):
        temporal_frames_for(
            _frame(100),
            {95: _frame(95, episode_id="scene-b")},
            frame_offsets=(0, -5),
        )


def test_input_builder_matches_official_short_and_longterm_shapes():
    short_inputs, long_inputs = henet_inputs_for(
        _frame(100),
        {
            95: _frame(95, longitude=139.0001),
            90: _frame(90, longitude=139.0002),
        },
        device=torch.device("cpu"),
    )

    assert short_inputs[0].shape == (
        1,
        18,
        3,
        HENET_INPUT_HEIGHT,
        HENET_INPUT_WIDTH,
    )
    assert long_inputs[0].shape == (
        1,
        54,
        3,
        HENET_LONGTERM_INPUT_HEIGHT,
        HENET_LONGTERM_INPUT_WIDTH,
    )
    assert short_inputs[1].shape == (1, 18, 4, 4)
    assert long_inputs[1].shape == (1, 54, 4, 4)
    assert short_inputs[3].shape == (1, 18, 3, 3)
    assert long_inputs[5].shape == (1, 54, 3)


def test_result_extraction_preserves_official_probability_tensor():
    values = torch.full((3, 200, 200), 0.75)

    output = henet_segmentation_from_result([{"pts_seg": values}])

    assert output.shape == (3, 200, 200)
    assert output.dtype == np.float32
    assert output.max() == pytest.approx(0.75)


@pytest.mark.parametrize(
    "result",
    [
        [],
        [{}],
        [{"pts_seg": torch.zeros((3, 200, 199))}],
        [{"pts_seg": torch.full((3, 200, 200), 1.1)}],
    ],
)
def test_result_extraction_rejects_invalid_model_output(result):
    with pytest.raises(ValueError):
        henet_segmentation_from_result(result)


def test_history_cache_retains_the_longterm_window_only():
    history = {
        59: _frame(59),
        60: _frame(60),
        61: _frame(61),
    }

    remember_history_frame(history, _frame(100))

    assert set(history) == {60, 61, 100}
