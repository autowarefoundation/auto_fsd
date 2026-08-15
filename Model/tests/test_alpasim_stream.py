"""Unit and parity tests for AlpaSim stream parser and driver plugin.

Guards tensor shapes, dtypes, ImageNet normalization, egomotion buffer formatting,
camera projection math, and driver plugin contracts between live streaming input
(PredictionInput) and offline KitScenes pre-extracted datasets.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from PIL import Image

_ALPASIM_DRIVER_DIR = Path(__file__).resolve().parents[1] / "plugins" / "alpasim_driver"
if str(_ALPASIM_DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(_ALPASIM_DRIVER_DIR))

from alpasim_autoe2e.config import AutoE2EAlpaSimConfig  # noqa: E402
from alpasim_autoe2e.plugin import (  # noqa: E402
    AutoE2EDriver,
    ModelPrediction,
    PredictionInput as PluginPredictionInput,
)
from alpasim_autoe2e.parser import (  # noqa: E402
    AlpasimStreamParser,
    PredictionInput,
)
PARSER_CAMERA_NAMES = AutoE2EAlpaSimConfig(checkpoint_path='dummy.ckpt').camera_names

from data_parsing.pre_extracted import (  # noqa: E402
    _VISUAL_HISTORY_DIM,
    _decode_image as _decode_pre_extracted_image,
)


class MockAutoE2EModel(torch.nn.Module):
    def forward(self, **kwargs):
        return {
            "trajectory_points": torch.zeros((1, 64, 2)),
            "headings": torch.zeros((1, 64))
        }

torch.serialization.add_safe_globals([MockAutoE2EModel])

@pytest.fixture
def dummy_checkpoint(tmp_path) -> str:
    ckpt_path = tmp_path / "dummy_random.ckpt"
    torch.save(MockAutoE2EModel(), ckpt_path)
    return str(ckpt_path)

@pytest.fixture
def sample_rgb_images() -> Dict[str, Image.Image]:

    """Generate 7 synthetic PIL images for KitScenes camera topology.

    Returns a mapping from KitScenes camera names to 256x256 RGB images.
    """
    images: Dict[str, Image.Image] = {}
    for idx, cam_name in enumerate(PARSER_CAMERA_NAMES):
        color = (idx * 30, (idx * 50) % 255, (255 - idx * 30) % 255)
        images[cam_name] = Image.new("RGB", (256, 256), color)
    return images


@pytest.fixture
def sample_numpy_frames() -> Dict[str, np.ndarray]:
    """Generate 7 synthetic uint8 numpy arrays for KitScenes camera topology.

    Returns a mapping from KitScenes camera names to ``(256, 256, 3)`` arrays.
    """
    frames: Dict[str, np.ndarray] = {}
    for idx, cam_name in enumerate(PARSER_CAMERA_NAMES):
        array = np.full((256, 256, 3), (idx * 35) % 256, dtype=np.uint8)
        frames[cam_name] = array
    return frames


@pytest.fixture
def sample_jpeg_bytes(sample_rgb_images: Dict[str, Image.Image]) -> Dict[str, bytes]:
    """Generate 7 synthetic JPEG byte blobs for KitScenes camera topology.

    Returns a mapping from KitScenes camera names to JPEG bytes.
    """
    encoded: Dict[str, bytes] = {}
    for cam_name, img in sample_rgb_images.items():
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded[cam_name] = buf.getvalue()
    return encoded


@pytest.fixture
def valid_prediction_input(
    sample_rgb_images: Dict[str, Image.Image],
) -> PredictionInput:
    """Return a valid happy-path dict ``PredictionInput`` payload."""
    return {
        "cameras": sample_rgb_images,
        "speed": 12.5,
        "acceleration": 0.5,
        "command": 1,
        "ego_pose": (0.0, 0.0, 0.0),
    }


@pytest.fixture
def stream_sequence_10hz(
    sample_numpy_frames: Dict[str, np.ndarray],
) -> List[PredictionInput]:
    """Generate a 70-frame sequence (7 seconds at 10 Hz) of streaming inputs.

    Simulates realistic ego motion accelerating from 0.0 to 14.0 m/s.
    """
    sequence: List[PredictionInput] = []
    for step in range(70):
        speed = float(step * 0.2)
        acceleration = 0.2
        sequence.append(
            {
                "cameras": sample_numpy_frames,
                "speed": speed,
                "acceleration": acceleration,
                "command": 1,
        "ego_pose": (0.0, 0.0, 0.0),
            }
        )
    return sequence



def mock_parser_deps(parser, navigation_map=None, scene_path=None):
    class MockRaster:
        route_mask = np.zeros((2, 256, 256), dtype=np.float32)
        route_valid = True
    class MockRasterizer:
        def render(self, nav_map, route, live_pose):
            return MockRaster()
    parser.rasterizer = MockRasterizer()
    parser.route = True
    if navigation_map is not None:
        parser.navigation_map = navigation_map
    if scene_path is not None:
        parser.scene_path = scene_path
    return parser

class TestAlpasimStreamParserFixturesAndBasicShape:
    """Verify basic shape, dtype, and input decoding of AlpasimStreamParser."""

    def test_happy_path_tensor_shapes(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify tensor shapes produced by ``parse_observation``.

        Expected shapes:
          - ``camera_tiles``: ``[1, 7, 3, 256, 256]``
          - ``egomotion_history``: ``[1, 256]``
          - ``visual_history``: ``[1, 896]``
          - ``map_context``: ``[1, 3, 256, 256]``
          - ``route_mask``: ``[1, 2, 256, 256]``
          - ``map_valid``: ``[1]``
          - ``route_valid``: ``[1]``
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        tensors = parser.parse_observation(valid_prediction_input)

        assert tensors["camera_tiles"].shape == (1, 7, 3, 256, 256)
        assert tensors["egomotion_history"].shape == (1, 256)
        assert tensors["visual_history"].shape == (1, _VISUAL_HISTORY_DIM)
        assert tensors["map_context"].shape == (1, 3, 256, 256)
        assert tensors["route_mask"].shape == (1, 2, 256, 256)
        assert tensors["map_valid"].shape == (1,)
        assert tensors["route_valid"].shape == (1,)

    def test_happy_path_tensor_dtypes(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify tensor data types produced by ``parse_observation``.

        Float Tensors must be ``torch.float32``; validity flags must be ``torch.bool``.
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        tensors = parser.parse_observation(valid_prediction_input)

        assert tensors["camera_tiles"].dtype == torch.float32
        assert tensors["egomotion_history"].dtype == torch.float32
        assert tensors["visual_history"].dtype == torch.float32
        assert tensors["map_context"].dtype == torch.float32
        assert tensors["route_mask"].dtype == torch.float32
        assert tensors["map_valid"].dtype == torch.bool
        assert tensors["route_valid"].dtype == torch.bool

    def test_input_types_pil_numpy_bytes(
        self,
        sample_rgb_images: Dict[str, Image.Image],
        sample_numpy_frames: Dict[str, np.ndarray],
        sample_jpeg_bytes: Dict[str, bytes],
    ) -> None:
        """Verify ``_decode_image`` supports PIL Image, numpy array, and JPEG bytes."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)

        t1 = parser.parse_observation(
            {"cameras": sample_rgb_images, "speed": 5.0, "acceleration": 0.0, "command": 0, "ego_pose": (0.0, 0.0, 0.0)}
        )["camera_tiles"]
        t2 = parser.parse_observation(
            {"cameras": sample_numpy_frames, "speed": 5.0, "acceleration": 0.0, "command": 0, "ego_pose": (0.0, 0.0, 0.0)}
        )["camera_tiles"]
        t3 = parser.parse_observation(
            {"cameras": sample_jpeg_bytes, "speed": 5.0, "acceleration": 0.0, "command": 0, "ego_pose": (0.0, 0.0, 0.0)}
        )["camera_tiles"]

        assert t1.shape == (1, 7, 3, 256, 256)
        assert t2.shape == (1, 7, 3, 256, 256)
        assert t3.shape == (1, 7, 3, 256, 256)

    def test_route_mask_rendering(self, sample_rgb_images: Dict[str, Image.Image]) -> None:
        """Verify the route mask logic interacts correctly with the rasterizer."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        
        tensors = parser.parse_observation(
            {"cameras": sample_rgb_images, "speed": 0.0, "acceleration": 0.0, "command": 0, "ego_pose": (0.0, 0.0, 0.0)}
        )
        mask = tensors["route_mask"][0, 0]
        assert mask.shape == (256, 256)


class TestOfflineKitScenesParity:
    """Parity assertions between AlpasimStreamParser and offline KitScenes datasets."""

    def test_image_normalization_parity(
        self, sample_jpeg_bytes: Dict[str, bytes]
    ) -> None:
        """Verify stream parser image normalization equals offline ``pre_extracted`` decode.

        Both paths run ImageNet Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
        Tolerance: bit-for-bit or ``atol=1e-5`` since floating-point ops are deterministic.
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        cam_key = PARSER_CAMERA_NAMES[0]
        jpeg_data = sample_jpeg_bytes[cam_key]

        live_tile = parser._decode_image(jpeg_data)
        offline_tile = _decode_pre_extracted_image(jpeg_data)

        assert torch.allclose(live_tile, offline_tile, atol=1e-5, rtol=1e-5), (
            "Live stream image decode must produce tensors identical to offline decode."
        )

    def test_image_normalization_range_mean_std(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify decoded image values follow ImageNet mean/std normalization ranges.

        Normalized values for RGB [0, 255] must lie approximately in [-2.12, 2.64].
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        cam_key = PARSER_CAMERA_NAMES[0]

        tile = parser._decode_image(sample_rgb_images[cam_key])
        assert tile.min() >= -2.5
        assert tile.max() <= 3.0

        # Uniform gray image (128, 128, 128) -> ToTensor = ~0.50196
        gray_img = Image.new("RGB", (256, 256), (128, 128, 128))
        gray_tile = parser._decode_image(gray_img)
        # Channel 0: (0.50196 - 0.485) / 0.229 ≈ 0.074
        assert torch.isclose(
            gray_tile[0].mean(), torch.tensor(0.074, dtype=torch.float32), atol=1e-2
        )

    def test_egomotion_history_formatting_and_dimensions(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify egomotion history layout matches offline 64-step x 4-signal format.

        Offline egomotion vector has shape ``(256,)`` (64 timesteps x 4 signals).
        Signals per timestep: ``[speed, acceleration, yaw_rate, curvature]``.
        The live parser emits ``[1, 256]``.
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        tensors = parser.parse_observation(valid_prediction_input)
        ego_hist = tensors["egomotion_history"]

        assert ego_hist.shape == (1, 256)
        assert ego_hist.dtype == torch.float32

        # Check last timestep in the history buffer (indices 252..255)
        last_timestep_ego = ego_hist[0, -4:]
        assert torch.isclose(
            last_timestep_ego[0], torch.tensor(12.5, dtype=torch.float32)
        )
        assert torch.isclose(
            last_timestep_ego[1], torch.tensor(0.5, dtype=torch.float32)
        )
        assert last_timestep_ego[2].item() == 0.0  # yaw_rate default
        assert last_timestep_ego[3].item() == 0.0  # curvature default

    def test_egomotion_10hz_sequence_sliding_window(
        self, stream_sequence_10hz: List[PredictionInput]
    ) -> None:
        """Verify egomotion deque buffer accumulates a 64-step sliding window at 10 Hz.

        After feeding 70 frames (7 s), the buffer must hold step 6 to step 69.
        Step 0 (speed 0.0) must be evicted.
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)

        last_tensors: Dict[str, torch.Tensor] = {}
        for observation in stream_sequence_10hz:
            last_tensors = parser.parse_observation(observation)

        ego_hist = last_tensors["egomotion_history"][0].reshape(64, 4)

        # Step 69 speed = 69 * 0.2 = 13.8 m/s
        expected_latest_speed = 69 * 0.2
        actual_latest_speed = ego_hist[-1, 0].item()
        assert torch.isclose(
            torch.tensor(actual_latest_speed),
            torch.tensor(expected_latest_speed),
            atol=1e-4,
        )

        # Earliest step in 64-step window is step 6 (6 * 0.2 = 1.2 m/s)
        expected_oldest_speed = 6 * 0.2
        actual_oldest_speed = ego_hist[0, 0].item()
        assert torch.isclose(
            torch.tensor(actual_oldest_speed),
            torch.tensor(expected_oldest_speed),
            atol=1e-4,
        )

    def test_camera_topology_parity(self) -> None:
        """Verify the AlpaSim stream parser topology matches the KIT offline topology.
        
        This parity check ensures that the names and order of the 7 camera streams
        expected by the runtime parser perfectly match the dataset training pipeline.
        """
        # Hardcoded contract representing the offline training dataset topology
        # to avoid CI dependency issues with the 'kitscenes' package.
        EXPECTED_KITSCENES_TOPOLOGY = [
            "camera_base_front_center",
            "camera_ring_front",
            "camera_ring_front_left",
            "camera_ring_front_right",
            "camera_ring_rear",
            "camera_ring_rear_left",
            "camera_ring_rear_right",
        ]
        
        assert PARSER_CAMERA_NAMES == EXPECTED_KITSCENES_TOPOLOGY, (
            f"Runtime parser camera topology MUST match the offline training topology.\n"
            f"Parser:  {PARSER_CAMERA_NAMES}\n"
            f"Offline: {EXPECTED_KITSCENES_TOPOLOGY}"
        )
        assert len(PARSER_CAMERA_NAMES) == 7, "AutoE2E expects exactly 7 cameras."



class TestEdgeCasesAndDiscrepancies:
    """Test edge cases and document implementation discrepancies found during investigation."""

    def test_edge_case_missing_camera_frame(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify behavior when a camera frame is missing from PredictionInput.

        If a camera is missing, parser should raise a ValueError.
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        partial_cams = dict(sample_rgb_images)
        missing_cam = "camera_ring_rear_left"
        del partial_cams[missing_cam]

        import pytest
        with pytest.raises(ValueError, match=f"Missing camera frame for {missing_cam}"):
            parser.parse_observation(
                {"cameras": partial_cams, "speed": 10.0, "acceleration": 0.0, "command": 1, "ego_pose": (0.0, 0.0, 0.0)}
            )

    def test_edge_case_out_of_range_ego_values(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify parser handles extreme / negative speed and acceleration values."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)

        tensors = parser.parse_observation(
            {
                "cameras": sample_rgb_images,
                "speed": -15.0,
                "acceleration": 250.0,
                "command": -1,
                "ego_pose": (0.0, 0.0, 0.0),
            }
        )

        last_ego = tensors["egomotion_history"][0, -4:]
        assert last_ego[0].item() == -15.0
        assert last_ego[1].item() == 250.0

    def test_edge_case_malformed_command(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify parser behavior when command field is non-integer or unexpected type."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)

        input_data: Dict[str, object] = {
            "cameras": sample_rgb_images,
            "speed": 0.0,
            "acceleration": 0.0,
            "command": None,
            "ego_pose": (0.0, 0.0, 0.0),
        }
        tensors = parser.parse_observation(input_data)  # type: ignore[arg-type]
        assert tensors["camera_tiles"].shape == (1, 7, 3, 256, 256)

    def test_config_camera_names_match_parser(
        self, sample_rgb_images: Dict[str, Image.Image]
    ) -> None:
        """Verify AutoE2EAlpaSimConfig.camera_names match AlpasimStreamParser.CAMERA_NAMES.

        Passing inputs keyed by config camera names should successfully populate frames.
        """
        config = AutoE2EAlpaSimConfig(checkpoint_path='dummy_random.ckpt')
        config_cams = config.camera_names  # ['cam_front', 'cam_front_left', ...]

        assert list(config_cams) == list(PARSER_CAMERA_NAMES), (
            "Config camera names should match parser camera names."
        )

        # Build prediction input using config's camera names
        cams_with_config_keys = {
            name: img for name, img in zip(config_cams, sample_rgb_images.values())
        }

        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        tensors = parser.parse_observation(
            {"cameras": cams_with_config_keys, "speed": 10.0, "acceleration": 0.0, "command": 1, "ego_pose": (0.0, 0.0, 0.0)}
        )

        # Frames should not be empty since the camera names match
        visual_tiles = tensors["camera_tiles"]
        assert not (visual_tiles == 0.0).all(), (
            "Frames should not be empty since the camera names match."
        )

    def test_camera_params_present_in_stream_parser(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify AlpasimStreamParser output dictionary contains 'camera_params'.

        It should provide dummy camera parameters matching the expected shape.
        """
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser = mock_parser_deps(parser)
        tensors = parser.parse_observation(valid_prediction_input)

        assert "camera_params" in tensors, (
            "AlpasimStreamParser should emit camera_params in output dict."
        )
        assert tensors["camera_params"].shape == (1, 7, 3, 4)
        assert tensors["camera_params"].dtype == torch.float32




class TestAlpasimDriverPlugin:
    """Verify AlpaSim driver plugin AutoE2EDriver interface and prediction return."""

    def test_driver_plugin_initialization(self, dummy_checkpoint: str) -> None:
        """Verify AutoE2EDriver initializes parser and device correctly."""
        driver = AutoE2EDriver(model_checkpoint=dummy_checkpoint, allow_mock=True)
        assert isinstance(driver.parser, AlpasimStreamParser)
        assert isinstance(driver.device, torch.device)

    def test_driver_plugin_predict_happy_path(
        self, sample_rgb_images: Dict[str, Image.Image],
        dummy_checkpoint: str
    ) -> None:
        """Verify AutoE2EDriver.predict accepts PluginPredictionInput and returns ModelPrediction.

        Expected output:
          - ``trajectory_points``: numpy array of shape ``(64, 2)`` and float32.
          - ``headings``: numpy array of shape ``(64,)`` and float32.
        """
        driver = AutoE2EDriver(model_checkpoint=dummy_checkpoint, allow_mock=True)
        mock_parser_deps(driver.parser)
        pred_input = PluginPredictionInput(
            camera_images=sample_rgb_images,
            speed=8.0,
            acceleration=0.1,
            command=1,
            ego_pose_history=[
                type("MockPoseAtTime", (), {"timestamp_us": 0, "pose": type("MockPose", (), {"quat": type("MockQuat", (), {"w":1.0, "x":0.0, "y":0.0, "z":0.0})(), "x":0.0, "y":0.0, "z":0.0})()})(),
                type("MockPoseAtTime", (), {"timestamp_us": 1, "pose": type("MockPose", (), {"quat": type("MockQuat", (), {"w":1.0, "x":0.0, "y":0.0, "z":0.0})(), "x":0.0, "y":0.0, "z":0.0})()})(),
            ],
            inference_seed=0,
        )

        result = driver.predict(pred_input)

        assert isinstance(result, ModelPrediction)
        assert hasattr(result, "trajectory_xy") or hasattr(result, "trajectory_points")
        traj = getattr(result, "trajectory_xy", getattr(result, "trajectory_points", None))
        assert isinstance(traj, np.ndarray)
        assert isinstance(result.headings, np.ndarray)
        assert traj.shape == (64, 2)
        assert result.headings.shape == (64,)
        assert traj.dtype == np.float32
        assert result.headings.dtype == np.float32

    def test_driver_plugin_strict_mock_disallowed(self) -> None:
        """Verify that initializing with allow_mock=False fails fast when using mock dependencies."""
        from alpasim_autoe2e.plugin import IS_MOCK_MODE
        if IS_MOCK_MODE:
            with pytest.raises(ImportError, match="allow_mock=True"):
                AutoE2EDriver(model_checkpoint="nonexistent.ckpt", allow_mock=False)

    def test_dynamic_camera_list(self) -> None:
        """Verify the parser and driver work correctly with an arbitrary list of camera names."""
        custom_cameras = ["camera_ring_front_left", "camera_ring_front_right"]
        parser = AlpasimStreamParser(camera_names=custom_cameras)
        parser = mock_parser_deps(parser)
        
        # Build fake observation
        from PIL import Image
        fake_images = {
            "camera_ring_front_left": Image.new("RGB", (256, 256), (255, 0, 0)),
            "camera_ring_front_right": Image.new("RGB", (256, 256), (0, 255, 0))
        }
        obs = {
            "cameras": fake_images,
            "speed": 5.0,
            "acceleration": 1.0,
            "command": 1,
            "ego_pose": (0.0, 0.0, 0.0),
        }
        
        tensors = parser.parse_observation(obs)
        assert tensors["camera_tiles"].shape == (1, 2, 3, 256, 256)
        assert tensors["camera_params"].shape == (1, 2, 3, 4)
        
        # Test driver fallback init with custom cameras
        driver = AutoE2EDriver(model_checkpoint="MOCK", allow_mock=True, camera_ids=custom_cameras)
        mock_parser_deps(driver.parser)
        assert len(driver.camera_ids) == 2
        # Mock prediction output
        fake_history = [
                type("MockPoseAtTime", (), {"timestamp_us": 0, "pose": type("MockPose", (), {"quat": type("MockQuat", (), {"w":1.0, "x":0.0, "y":0.0, "z":0.0})(), "x":0.0, "y":0.0, "z":0.0})()})(),
                type("MockPoseAtTime", (), {"timestamp_us": 1, "pose": type("MockPose", (), {"quat": type("MockQuat", (), {"w":1.0, "x":0.0, "y":0.0, "z":0.0})(), "x":0.0, "y":0.0, "z":0.0})()})(),
        ]
        pred = driver.predict(PluginPredictionInput(camera_images=fake_images, speed=5.0, acceleration=1.0, command=1, ego_pose_history=fake_history, inference_seed=0))
        assert pred.trajectory_xy.shape == (64, 2)

    def test_dynamic_yaw_rate_and_curvature(self, dummy_checkpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify yaw_rate and curvature are computed dynamically from ego_pose_history."""
        import math
        from dataclasses import dataclass

        @dataclass
        class MockQuat:
            w: float
            x: float
            y: float
            z: float

        @dataclass
        class MockPose:
            quat: MockQuat
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0

        @dataclass
        class MockPoseAtTime:
            timestamp_us: int
            pose: MockPose

        # A pure yaw rotation has w = cos(theta/2), z = sin(theta/2), x=0, y=0
        # Let's say prev_yaw = 0.0, curr_yaw = 0.1. dt = 1 second.
        prev_quat = MockQuat(w=math.cos(0.0 / 2.0), x=0.0, y=0.0, z=math.sin(0.0 / 2.0))
        curr_quat = MockQuat(w=math.cos(0.1 / 2.0), x=0.0, y=0.0, z=math.sin(0.1 / 2.0))

        prev_pose = MockPoseAtTime(timestamp_us=1000000, pose=MockPose(quat=prev_quat))
        curr_pose = MockPoseAtTime(timestamp_us=2000000, pose=MockPose(quat=curr_quat))

        driver = AutoE2EDriver(model_checkpoint=dummy_checkpoint, allow_mock=True)

        captured_input = {}
        def mock_parse_observation(input_dict):
            nonlocal captured_input
            captured_input = input_dict
            # Return dummy tensors to prevent failure
            return {
                "camera_tiles": torch.zeros((1, 7, 3, 256, 256)),
                "camera_params": torch.zeros((1, 7, 3, 4)),
            }
        
        monkeypatch.setattr(driver.parser, "parse_observation", mock_parse_observation)

        pred_input = PluginPredictionInput(
            camera_images={},
            speed=10.0,
            acceleration=0.0,
            command=1,
            ego_pose_history=[prev_pose, curr_pose],
            inference_seed=0,
        )

        driver.predict(pred_input)

        assert "yaw_rate" in captured_input
        assert "curvature" in captured_input
        assert captured_input["yaw_rate"] == pytest.approx(0.1, abs=1e-5)
        # curvature = yaw_rate / max(speed, 0.1) -> 0.1 / 10.0 = 0.01
        assert captured_input["curvature"] == pytest.approx(0.01, abs=1e-5)


class TestDynamicBevMapGeneration:
    """Verify dynamic BEV map tile rasterization and error handling in AlpasimStreamParser."""

    def test_dynamic_bev_map_tile_generation_success(
        self, valid_prediction_input: PredictionInput, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify generate_bev_map_tile is dynamically invoked when scene_path and navigation_map exist."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        scene_dir = tmp_path / "mock_val_scene"
        scene_dir.mkdir()
        mock_parser_deps(parser, navigation_map=object(), scene_path=scene_dir)

        synthetic_tile = np.zeros((256, 256, 3), dtype=np.uint8)
        synthetic_tile[10, 20] = [255, 128, 64]
        captured_kwargs = {}

        def mock_generate_bev_map_tile(**kwargs):
            captured_kwargs.update(kwargs)
            return synthetic_tile

        monkeypatch.setattr(
            "data_parsing.kit_scenes.map.generate_bev_map_tile",
            mock_generate_bev_map_tile,
        )

        valid_prediction_input["ego_pose"] = (15.5, -20.25, 1.57)
        tensors = parser.parse_observation(valid_prediction_input)

        assert captured_kwargs == {
            "scene_path": scene_dir,
            "ego_x": 15.5,
            "ego_y": -20.25,
            "ego_yaw": 1.57,
            "canvas_size": 256,
        }
        assert tensors["map_context"].shape == (1, 3, 256, 256)
        assert tensors["map_context"].dtype == torch.float32
        assert tensors["map_valid"].item() is True
        # Check channel permutation: uint8 HWC -> float CHW
        assert torch.allclose(
            tensors["map_context"][0, :, 10, 20],
            torch.tensor([255.0, 128.0, 64.0], dtype=torch.float32),
        )

    def test_dynamic_bev_map_returns_none_raises_runtime_error(
        self, valid_prediction_input: PredictionInput, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify fail-loud RuntimeError is raised when generate_bev_map_tile returns None."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        scene_dir = tmp_path / "mock_corrupt_scene"
        scene_dir.mkdir()
        mock_parser_deps(parser, navigation_map=object(), scene_path=scene_dir)

        monkeypatch.setattr(
            "data_parsing.kit_scenes.map.generate_bev_map_tile",
            lambda **kwargs: None,
        )

        with pytest.raises(
            RuntimeError,
            match="generate_bev_map_tile failed and returned None. Ensure the scene map is valid and Lanelet2 is able to extract vectors.",
        ):
            parser.parse_observation(valid_prediction_input)

    def test_map_context_zero_when_no_scene_path_or_navigation_map(
        self, valid_prediction_input: PredictionInput, tmp_path: Path
    ) -> None:
        """Verify map_context defaults to zero tensor and map_valid flag is False when no scene_path."""
        # Case 1: scene_path=None, navigation_map=None
        parser1 = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        mock_parser_deps(parser1)
        assert parser1.scene_path is None
        assert parser1.navigation_map is None

        tensors1 = parser1.parse_observation(valid_prediction_input)
        assert tensors1["map_context"].shape == (1, 3, 256, 256)
        assert torch.count_nonzero(tensors1["map_context"]) == 0
        assert tensors1["map_valid"].item() is False

        # Case 2: scene_path provided, but navigation_map is None
        parser2 = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        scene_dir = tmp_path / "scene_without_nav"
        scene_dir.mkdir()
        mock_parser_deps(parser2, navigation_map=None, scene_path=scene_dir)

        tensors2 = parser2.parse_observation(valid_prediction_input)
        assert tensors2["map_context"].shape == (1, 3, 256, 256)
        assert torch.count_nonzero(tensors2["map_context"]) == 0
        assert tensors2["map_valid"].item() is False

    def test_missing_ego_pose_raises_value_error(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify ValueError is raised when ego_pose is missing from observation."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        mock_parser_deps(parser)
        valid_prediction_input["ego_pose"] = None

        with pytest.raises(
            ValueError,
            match="Ego pose is missing from the observation, cannot render route mask.",
        ):
            parser.parse_observation(valid_prediction_input)

    def test_missing_rasterizer_or_route_raises_import_error(
        self, valid_prediction_input: PredictionInput
    ) -> None:
        """Verify ImportError is raised when rasterizer or route is None."""
        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES)
        parser.rasterizer = None
        parser.route = None

        with pytest.raises(
            ImportError,
            match="The rasterizer and/or route are missing, cannot render route mask.",
        ):
            parser.parse_observation(valid_prediction_input)

    def test_parser_init_with_kitscenes_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify parser.__init__ resolves scene_path from KITSCENES_ROOT for val and train splits."""
        kitscenes_root = tmp_path / "kitscenes"
        val_scene = kitscenes_root / "data" / "val" / "scene_val_001"
        val_scene.mkdir(parents=True)
        poses_file = val_scene / "poses.txt"
        # Multi-row pose data: timestamp, x, y, z, qx, qy, qz, qw
        np.savetxt(
            poses_file,
            [
                [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, 1.1, 2.1, 3.1, 0.0, 0.0, 0.0, 1.0],
            ],
        )

        monkeypatch.setenv("KITSCENES_ROOT", str(kitscenes_root))

        mock_nav_called = False
        class MockNavResult:
            navigation_map = object()
            route = object()

        def mock_build_scene_navigation(**kwargs):
            nonlocal mock_nav_called
            mock_nav_called = True
            return MockNavResult()

        monkeypatch.setattr(
            "navigation.rasterizer.NativeNavigationRasterizer",
            lambda: object(),
        )
        monkeypatch.setattr(
            "data_parsing.kit_scenes.navigation.build_scene_navigation",
            mock_build_scene_navigation,
        )

        parser = AlpasimStreamParser(camera_names=PARSER_CAMERA_NAMES, scene_id="scene_val_001")
        assert parser.scene_path == val_scene
        assert mock_nav_called is True
        assert parser.navigation_map is not None
        assert parser.route is not None

    def test_driver_predict_with_dynamic_bev_map(
        self,
        sample_rgb_images: Dict[str, Image.Image],
        dummy_checkpoint: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify AutoE2EDriver.predict end-to-end when dynamic BEV map tile rasterization is active."""
        driver = AutoE2EDriver(model_checkpoint=dummy_checkpoint, allow_mock=True)
        scene_dir = tmp_path / "mock_scene"
        scene_dir.mkdir()
        mock_parser_deps(driver.parser, navigation_map=object(), scene_path=scene_dir)

        synthetic_tile = np.full((256, 256, 3), 200, dtype=np.uint8)
        monkeypatch.setattr(
            "data_parsing.kit_scenes.map.generate_bev_map_tile",
            lambda **kwargs: synthetic_tile,
        )

        pred_input = PluginPredictionInput(
            camera_images=sample_rgb_images,
            speed=8.0,
            acceleration=0.1,
            command=1,
            ego_pose_history=[
                type("MockPoseAtTime", (), {"timestamp_us": 0, "pose": type("MockPose", (), {"quat": type("MockQuat", (), {"w":1.0, "x":0.0, "y":0.0, "z":0.0})(), "x":0.0, "y":0.0, "z":0.0})()})(),
                type("MockPoseAtTime", (), {"timestamp_us": 1, "pose": type("MockPose", (), {"quat": type("MockQuat", (), {"w":1.0, "x":0.0, "y":0.0, "z":0.0})(), "x":0.0, "y":0.0, "z":0.0})()})(),
            ],
            inference_seed=0,
        )

        result = driver.predict(pred_input)
        assert isinstance(result, ModelPrediction)
        assert result.trajectory_xy.shape == (64, 2)
        assert result.headings.shape == (64,)
