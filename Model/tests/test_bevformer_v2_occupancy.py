import math

import numpy as np
import pytest

from navigation.geometry import NavigationRasterGeometry
from Platform.pipelines.bevformer_v2_occupancy import (
    BEVFORMER_V2_FRAMES,
    DetectionBox,
    align_history_projection_to_current,
    pose_to_world_from_top_lidar,
    provenance,
    rasterize_detection_boxes,
    scale_packed_projection,
    semantic_class_for_detection,
    temporal_frame_indices,
)


def _geometry() -> NavigationRasterGeometry:
    return NavigationRasterGeometry(
        geometry_id="test-10x10-1m",
        height_px=10,
        width_px=10,
        meters_per_pixel=1.0,
        x_min_m=-5.0,
        x_max_m=5.0,
        y_min_m=-5.0,
        y_max_m=5.0,
        ego_anchor_row=4.5,
        ego_anchor_col=4.5,
        matching_pc_range=(-5.0, -5.0, -2.0, 5.0, 5.0, 2.0),
        matching_bev_h=10,
        matching_bev_w=10,
        route_corridor_width_m=3.5,
        destination_marker_radius_m=1.0,
        route_rear_clip_m=2.0,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("car", "vehicle"),
        ("truck", "vehicle"),
        ("pedestrian", "vulnerable_road_user"),
        ("bicycle", "vulnerable_road_user"),
        ("barrier", "other_obstacle"),
        ("traffic_cone", "other_obstacle"),
    ],
)
def test_bevformer_classes_map_only_to_supported_object_classes(source, target):
    assert semantic_class_for_detection(source) == target


def test_detection_raster_uses_physical_box_footprint_without_rescaling():
    geometry = _geometry()
    raster = rasterize_detection_boxes(
        [
            DetectionBox(
                class_name="car",
                score=0.75,
                center_x_m=0.0,
                center_y_m=0.0,
                length_m=4.0,
                width_m=2.0,
                yaw_rad=0.0,
            )
        ],
        geometry=geometry,
    )

    occupied = np.argwhere(raster[5] > 0)
    assert occupied.shape == (8, 2)
    assert occupied[:, 0].min() == 3
    assert occupied[:, 0].max() == 6
    assert occupied[:, 1].min() == 4
    assert occupied[:, 1].max() == 5
    assert np.all(raster[:5] == 0)
    assert np.all(raster[6:] == 0)
    assert raster[5].max() == pytest.approx(0.75)


def test_detection_yaw_rotates_the_footprint_and_overlap_uses_max_score():
    geometry = _geometry()
    raster = rasterize_detection_boxes(
        [
            DetectionBox("car", 0.4, 0.0, 0.0, 4.0, 2.0, math.pi / 2),
            DetectionBox("bus", 0.9, 0.0, 0.0, 2.0, 2.0, 0.0),
        ],
        geometry=geometry,
    )

    occupied = np.argwhere(raster[5] > 0)
    assert occupied[:, 0].min() == 4
    assert occupied[:, 0].max() == 5
    assert occupied[:, 1].min() == 3
    assert occupied[:, 1].max() == 6
    assert np.count_nonzero(np.isclose(raster[5], 0.9)) == 4


def test_detection_yaw_matches_mmdet_clockwise_lidar_corners():
    geometry = _geometry()
    raster = rasterize_detection_boxes(
        [
            DetectionBox(
                "car",
                0.8,
                0.0,
                0.0,
                4.0,
                1.0,
                math.pi / 4,
            )
        ],
        geometry=geometry,
    )
    x_grid, y_grid = geometry.pixel_center_grids()
    clockwise = np.isclose(x_grid, 0.5) & np.isclose(y_grid, -0.5)
    mirrored = np.isclose(x_grid, 0.5) & np.isclose(y_grid, 0.5)

    assert raster[5][clockwise].item() == pytest.approx(0.8)
    assert raster[5][mirrored].item() == pytest.approx(0.0)


def test_temporal_selection_matches_nuscenes_two_hz_history():
    assert temporal_frame_indices(100) == {
        -7: 65,
        -6: 70,
        -5: 75,
        -4: 80,
        -3: 85,
        -2: 90,
        -1: 95,
        0: 100,
    }
    assert temporal_frame_indices(12) == {-2: 2, -1: 7, 0: 12}
    assert tuple(BEVFORMER_V2_FRAMES) == tuple(range(-7, 1))


def test_packed_pose_uses_flu_axes_in_local_enu():
    north = pose_to_world_from_top_lidar(
        latitude_deg=35.0,
        longitude_deg=139.0,
        heading_deg_cw_from_north=0.0,
        origin_latitude_deg=35.0,
        origin_longitude_deg=139.0,
    )
    east = pose_to_world_from_top_lidar(
        latitude_deg=35.0,
        longitude_deg=139.0,
        heading_deg_cw_from_north=90.0,
        origin_latitude_deg=35.0,
        origin_longitude_deg=139.0,
    )

    np.testing.assert_allclose(north[:2, 0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(north[:2, 1], [-1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(east[:2, 0], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(east[:2, 1], [0.0, 1.0], atol=1e-12)


def test_history_projection_is_expressed_in_current_lidar_coordinates():
    projection = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    history_to_world = np.eye(4)
    history_to_world[0, 3] = 5.0
    current_to_world = np.eye(4)

    aligned, history_to_current = align_history_projection_to_current(
        projection,
        history_to_world=history_to_world,
        current_to_world=current_to_world,
    )

    assert history_to_current[0, 3] == pytest.approx(5.0)
    assert aligned[0, 3] == pytest.approx(-5.0)


def test_packed_projection_scaling_matches_model_tensor_dimensions():
    projection = np.eye(3, 4)
    projection[0, 2] = 128.0
    projection[1, 2] = 128.0

    scaled = scale_packed_projection(projection)

    assert scaled[0, 0] == pytest.approx(2.5)
    assert scaled[0, 2] == pytest.approx(320.0)
    assert scaled[1, 1] == pytest.approx(1.0)
    assert scaled[1, 2] == pytest.approx(128.0)


def test_provenance_discloses_detection_boundary_and_missing_teacher():
    metadata = provenance()

    assert metadata["artifact_kind"] == "detection-derived-occupancy"
    assert metadata["teacher_available"] is False
    assert metadata["supported_semantic_classes"] == [
        "other_obstacle",
        "vehicle",
        "vulnerable_road_user",
    ]
    assert any(
        "not BEV segmentation" in limitation
        for limitation in metadata["limitations"]
    )
    assert metadata["code_license_spdx"] == "Apache-2.0"
    assert metadata["weight_license_spdx"] == "NOASSERTION"
    assert metadata["training_data_license_spdx"] == "CC-BY-NC-SA-4.0"
    assert metadata["weight_source_url"].startswith("https://drive.google.com/")
