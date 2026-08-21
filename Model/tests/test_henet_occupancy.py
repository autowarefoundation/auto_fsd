import numpy as np
import pytest

from navigation.geometry import NavigationRasterGeometry
from Platform.pipelines.henet_occupancy import (
    HENET_CODE_LICENSE_SPDX,
    HENET_SEGMENTATION_CLASS_NAMES,
    adapt_henet_segmentation,
    decompose_pinhole_projection,
    provenance,
)


def _geometry() -> NavigationRasterGeometry:
    return NavigationRasterGeometry(
        geometry_id="henet-test-2x2-1m",
        height_px=2,
        width_px=2,
        meters_per_pixel=1.0,
        x_min_m=-1.0,
        x_max_m=1.0,
        y_min_m=-1.0,
        y_max_m=1.0,
        ego_anchor_row=0.5,
        ego_anchor_col=0.5,
        matching_pc_range=(-1.0, -1.0, -2.0, 1.0, 1.0, 2.0),
        matching_bev_h=2,
        matching_bev_w=2,
        route_corridor_width_m=1.0,
        destination_marker_radius_m=1.0,
        route_rear_clip_m=1.0,
    )


def test_projection_decomposition_recovers_intrinsics_and_camera_pose():
    intrinsic = np.asarray(
        [
            [800.0, 0.0, 128.0],
            [0.0, 810.0, 129.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_to_ego = np.eye(4)
    camera_to_ego[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_to_ego[:3, 3] = [1.2, -0.4, 1.5]
    ego_to_camera = np.linalg.inv(camera_to_ego)
    projection = intrinsic @ ego_to_camera[:3, :]

    recovered_intrinsic, recovered_camera_to_ego = (
        decompose_pinhole_projection(projection)
    )

    np.testing.assert_allclose(recovered_intrinsic, intrinsic, atol=1e-8)
    np.testing.assert_allclose(
        recovered_camera_to_ego,
        camera_to_ego,
        atol=1e-8,
    )


def test_adaptation_uses_physical_coordinates_and_fixed_taxonomy():
    source = np.zeros((3, 200, 200), dtype=np.float32)
    source[
        HENET_SEGMENTATION_CLASS_NAMES.index("drivable_area")
    ] = (
        np.arange(200, dtype=np.float32)[:, None] * 0.5 + 0.25
    ) / 100.0
    source[
        HENET_SEGMENTATION_CLASS_NAMES.index("divider")
    ] = (
        np.arange(200, dtype=np.float32)[None, :] * 0.5 + 0.25
    ) / 100.0
    source[
        HENET_SEGMENTATION_CLASS_NAMES.index("vehicle")
    ].fill(0.75)

    output = adapt_henet_segmentation(source, geometry=_geometry())

    assert output.shape == (8, 2, 2)
    np.testing.assert_allclose(
        output[0],
        [[0.505, 0.505], [0.495, 0.495]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        output[1],
        [[0.505, 0.495], [0.505, 0.495]],
        atol=1e-6,
    )
    np.testing.assert_allclose(output[5], 0.75, atol=1e-6)
    assert not output[2:5].any()
    assert not output[6:].any()


def test_adaptation_zeroes_dashboard_cells_outside_henet_extent():
    source = np.ones((3, 200, 200), dtype=np.float32)
    geometry = NavigationRasterGeometry(
        geometry_id="henet-test-2x2-50m",
        height_px=2,
        width_px=2,
        meters_per_pixel=50.0,
        x_min_m=-50.0,
        x_max_m=50.0,
        y_min_m=-50.0,
        y_max_m=50.0,
        ego_anchor_row=0.5,
        ego_anchor_col=0.5,
        matching_pc_range=(-50.0, -50.0, -2.0, 50.0, 50.0, 2.0),
        matching_bev_h=2,
        matching_bev_w=2,
        route_corridor_width_m=1.0,
        destination_marker_radius_m=1.0,
        route_rear_clip_m=1.0,
    )

    output = adapt_henet_segmentation(source, geometry=geometry)

    assert np.all(output[[0, 1, 5]] == 1.0)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((3, 199, 200), dtype=np.float32),
        np.full((3, 200, 200), np.nan, dtype=np.float32),
        np.full((3, 200, 200), 1.1, dtype=np.float32),
    ],
)
def test_adaptation_rejects_invalid_official_outputs(source):
    with pytest.raises(ValueError):
        adapt_henet_segmentation(source)


def test_provenance_discloses_research_only_license_and_divider_mapping():
    metadata = provenance("a" * 64)

    assert metadata["artifact_kind"] == "native-semantic-occupancy"
    assert metadata["teacher_available"] is False
    assert metadata["supported_semantic_classes"] == [
        "drivable_area",
        "lane_area",
        "vehicle",
    ]
    assert metadata["code_license_spdx"] == HENET_CODE_LICENSE_SPDX
    assert any(
        "road-divider" in limitation
        for limitation in metadata["limitations"]
    )
    assert any(
        "academic research" in limitation
        for limitation in metadata["limitations"]
    )
