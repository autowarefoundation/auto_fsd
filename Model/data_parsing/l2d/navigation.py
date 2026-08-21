"""Canonical L2D map and route rasters from a pinned OSM graph."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

from data_processing.reactive_training_artifacts import (
    encode_reactive_navigation,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
    MapChannel,
    NavigationRasterGeometry,
    RouteChannel,
)

EARTH_RADIUS_M = 6_378_137.0
L2D_OSM_GRAPH_SCHEMA_VERSION = "l2d_osm_graph_v1"
L2D_SPATIAL_CELL_DEGREES = 0.005
L2D_SPATIAL_QUERY_MARGIN_M = 40.0
_MAX_INDEXED_CELLS_PER_EDGE = 4_096


@dataclasses.dataclass(frozen=True)
class L2DNavigationTargets:
    map_context: np.ndarray
    map_valid: bool
    route_target: np.ndarray
    route_channel_valid: np.ndarray
    matched_node_count: int
    route_node_count: int


@dataclasses.dataclass(frozen=True)
class L2DGraphSpatialIndex:
    node_ids: tuple[Any, ...]
    node_coordinates_lon_lat: np.ndarray
    node_cells: dict[tuple[int, int], tuple[int, ...]]
    node_cell_bounds: tuple[int, int, int, int]
    edge_refs: tuple[tuple[Any, Any, Any], ...]
    edge_bounds_lon_lat: np.ndarray
    edge_cells: dict[tuple[int, int], tuple[int, ...]]
    overflow_edge_indices: tuple[int, ...]
    cell_size_degrees: float


@dataclasses.dataclass(frozen=True)
class L2DOSMGraphSnapshot:
    graph: nx.MultiDiGraph
    spatial_index: L2DGraphSpatialIndex
    source_sha256: str
    source_revision: str
    source_artifact_sha256: str
    source_date: str
    adapter_version: str
    attribution: str


def _grid_cell(value: float, cell_size_degrees: float) -> int:
    return math.floor(value / cell_size_degrees)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _build_graph_spatial_index(
    graph: nx.MultiDiGraph,
    *,
    cell_size_degrees: float = L2D_SPATIAL_CELL_DEGREES,
) -> L2DGraphSpatialIndex:
    if not math.isfinite(cell_size_degrees) or cell_size_degrees <= 0.0:
        raise ValueError("spatial index cell size must be positive")

    node_ids = tuple(graph.nodes)
    if not node_ids:
        raise ValueError("OSM graph contains no nodes")
    node_coordinates = _freeze_array(
        np.asarray(
            [
                (
                    float(graph.nodes[node]["x"]),
                    float(graph.nodes[node]["y"]),
                )
                for node in node_ids
            ],
            dtype=np.float64,
        )
    )
    node_cells_mutable: dict[tuple[int, int], list[int]] = defaultdict(
        list
    )
    for index, (longitude, latitude) in enumerate(node_coordinates):
        node_cells_mutable[
            (
                _grid_cell(float(longitude), cell_size_degrees),
                _grid_cell(float(latitude), cell_size_degrees),
            )
        ].append(index)
    node_cells = {
        cell: tuple(indices)
        for cell, indices in node_cells_mutable.items()
    }
    node_cell_x = [cell[0] for cell in node_cells]
    node_cell_y = [cell[1] for cell in node_cells]

    edge_refs: list[tuple[Any, Any, Any]] = []
    edge_bounds: list[tuple[float, float, float, float]] = []
    edge_cells_mutable: dict[tuple[int, int], list[int]] = defaultdict(
        list
    )
    overflow_edge_indices: list[int] = []
    for source, destination, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        lon_lat = _edge_lon_lat(graph, source, destination, data)
        minimum = lon_lat.min(axis=0)
        maximum = lon_lat.max(axis=0)
        edge_index = len(edge_refs)
        edge_refs.append((source, destination, key))
        edge_bounds.append(
            (
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0]),
                float(maximum[1]),
            )
        )
        minimum_x = _grid_cell(float(minimum[0]), cell_size_degrees)
        maximum_x = _grid_cell(float(maximum[0]), cell_size_degrees)
        minimum_y = _grid_cell(float(minimum[1]), cell_size_degrees)
        maximum_y = _grid_cell(float(maximum[1]), cell_size_degrees)
        cell_count = (maximum_x - minimum_x + 1) * (
            maximum_y - minimum_y + 1
        )
        if cell_count > _MAX_INDEXED_CELLS_PER_EDGE:
            overflow_edge_indices.append(edge_index)
            continue
        for cell_x in range(minimum_x, maximum_x + 1):
            for cell_y in range(minimum_y, maximum_y + 1):
                edge_cells_mutable[(cell_x, cell_y)].append(edge_index)

    edge_bounds_array = np.asarray(edge_bounds, dtype=np.float64).reshape(
        -1,
        4,
    )
    return L2DGraphSpatialIndex(
        node_ids=node_ids,
        node_coordinates_lon_lat=node_coordinates,
        node_cells=node_cells,
        node_cell_bounds=(
            min(node_cell_x),
            max(node_cell_x),
            min(node_cell_y),
            max(node_cell_y),
        ),
        edge_refs=tuple(edge_refs),
        edge_bounds_lon_lat=_freeze_array(edge_bounds_array),
        edge_cells={
            cell: tuple(indices)
            for cell, indices in edge_cells_mutable.items()
        },
        overflow_edge_indices=tuple(overflow_edge_indices),
        cell_size_degrees=cell_size_degrees,
    )


def load_l2d_osm_graph_snapshot(
    path: str | Path,
) -> L2DOSMGraphSnapshot:
    """Load a pinned, network-free OSM graph used by the L2D packer."""
    source = Path(path)
    payload_bytes = source.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("L2D OSM graph snapshot is not valid JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != L2D_OSM_GRAPH_SCHEMA_VERSION
    ):
        raise ValueError("unsupported L2D OSM graph snapshot schema")
    source_revision = payload.get("source_revision")
    source_artifact_sha256 = payload.get("source_artifact_sha256", "")
    source_date = payload.get("source_date", "")
    adapter_version = payload.get("adapter_version", "")
    attribution = payload.get("attribution")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("OSM snapshot source_revision must not be empty")
    for name, value in (
        ("source_date", source_date),
        ("adapter_version", adapter_version),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"OSM snapshot {name} must not be empty")
    if (
        not isinstance(source_artifact_sha256, str)
        or len(source_artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_artifact_sha256
        )
    ):
        raise ValueError(
            "OSM snapshot source_artifact_sha256 must be lowercase SHA-256"
        )
    if not isinstance(attribution, str) or not attribution:
        raise ValueError("OSM snapshot attribution must not be empty")
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("OSM snapshot must contain nodes")
    if not isinstance(edges, list) or not edges:
        raise ValueError("OSM snapshot must contain edges")

    graph = nx.MultiDiGraph()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("OSM snapshot node must be an object")
        node_id = str(node.get("id", ""))
        longitude = float(node["longitude_deg"])
        latitude = float(node["latitude_deg"])
        if (
            not node_id
            or not math.isfinite(longitude)
            or not math.isfinite(latitude)
        ):
            raise ValueError("OSM snapshot node is invalid")
        if node_id in graph:
            raise ValueError(f"duplicate OSM node {node_id!r}")
        graph.add_node(node_id, x=longitude, y=latitude)

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("OSM snapshot edge must be an object")
        source_id = str(edge.get("source", ""))
        destination_id = str(edge.get("destination", ""))
        key = str(edge.get("key", "0"))
        if source_id not in graph or destination_id not in graph:
            raise ValueError("OSM snapshot edge references an unknown node")
        length_m = float(edge.get("length_m", 0.0))
        if not math.isfinite(length_m) or length_m < 0.0:
            raise ValueError("OSM snapshot edge length is invalid")
        geometry = edge.get("geometry_lon_lat")
        edge_attributes: dict[str, Any] = {
            "length": length_m,
            "lanes": edge.get("lanes", 2),
            "width": edge.get("width_m"),
        }
        if geometry is not None:
            geometry_array = np.asarray(geometry, dtype=np.float64)
            if (
                geometry_array.ndim != 2
                or geometry_array.shape[0] < 2
                or geometry_array.shape[1] != 2
                or not np.isfinite(geometry_array).all()
            ):
                raise ValueError("OSM edge geometry must be finite [N,2]")
            edge_attributes["geometry"] = geometry_array
        if graph.has_edge(source_id, destination_id, key=key):
            raise ValueError(
                "duplicate OSM edge "
                f"{source_id!r}->{destination_id!r}:{key!r}"
            )
        graph.add_edge(
            source_id,
            destination_id,
            key=key,
            **edge_attributes,
        )
    return L2DOSMGraphSnapshot(
        graph=graph,
        spatial_index=_build_graph_spatial_index(graph),
        source_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        source_date=source_date,
        adapter_version=adapter_version,
        attribution=attribution,
    )


def _project_to_ego_local(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    ego_lat: float,
    ego_lon: float,
    ego_heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_lat = math.cos(math.radians(ego_lat))
    degrees_to_meters = EARTH_RADIUS_M * math.pi / 180.0
    x_east = (longitudes - ego_lon) * cos_lat * degrees_to_meters
    y_north = (latitudes - ego_lat) * degrees_to_meters
    cosine = math.cos(-ego_heading)
    sine = math.sin(-ego_heading)
    return (
        x_east * cosine - y_north * sine,
        x_east * sine + y_north * cosine,
    )


def _nearest_node(
    graph: nx.MultiDiGraph,
    longitude: float,
    latitude: float,
    spatial_index: L2DGraphSpatialIndex | None = None,
) -> Any:
    if spatial_index is not None:
        cell_size = spatial_index.cell_size_degrees
        query_x = _grid_cell(longitude, cell_size)
        query_y = _grid_cell(latitude, cell_size)
        minimum_x, maximum_x, minimum_y, maximum_y = (
            spatial_index.node_cell_bounds
        )
        if (
            minimum_x <= query_x <= maximum_x
            and minimum_y <= query_y <= maximum_y
        ):
            candidate_indices: list[int] = []
            maximum_radius = max(
                query_x - minimum_x,
                maximum_x - query_x,
                query_y - minimum_y,
                maximum_y - query_y,
            )
            longitude_scale = math.cos(math.radians(latitude))
            for radius in range(maximum_radius + 1):
                if radius == 0:
                    cells: tuple[tuple[int, int], ...] = (
                        (query_x, query_y),
                    )
                else:
                    cells = tuple(
                        (cell_x, cell_y)
                        for cell_x in range(
                            query_x - radius,
                            query_x + radius + 1,
                        )
                        for cell_y in range(
                            query_y - radius,
                            query_y + radius + 1,
                        )
                        if (
                            abs(cell_x - query_x) == radius
                            or abs(cell_y - query_y) == radius
                        )
                    )
                for cell in cells:
                    candidate_indices.extend(
                        spatial_index.node_cells.get(cell, ())
                    )
                if not candidate_indices:
                    continue
                candidate_array = np.asarray(
                    candidate_indices,
                    dtype=np.int64,
                )
                coordinates = spatial_index.node_coordinates_lon_lat[
                    candidate_array
                ]
                distance = (
                    (
                        (coordinates[:, 0] - longitude)
                        * longitude_scale
                    )
                    ** 2
                    + (coordinates[:, 1] - latitude) ** 2
                )
                nearest_offset = int(np.argmin(distance))
                best_distance = float(distance[nearest_offset])
                left = (
                    longitude - (query_x - radius) * cell_size
                ) * longitude_scale
                right = (
                    (query_x + radius + 1) * cell_size - longitude
                ) * longitude_scale
                bottom = latitude - (query_y - radius) * cell_size
                top = (query_y + radius + 1) * cell_size - latitude
                boundary_distance = min(left, right, bottom, top)
                if best_distance <= boundary_distance**2:
                    return spatial_index.node_ids[
                        int(candidate_array[nearest_offset])
                    ]

    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("OSM graph contains no nodes")
    coordinates = np.asarray(
        [
            (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
            for node in nodes
        ],
        dtype=np.float64,
    )
    longitude_scale = math.cos(math.radians(latitude))
    distance = (
        ((coordinates[:, 0] - longitude) * longitude_scale) ** 2
        + (coordinates[:, 1] - latitude) ** 2
    )
    return nodes[int(np.argmin(distance))]


def _map_match_waypoints(
    graph: nx.MultiDiGraph,
    waypoints_lon_lat: np.ndarray,
    spatial_index: L2DGraphSpatialIndex | None = None,
) -> tuple[list[Any], list[Any]]:
    matched = [
        _nearest_node(
            graph,
            float(longitude),
            float(latitude),
            spatial_index,
        )
        for longitude, latitude in waypoints_lon_lat
    ]
    route: list[Any] = []
    for source, destination in zip(matched[:-1], matched[1:]):
        if source == destination:
            if not route or route[-1] != source:
                route.append(source)
            continue
        try:
            segment = nx.shortest_path(
                graph,
                source,
                destination,
                weight="length",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return matched, []
        if route and route[-1] == segment[0]:
            segment = segment[1:]
        route.extend(segment)
    if not route and matched:
        route = [matched[0]]
    return matched, route


def _ego_flu_from_lon_lat(
    lon_lat: np.ndarray,
    *,
    ego_lat: float,
    ego_lon: float,
    heading_deg_cw_from_north: float,
) -> np.ndarray:
    points = np.asarray(lon_lat, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("longitude/latitude points must have shape [N,2]")
    x_right, y_forward = _project_to_ego_local(
        points[:, 1],
        points[:, 0],
        ego_lat,
        ego_lon,
        math.radians(heading_deg_cw_from_north),
    )
    return np.column_stack([y_forward, -x_right])


def _polyline_mask(
    points_xy: np.ndarray,
    geometry: NavigationRasterGeometry,
    *,
    width_m: float,
) -> np.ndarray:
    output = Image.new(
        "L",
        (geometry.width_px, geometry.height_px),
        color=0,
    )
    if len(points_xy) < 2:
        return np.asarray(output, dtype=np.uint8)
    pixels = geometry.ego_to_pixel(points_xy)
    ImageDraw.Draw(output).line(
        [tuple(value) for value in pixels[:, ::-1]],
        fill=1,
        width=max(1, round(width_m / geometry.meters_per_pixel)),
        joint="curve",
    )
    return np.asarray(output, dtype=np.uint8)


def _edge_lon_lat(
    graph: nx.MultiDiGraph,
    source: Any,
    destination: Any,
    data: dict[str, Any],
) -> np.ndarray:
    edge_geometry = data.get("geometry")
    if edge_geometry is not None:
        coordinates = (
            edge_geometry.coords
            if hasattr(edge_geometry, "coords")
            else edge_geometry
        )
        return np.asarray(coordinates, dtype=np.float64)
    return np.asarray(
        [
            (graph.nodes[source]["x"], graph.nodes[source]["y"]),
            (graph.nodes[destination]["x"], graph.nodes[destination]["y"]),
        ],
        dtype=np.float64,
    )


def _road_width_m(data: dict[str, Any]) -> float:
    raw_width = data.get("width")
    try:
        if raw_width is not None:
            width = float(str(raw_width).split()[0])
            if math.isfinite(width) and width > 0.0:
                return min(width, 30.0)
    except (TypeError, ValueError):
        pass
    raw_lanes = data.get("lanes", 2)
    try:
        lanes = max(1, min(int(str(raw_lanes).split(";")[0]), 8))
    except (TypeError, ValueError):
        lanes = 2
    return 3.5 * lanes


def _navigation_query_bounds_lon_lat(
    *,
    ego_lat: float,
    ego_lon: float,
    geometry: NavigationRasterGeometry,
) -> tuple[float, float, float, float]:
    maximum_radius_m = (
        max(
            math.hypot(longitudinal, lateral)
            for longitudinal in (geometry.x_min_m, geometry.x_max_m)
            for lateral in (geometry.y_min_m, geometry.y_max_m)
        )
        + L2D_SPATIAL_QUERY_MARGIN_M
    )
    degrees_to_meters = EARTH_RADIUS_M * math.pi / 180.0
    latitude_delta = maximum_radius_m / degrees_to_meters
    longitude_scale = max(abs(math.cos(math.radians(ego_lat))), 1e-6)
    longitude_delta = maximum_radius_m / (
        degrees_to_meters * longitude_scale
    )
    return (
        ego_lon - longitude_delta,
        ego_lat - latitude_delta,
        ego_lon + longitude_delta,
        ego_lat + latitude_delta,
    )


def _candidate_edge_refs(
    graph: nx.MultiDiGraph,
    spatial_index: L2DGraphSpatialIndex | None,
    *,
    ego_lat: float,
    ego_lon: float,
    geometry: NavigationRasterGeometry,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    if spatial_index is None:
        return [
            (source, destination, data)
            for source, destination, data in graph.edges(data=True)
        ]

    minimum_lon, minimum_lat, maximum_lon, maximum_lat = (
        _navigation_query_bounds_lon_lat(
            ego_lat=ego_lat,
            ego_lon=ego_lon,
            geometry=geometry,
        )
    )
    cell_size = spatial_index.cell_size_degrees
    candidate_indices = set(spatial_index.overflow_edge_indices)
    for cell_x in range(
        _grid_cell(minimum_lon, cell_size),
        _grid_cell(maximum_lon, cell_size) + 1,
    ):
        for cell_y in range(
            _grid_cell(minimum_lat, cell_size),
            _grid_cell(maximum_lat, cell_size) + 1,
        ):
            candidate_indices.update(
                spatial_index.edge_cells.get((cell_x, cell_y), ())
            )

    selected: list[tuple[Any, Any, dict[str, Any]]] = []
    bounds = spatial_index.edge_bounds_lon_lat
    for edge_index in sorted(candidate_indices):
        edge_minimum_lon, edge_minimum_lat, edge_maximum_lon, (
            edge_maximum_lat
        ) = bounds[edge_index]
        if (
            edge_maximum_lon < minimum_lon
            or edge_minimum_lon > maximum_lon
            or edge_maximum_lat < minimum_lat
            or edge_minimum_lat > maximum_lat
        ):
            continue
        source, destination, key = spatial_index.edge_refs[edge_index]
        data = graph.get_edge_data(source, destination, key)
        if data is None:
            raise ValueError("OSM spatial index references a missing edge")
        selected.append((source, destination, data))
    return selected


def _destination_heatmap(
    destination_xy: np.ndarray,
    geometry: NavigationRasterGeometry,
) -> tuple[np.ndarray, bool]:
    output = np.zeros(
        (geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    if not bool(geometry.contains_ego_points(destination_xy[None])[0]):
        return output, False
    center = geometry.ego_to_pixel(destination_xy[None])[0]
    center_row = int(
        np.clip(round(float(center[0])), 0, geometry.height_px - 1)
    )
    center_col = int(
        np.clip(round(float(center[1])), 0, geometry.width_px - 1)
    )
    rows, cols = np.meshgrid(
        np.arange(geometry.height_px, dtype=np.float32),
        np.arange(geometry.width_px, dtype=np.float32),
        indexing="ij",
    )
    sigma_px = geometry.destination_marker_radius_m / (
        2.0 * geometry.meters_per_pixel
    )
    output = np.exp(
        -((rows - center_row) ** 2 + (cols - center_col) ** 2)
        / (2.0 * sigma_px**2)
    ).astype(np.float32)
    return output, True


def build_l2d_navigation_targets(
    graph: nx.MultiDiGraph,
    route_waypoints_lon_lat: np.ndarray,
    *,
    ego_lat: float,
    ego_lon: float,
    heading_deg_cw_from_north: float,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    spatial_index: L2DGraphSpatialIndex | None = None,
) -> L2DNavigationTargets:
    """Build map/route inputs without reading the imitation future."""
    waypoints = np.asarray(route_waypoints_lon_lat, dtype=np.float64)
    if waypoints.shape != (10, 2) or not np.isfinite(waypoints).all():
        raise ValueError("L2D route waypoints must be finite [10,2]")
    matched_nodes, route_nodes = _map_match_waypoints(
        graph,
        waypoints,
        spatial_index,
    )

    map_context = np.zeros(
        (MAP_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    drivable = np.zeros(map_context.shape[1:], dtype=np.uint8)
    direction_sin = np.zeros(map_context.shape[1:], dtype=np.float32)
    direction_cos = np.zeros_like(direction_sin)
    direction_count = np.zeros_like(direction_sin)
    for source, destination, data in _candidate_edge_refs(
        graph,
        spatial_index,
        ego_lat=ego_lat,
        ego_lon=ego_lon,
        geometry=geometry,
    ):
        lon_lat = _edge_lon_lat(graph, source, destination, data)
        ego_points = _ego_flu_from_lon_lat(
            lon_lat,
            ego_lat=ego_lat,
            ego_lon=ego_lon,
            heading_deg_cw_from_north=heading_deg_cw_from_north,
        )
        if len(ego_points) < 2:
            continue
        drivable = np.maximum(
            drivable,
            _polyline_mask(
                ego_points,
                geometry,
                width_m=_road_width_m(data),
            ),
        )
        centerline_uint8 = _polyline_mask(
            ego_points,
            geometry,
            width_m=geometry.meters_per_pixel,
        )
        map_context[MapChannel.LANE_CENTERLINE] = np.maximum(
            map_context[MapChannel.LANE_CENTERLINE],
            centerline_uint8,
        )
        delta = ego_points[-1] - ego_points[0]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            centerline = centerline_uint8.astype(bool)
            direction_sin[centerline] += delta[1] / norm
            direction_cos[centerline] += delta[0] / norm
            direction_count[centerline] += 1.0

    if bool(drivable.any()):
        map_context[MapChannel.DRIVABLE_AREA] = drivable
        map_context[MapChannel.KNOWN_MAP_AREA] = drivable
    direction_valid = direction_count > 0
    map_context[MapChannel.TRAFFIC_DIRECTION_VALID] = direction_valid
    map_context[MapChannel.TRAFFIC_DIRECTION_SIN][direction_valid] = (
        (
            direction_sin[direction_valid]
            / direction_count[direction_valid]
            + 1.0
        )
        * 0.5
    )
    map_context[MapChannel.TRAFFIC_DIRECTION_COS][direction_valid] = (
        (
            direction_cos[direction_valid]
            / direction_count[direction_valid]
            + 1.0
        )
        * 0.5
    )

    route_target = np.zeros(
        (ROUTE_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    route_valid = np.zeros(ROUTE_CHANNEL_COUNT, dtype=np.bool_)
    if route_nodes:
        route_lon_lat = np.asarray(
            [
                (graph.nodes[node]["x"], graph.nodes[node]["y"])
                for node in route_nodes
            ],
            dtype=np.float64,
        )
        route_xy = _ego_flu_from_lon_lat(
            route_lon_lat,
            ego_lat=ego_lat,
            ego_lon=ego_lon,
            heading_deg_cw_from_north=heading_deg_cw_from_north,
        )
        route_target[RouteChannel.SELECTED_CORRIDOR] = _polyline_mask(
            route_xy,
            geometry,
            width_m=geometry.route_corridor_width_m,
        )
        route_valid[RouteChannel.SELECTED_CORRIDOR] = len(route_xy) >= 2

    waypoint_xy = _ego_flu_from_lon_lat(
        waypoints,
        ego_lat=ego_lat,
        ego_lon=ego_lon,
        heading_deg_cw_from_north=heading_deg_cw_from_north,
    )
    visible = geometry.contains_ego_points(waypoint_xy)
    if bool(visible.any()):
        destination, destination_valid = _destination_heatmap(
            waypoint_xy[np.flatnonzero(visible)[-1]],
            geometry,
        )
        route_target[RouteChannel.DESTINATION] = destination
        route_valid[RouteChannel.DESTINATION] = destination_valid

    return L2DNavigationTargets(
        map_context=map_context,
        map_valid=bool(drivable.any()),
        route_target=route_target,
        route_channel_valid=route_valid,
        matched_node_count=len(matched_nodes),
        route_node_count=len(route_nodes),
    )


def l2d_reactive_navigation_members(
    snapshot: L2DOSMGraphSnapshot,
    route_waypoints_lon_lat: np.ndarray,
    pose_current: dict[str, float | int],
    *,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
) -> dict[str, bytes]:
    """Encode one L2D Map/Route target from pinned OSM and route intent."""
    targets = build_l2d_navigation_targets(
        snapshot.graph,
        route_waypoints_lon_lat,
        ego_lat=float(pose_current["latitude_deg"]),
        ego_lon=float(pose_current["longitude_deg"]),
        heading_deg_cw_from_north=float(
            pose_current["heading_deg_cw_from_north"]
        ),
        geometry=geometry,
        spatial_index=snapshot.spatial_index,
    )
    return encode_reactive_navigation(
        targets.map_context,
        targets.route_target,
        map_valid=targets.map_valid,
        route_channel_valid=targets.route_channel_valid,
        geometry=geometry,
        metadata={
            "map_source": "pinned_osm_graph",
            "route_source": "l2d_observation_state_waypoints",
            "osm_source_sha256": snapshot.source_sha256,
            "osm_source_artifact_sha256": (
                snapshot.source_artifact_sha256
            ),
            "osm_source_date": snapshot.source_date,
            "osm_source_revision": snapshot.source_revision,
            "osm_adapter_version": snapshot.adapter_version,
            "osm_attribution": snapshot.attribution,
            "matched_node_count": targets.matched_node_count,
            "route_node_count": targets.route_node_count,
        },
    )
