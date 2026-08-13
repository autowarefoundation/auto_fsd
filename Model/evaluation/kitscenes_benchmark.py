"""Reproducible KITScenes Multimodal E2E benchmark contracts.

KITScenes currently publishes the paper protocol but not the exact 200-sample
manifest or community evaluator. This module therefore makes provenance an
input: a checkpoint can be evaluated later against an immutable manifest, while
MLflow can distinguish development, paper-protocol approximation, and official
release results.

Only displacement metrics are computed here. Drivable-surface survival,
collision-free rate, centerline distance, and MMS require benchmark-authority
map/occupancy/maneuver assets and must not be approximated from the model's
raster map input.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluation.metrics import integrate_trajectory


MANIFEST_SCHEMA_VERSION = "kitscenes_e2e_benchmark_manifest_v1"
PROTOCOL_ID = "kitscenes_multimodal_e2e_v1"
EVALUATOR_VERSION = "kitscenes_pose_displacement_v1"
PAPER_PROTOCOL_SOURCE = "https://arxiv.org/html/2606.02956#A8.SS4"
PAPER_APPROXIMATION_SELECTION_VERSION = (
    "kitscenes_paper_approximation_sha256_v1"
)
PAPER_APPROXIMATION_SELECTION_SEED = (
    "kitscenes-multimodal-e2e-paper-approximation-v1"
)
PAPER_WINDOW_STEPS = 90

_PROTOCOL_STATUSES = {
    "development",
    "paper_protocol_approximation",
    "official",
}
_KNOWN_SPLITS = {
    "train",
    "val",
    "overlap-train-val",
    "test",
    "test-e2e",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_SAMPLE_UID = re.compile(
    r"^kitscenes-v1-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-"
    r"[0-9a-f]{12}-f[0-9]{6}$"
)
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class KITScenesBenchmarkManifest:
    benchmark_id: str
    protocol_status: str
    protocol_source: str
    authority: str
    release_id: str
    dataset_revision: str
    sdk_revision: str
    source_splits: tuple[str, ...]
    sample_uids: tuple[str, ...]
    frequency_hz: int
    past_seconds: int
    horizons_seconds: tuple[int, ...]
    input_track: str
    history_adapter: str

    @property
    def observation_steps(self) -> int:
        return self.frequency_hz * self.past_seconds

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        return tuple(
            self.frequency_hz * seconds
            for seconds in self.horizons_seconds
        )


@dataclass(frozen=True)
class KITScenesBenchmarkCandidate:
    sample_uid: str
    source_split: str
    scene_id: str
    frame_index: int


def select_paper_approximation_samples(
    candidates_by_split: Mapping[
        str, Sequence[KITScenesBenchmarkCandidate]
    ],
    *,
    sample_count: int = 200,
    window_steps: int = PAPER_WINDOW_STEPS,
    selection_seed: str = PAPER_APPROXIMATION_SELECTION_SEED,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Select a metric-independent approximation of the unpublished 200 UIDs.

    KITScenes specifies 200 non-overlapping nine-second windows from
    ``val union overlap-train-val`` but does not publish the anchor UIDs or
    sampling seed. This contract first greedily removes overlapping windows
    inside each scene, then ranks the remaining candidates by a pinned SHA-256
    seed. The final UID list is sorted only to make the manifest canonical.
    """
    expected_splits = {"val", "overlap-train-val"}
    if set(candidates_by_split) != expected_splits:
        raise ValueError(
            "paper approximation candidates must contain exactly "
            "val and overlap-train-val"
        )
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    if not selection_seed:
        raise ValueError("selection_seed must not be empty")

    seen_uids: set[str] = set()
    non_overlapping: list[KITScenesBenchmarkCandidate] = []
    candidate_counts: dict[str, int] = {}
    for split in sorted(expected_splits):
        by_scene: dict[str, list[KITScenesBenchmarkCandidate]] = defaultdict(
            list
        )
        for candidate in candidates_by_split[split]:
            if candidate.source_split != split:
                raise ValueError(
                    "candidate source split differs from its collection: "
                    f"{candidate.source_split!r} != {split!r}"
                )
            if not _SAFE_SAMPLE_UID.fullmatch(candidate.sample_uid):
                raise ValueError(
                    f"unsafe KITScenes candidate UID {candidate.sample_uid!r}"
                )
            if not candidate.scene_id:
                raise ValueError("candidate scene_id must not be empty")
            if candidate.frame_index < 0:
                raise ValueError("candidate frame_index must be non-negative")
            if candidate.sample_uid in seen_uids:
                raise ValueError(
                    f"duplicate KITScenes candidate UID {candidate.sample_uid}"
                )
            seen_uids.add(candidate.sample_uid)
            by_scene[candidate.scene_id].append(candidate)

        selected_for_split: list[KITScenesBenchmarkCandidate] = []
        for scene_id in sorted(by_scene):
            scene_candidates = sorted(
                by_scene[scene_id],
                key=lambda item: (item.frame_index, item.sample_uid),
            )
            seen_frames: set[int] = set()
            last_frame: int | None = None
            for candidate in scene_candidates:
                if candidate.frame_index in seen_frames:
                    raise ValueError(
                        "duplicate KITScenes candidate frame in scene "
                        f"{scene_id}: {candidate.frame_index}"
                    )
                seen_frames.add(candidate.frame_index)
                if (
                    last_frame is None
                    or candidate.frame_index - last_frame >= window_steps
                ):
                    selected_for_split.append(candidate)
                    last_frame = candidate.frame_index
        candidate_counts[split] = len(selected_for_split)
        non_overlapping.extend(selected_for_split)

    if len(non_overlapping) < sample_count:
        raise ValueError(
            "KITScenes paper approximation has too few non-overlapping "
            f"candidates: {len(non_overlapping)} < {sample_count}"
        )

    def rank(candidate: KITScenesBenchmarkCandidate) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{selection_seed}:{candidate.sample_uid}".encode("ascii")
        ).hexdigest()
        return digest, candidate.sample_uid

    selected = sorted(non_overlapping, key=rank)[:sample_count]
    selected_counts = Counter(
        candidate.source_split for candidate in selected
    )
    missing_splits = expected_splits - set(selected_counts)
    if missing_splits:
        raise ValueError(
            "hash-ranked KITScenes selection omitted source splits: "
            f"{sorted(missing_splits)}"
        )
    sample_uids = tuple(sorted(candidate.sample_uid for candidate in selected))
    metadata: dict[str, Any] = {
        "candidate_count": len(non_overlapping),
        "candidate_count_by_split": {
            split: candidate_counts[split] for split in sorted(expected_splits)
        },
        "non_overlap_window_steps": window_steps,
        "selected_count_by_split": {
            split: selected_counts[split] for split in sorted(expected_splits)
        },
        "selection_seed": selection_seed,
        "selection_version": PAPER_APPROXIMATION_SELECTION_VERSION,
    }
    return sample_uids, metadata


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"benchmark manifest is missing {key!r}")
    return payload[key]


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def parse_benchmark_manifest(
    payload: Mapping[str, Any],
) -> KITScenesBenchmarkManifest:
    """Validate and normalize one immutable benchmark sample manifest."""
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark manifest schema "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"benchmark protocol_id must be {PROTOCOL_ID!r}"
        )

    benchmark_id = str(_required(payload, "benchmark_id"))
    if not _SAFE_ID.fullmatch(benchmark_id):
        raise ValueError(f"unsafe benchmark_id {benchmark_id!r}")

    protocol_status = str(_required(payload, "protocol_status"))
    if protocol_status not in _PROTOCOL_STATUSES:
        raise ValueError(
            f"unsupported protocol_status {protocol_status!r}"
        )
    protocol_source = str(_required(payload, "protocol_source"))
    if not protocol_source.startswith("https://"):
        raise ValueError("protocol_source must be an HTTPS URL")

    authority = str(_required(payload, "authority"))
    release_id = str(_required(payload, "release_id"))
    if not _SAFE_ID.fullmatch(authority):
        raise ValueError(f"unsafe authority {authority!r}")
    if not _SAFE_ID.fullmatch(release_id):
        raise ValueError(f"unsafe release_id {release_id!r}")

    dataset_revision = str(_required(payload, "dataset_revision"))
    sdk_revision = str(_required(payload, "sdk_revision"))
    if not _COMMIT_SHA.fullmatch(dataset_revision):
        raise ValueError("dataset_revision must be a 40-character commit SHA")
    if not _COMMIT_SHA.fullmatch(sdk_revision):
        raise ValueError("sdk_revision must be a 40-character commit SHA")

    raw_splits = _required(payload, "source_splits")
    if (
        not isinstance(raw_splits, list)
        or not raw_splits
        or any(not isinstance(split, str) for split in raw_splits)
    ):
        raise ValueError("source_splits must be a non-empty string list")
    source_splits = tuple(raw_splits)
    if len(set(source_splits)) != len(source_splits):
        raise ValueError("source_splits contains duplicates")
    unknown_splits = set(source_splits) - _KNOWN_SPLITS
    if unknown_splits:
        raise ValueError(
            f"benchmark manifest has unknown source splits {sorted(unknown_splits)}"
        )

    raw_uids = _required(payload, "sample_uids")
    if (
        not isinstance(raw_uids, list)
        or not raw_uids
        or any(not isinstance(uid, str) for uid in raw_uids)
    ):
        raise ValueError("sample_uids must be a non-empty string list")
    sample_uids = tuple(raw_uids)
    if len(set(sample_uids)) != len(sample_uids):
        raise ValueError("sample_uids contains duplicates")
    invalid_uids = [
        uid for uid in sample_uids if not _SAFE_SAMPLE_UID.fullmatch(uid)
    ]
    if invalid_uids:
        raise ValueError(
            f"benchmark manifest has unsafe sample UIDs {invalid_uids[:3]}"
        )
    sample_count = _positive_int(
        _required(payload, "sample_count"), "sample_count"
    )
    if sample_count != len(sample_uids):
        raise ValueError(
            f"sample_count={sample_count} does not match "
            f"{len(sample_uids)} sample_uids"
        )

    frequency_hz = _positive_int(
        _required(payload, "frequency_hz"), "frequency_hz"
    )
    past_seconds = _positive_int(
        _required(payload, "past_seconds"), "past_seconds"
    )
    raw_horizons = _required(payload, "horizons_seconds")
    if (
        not isinstance(raw_horizons, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_horizons
        )
    ):
        raise ValueError("horizons_seconds must be an integer list")
    horizons_seconds = tuple(raw_horizons)
    if (
        frequency_hz != 10
        or past_seconds != 4
        or horizons_seconds != (3, 5)
    ):
        raise ValueError(
            "KITScenes E2E protocol requires 10 Hz, 4 seconds of past "
            "observation, and [3, 5] second horizons"
        )

    input_track = str(_required(payload, "input_track"))
    if not _SAFE_ID.fullmatch(input_track):
        raise ValueError(f"unsafe input_track {input_track!r}")
    history_adapter = str(_required(payload, "history_adapter"))
    if history_adapter != "left_zero_pad_to_64":
        raise ValueError(
            "history_adapter must be 'left_zero_pad_to_64'"
        )

    if protocol_status == "official":
        if authority != "KIT-MRT":
            raise ValueError(
                "official benchmark manifests must be issued by KIT-MRT"
            )
        if sample_count != 200:
            raise ValueError(
                "official KITScenes E2E manifests must contain 200 samples"
            )
    elif protocol_status == "paper_protocol_approximation":
        if protocol_source != PAPER_PROTOCOL_SOURCE:
            raise ValueError(
                "paper protocol approximation must cite the KITScenes paper"
            )
        if set(source_splits) != {"val", "overlap-train-val"}:
            raise ValueError(
                "paper protocol approximation must use val and "
                "overlap-train-val"
            )
        if sample_count != 200:
            raise ValueError(
                "paper protocol approximation must contain 200 samples"
            )

    return KITScenesBenchmarkManifest(
        benchmark_id=benchmark_id,
        protocol_status=protocol_status,
        protocol_source=protocol_source,
        authority=authority,
        release_id=release_id,
        dataset_revision=dataset_revision,
        sdk_revision=sdk_revision,
        source_splits=source_splits,
        sample_uids=sample_uids,
        frequency_hz=frequency_hz,
        past_seconds=past_seconds,
        horizons_seconds=horizons_seconds,
        input_track=input_track,
        history_adapter=history_adapter,
    )


def load_benchmark_manifest(
    path: str | Path,
) -> tuple[KITScenesBenchmarkManifest, str]:
    manifest_path = Path(path)
    payload_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("benchmark manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("benchmark manifest root must be an object")
    return (
        parse_benchmark_manifest(payload),
        hashlib.sha256(payload_bytes).hexdigest(),
    )


def sample_uid_digest(sample_uids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(sample_uids)).encode("utf-8")
    ).hexdigest()


def limit_egomotion_history(
    history: torch.Tensor,
    *,
    observation_steps: int,
) -> torch.Tensor:
    """Mask context older than the benchmark window without changing the ABI."""
    if history.ndim != 2 or history.shape[1] != 64 * 4:
        raise ValueError(
            "egomotion history must have shape [batch, 256]"
        )
    if not 0 < observation_steps <= 64:
        raise ValueError("observation_steps must be between 1 and 64")
    limited = history.reshape(history.shape[0], 64, 4).clone()
    limited[:, : 64 - observation_steps, :] = 0
    return limited.reshape(history.shape[0], 64 * 4)


def wgs84_trajectory_to_ego_xy(
    gps_future: np.ndarray,
    current_pose: np.ndarray,
) -> np.ndarray:
    """Convert packed KITScenes poses to future ego-frame XY in metres."""
    gps = np.asarray(gps_future, dtype=np.float64)
    pose = np.asarray(current_pose, dtype=np.float64)
    if gps.ndim != 3 or gps.shape[1:] != (65, 2):
        raise ValueError("gps_future must have shape [B,65,2]")
    if pose.shape != (gps.shape[0], 3):
        raise ValueError("current_pose must have shape [B,3]")
    if not np.isfinite(gps).all() or not np.isfinite(pose).all():
        raise ValueError("benchmark poses contain non-finite values")
    if (
        np.any(gps[:, :, 0] < -90.0)
        or np.any(gps[:, :, 0] > 90.0)
        or np.any(gps[:, :, 1] < -180.0)
        or np.any(gps[:, :, 1] > 180.0)
    ):
        raise ValueError("benchmark GPS coordinates are out of range")
    if not np.allclose(gps[:, 0, :], pose[:, :2], atol=1e-10, rtol=0.0):
        raise ValueError(
            "current pose does not match the first packed GPS point"
        )

    from pyproj import Transformer

    flattened = gps.reshape(-1, 2)
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32632", always_xy=True
    )
    east, north = transformer.transform(
        flattened[:, 1], flattened[:, 0]
    )
    utm = np.column_stack([east, north]).reshape(gps.shape)
    offsets = utm - utm[:, :1, :]

    heading = np.radians(pose[:, 2])[:, None]
    forward = (
        offsets[:, :, 0] * np.sin(heading)
        + offsets[:, :, 1] * np.cos(heading)
    )
    left = (
        -offsets[:, :, 0] * np.cos(heading)
        + offsets[:, :, 1] * np.sin(heading)
    )
    return np.stack([forward, left], axis=2)[:, 1:, :]


def compute_displacement_metrics(
    predicted_controls: np.ndarray,
    target_xy: np.ndarray,
    initial_speeds: np.ndarray,
    *,
    frequency_hz: int = 10,
    horizons_seconds: Sequence[int] = (3, 5),
) -> tuple[dict[str, float], np.ndarray]:
    """Compute pose-grounded KITScenes ADE/FDE and predicted XY trajectories."""
    predicted = np.asarray(predicted_controls, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    speeds = np.asarray(initial_speeds, dtype=np.float64)
    if (
        predicted.ndim != 3
        or predicted.shape[2] != 2
        or target.shape != predicted.shape
    ):
        raise ValueError(
            "predicted_controls and target_xy must share shape [B,T,2]"
        )
    if speeds.shape != (predicted.shape[0],):
        raise ValueError("initial_speeds must have shape [B]")
    if predicted.shape[0] == 0:
        raise ValueError("benchmark batch must not be empty")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError(
            "benchmark controls or target poses contain non-finite values"
        )
    if not np.isfinite(speeds).all():
        raise ValueError("benchmark initial speeds contain non-finite values")
    if np.any(speeds < 0):
        raise ValueError("benchmark initial speeds must be non-negative")

    horizon_steps = tuple(
        _positive_int(seconds, "horizon")
        * _positive_int(frequency_hz, "frequency_hz")
        for seconds in horizons_seconds
    )
    if not horizon_steps or max(horizon_steps) > predicted.shape[1]:
        raise ValueError("benchmark horizon exceeds predicted trajectory")

    max_steps = max(horizon_steps)
    predicted_xy = np.empty(
        (predicted.shape[0], max_steps, 2), dtype=np.float64
    )
    for index in range(predicted.shape[0]):
        predicted_xy[index] = integrate_trajectory(
            predicted[index, :max_steps, 0],
            predicted[index, :max_steps, 1],
            float(speeds[index]),
            dt=1.0 / frequency_hz,
        )

    errors = np.linalg.norm(predicted_xy - target[:, :max_steps, :], axis=2)
    metrics: dict[str, float] = {}
    for seconds, steps in zip(horizons_seconds, horizon_steps):
        metrics[f"ade_{seconds}s"] = float(errors[:, :steps].mean())
        metrics[f"fde_{seconds}s"] = float(errors[:, steps - 1].mean())
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"non-finite benchmark metrics: {metrics}")
    return metrics, predicted_xy
