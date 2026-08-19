"""#122 external-truth check for intervention cause labels.

#122 forbids shipping ``counterfactual_gt`` until the labeller is scored
against something the network did not invent. KIT LiDAR / HD-map conflict is
the intended anchor; this module runs the same precision/recall protocol on
constructed geometric-conflict scenes so the number exists without a local
KIT SDK.

Two labellers share the same scenes:

  * ``oracle`` — intervention L2 is high iff the factor is in geometric
    conflict with the ego path (privileged / external-aligned).
  * ``blind``  — the model attends to a distractor and misses the conflict
    (the circularity failure #122 described).

Also reports Jaccard agreement with ``MockTeacher`` (an independent
correlational labeller) to quantify circularity vs a non-sensitivity source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from data_processing.reasoning_label_generation.intervention_cause import (
    InterventionCauseCandidate,
    select_causes_by_intervention,
)
from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.teacher_client import TeacherRequest


@dataclass(frozen=True)
class Factor:
    name: str
    in_conflict: bool
    oracle_delta: float
    blind_delta: float


@dataclass(frozen=True)
class Scene:
    scene_id: str
    factors: tuple[Factor, ...]

    def truth(self) -> set[str]:
        return {f.name for f in self.factors if f.in_conflict}

    def candidates(self, labeller: str) -> list[InterventionCauseCandidate]:
        key = "oracle_delta" if labeller == "oracle" else "blind_delta"
        return [
            InterventionCauseCandidate(name=f.name, trajectory_l2=float(getattr(f, key)))
            for f in self.factors
        ]


# Constructed KIT-like conflicts: ego-path geometry is the external anchor,
# not the model's attention. Names are taxonomy ``cause`` labels.
SCENES: tuple[Scene, ...] = (
    Scene("ped_in_crosswalk", (
        Factor("pedestrian_crossing", True, 0.45, 0.00),
        Factor("lead_vehicle", False, 0.02, 0.38),
        Factor("vru_conflict", False, 0.01, 0.01),
    )),
    Scene("vru_entering_path", (
        Factor("vru_conflict", True, 0.51, 0.00),
        Factor("route_turn", False, 0.03, 0.29),
        Factor("occlusion", False, 0.00, 0.02),
    )),
    Scene("red_light_ahead", (
        Factor("red_light", True, 0.40, 0.40),
        Factor("poor_visibility", False, 0.00, 0.00),
        Factor("lead_vehicle", False, 0.05, 0.22),
    )),
    Scene("stopped_lead", (
        Factor("stopped_lead_vehicle", True, 0.36, 0.36),
        Factor("pedestrian_about_to_cross", False, 0.02, 0.02),
        Factor("route_lane_change", False, 0.01, 0.01),
    )),
    Scene("cut_in", (
        Factor("cut_in_vehicle", True, 0.48, 0.00),
        Factor("slow_lead_vehicle", False, 0.04, 0.41),
        Factor("unknown_cause", False, 0.00, 0.00),
    )),
    Scene("cross_traffic", (
        Factor("cross_traffic", True, 0.44, 0.00),
        Factor("oncoming_vehicle", False, 0.03, 0.33),
        Factor("yellow_light", False, 0.02, 0.02),
    )),
    Scene("object_blocking", (
        Factor("object_blocking_path", True, 0.39, 0.39),
        Factor("construction_blocking_path", False, 0.04, 0.04),
        Factor("poor_visibility", False, 0.01, 0.01),
    )),
    Scene("stop_sign", (
        Factor("stop_sign", True, 0.31, 0.00),
        Factor("yield_sign", False, 0.02, 0.27),
        Factor("route_merge", False, 0.01, 0.01),
    )),
    Scene("lane_ending", (
        Factor("lane_ending", True, 0.34, 0.34),
        Factor("blocked_lane", False, 0.05, 0.05),
        Factor("lead_vehicle", False, 0.03, 0.03),
    )),
    Scene("ped_about_to_cross", (
        Factor("pedestrian_about_to_cross", True, 0.37, 0.00),
        Factor("human_direction", False, 0.00, 0.25),
        Factor("lead_vehicle", False, 0.04, 0.04),
    )),
    Scene("road_closed", (
        Factor("road_closed", True, 0.50, 0.50),
        Factor("route_turn", False, 0.06, 0.06),
        Factor("uncertainty_high", False, 0.02, 0.02),
    )),
    Scene("slippery_plus_lead", (
        Factor("slippery_road", True, 0.28, 0.00),
        Factor("slow_lead_vehicle", True, 0.33, 0.33),
        Factor("occlusion", False, 0.01, 0.24),
    )),
)


def _prf(pred: Iterable[str], truth: Iterable[str]) -> dict[str, float]:
    p, t = set(pred), set(truth)
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _micro(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_scenes": float(len(rows))}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def score_labeller(
    scenes: Sequence[Scene],
    labeller: str,
    *,
    min_delta: float = 1e-3,
    relative_to_max: float | None = 0.5,
    top_k: int | None = 3,
) -> dict[str, Any]:
    rows = []
    per_scene = []
    for scene in scenes:
        pred = select_causes_by_intervention(
            scene.candidates(labeller),
            min_delta=min_delta,
            relative_to_max=relative_to_max,
            top_k=top_k,
        )
        stats = _prf(pred, scene.truth())
        rows.append(stats)
        per_scene.append({
            "scene_id": scene.scene_id,
            "truth": sorted(scene.truth()),
            "pred": pred,
            **stats,
        })
    return {"micro": _micro(rows), "scenes": per_scene}


def mock_teacher_jaccard(
    scenes: Sequence[Scene],
    labeller: str,
    *,
    min_delta: float = 1e-3,
    relative_to_max: float | None = 0.5,
) -> dict[str, float]:
    teacher = MockTeacher()
    scores = []
    for scene in scenes:
        pred = set(select_causes_by_intervention(
            scene.candidates(labeller),
            min_delta=min_delta,
            relative_to_max=relative_to_max,
        ))
        rec = teacher.label(TeacherRequest(scene.scene_id, "kitscenes"))
        teacher_causes = set(rec.horizons[0].cause)
        scores.append(_jaccard(pred, teacher_causes))
    return {
        "mean_jaccard": float(sum(scores) / max(len(scores), 1)),
        "n": float(len(scores)),
    }


def run_cause_label_validation(
    *,
    min_delta: float = 1e-3,
    relative_to_max: float | None = 0.5,
) -> dict[str, Any]:
    oracle = score_labeller(SCENES, "oracle", min_delta=min_delta, relative_to_max=relative_to_max)
    blind = score_labeller(SCENES, "blind", min_delta=min_delta, relative_to_max=relative_to_max)
    return {
        "source": "constructed_geometric_conflict",
        "n_scenes": len(SCENES),
        "min_delta": min_delta,
        "relative_to_max": relative_to_max,
        "provenance_emitted": "counterfactual_weak",
        "oracle_vs_geometry": oracle["micro"],
        "blind_vs_geometry": blind["micro"],
        "oracle_vs_mock_teacher_jaccard": mock_teacher_jaccard(
            SCENES, "oracle", min_delta=min_delta, relative_to_max=relative_to_max,
        ),
        "blind_vs_mock_teacher_jaccard": mock_teacher_jaccard(
            SCENES, "blind", min_delta=min_delta, relative_to_max=relative_to_max,
        ),
        "oracle_scenes": oracle["scenes"],
        "blind_scenes": blind["scenes"],
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_cause_label_validation(min_delta=args.min_delta)
    # Keep the JSON small for the PR artifact.
    slim = {k: v for k, v in report.items() if k not in ("oracle_scenes", "blind_scenes")}
    slim["oracle_scenes"] = report["oracle_scenes"]
    text = json.dumps(slim, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
