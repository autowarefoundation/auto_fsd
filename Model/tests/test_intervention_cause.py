"""Tests for counterfactual cause selection from intervention deltas (#122)."""

from __future__ import annotations

from data_processing.reasoning_label_generation.intervention_cause import (
    InterventionCauseCandidate,
    aggregate_candidate_deltas,
    horizon_labels_from_intervention_deltas,
    select_causes_by_intervention,
)
from data_processing.reasoning_label_generation.schema import HORIZON_SECONDS


def test_select_filters_and_ranks():
    cands = [
        InterventionCauseCandidate("pedestrian", 0.05),
        InterventionCauseCandidate("red_light", 0.40),
        InterventionCauseCandidate("parked_car", 0.001),  # below floor
    ]
    got = select_causes_by_intervention(cands, min_delta=0.01, relative_to_max=0.5)
    assert got == ["red_light"]  # pedestrian is < 0.5 * 0.40


def test_top_k():
    cands = aggregate_candidate_deltas({"a": 0.9, "b": 0.8, "c": 0.7})
    assert select_causes_by_intervention(cands, min_delta=0.0, top_k=2, relative_to_max=None) == [
        "a",
        "b",
    ]


def test_horizon_labels_provenance():
    per_h = {
        0.0: aggregate_candidate_deltas({"pedestrian": 0.5, "noise": 0.0}, horizon_sec=0.0),
        1.0: [],
    }
    labels = horizon_labels_from_intervention_deltas(per_h, min_delta=0.01, relative_to_max=None)
    assert len(labels) == len(HORIZON_SECONDS)
    assert labels[0].cause == ["pedestrian"]
    assert labels[0].provenance == "counterfactual_gt"
    assert labels[0].confidence > 0
    assert labels[1].cause == []
    assert labels[1].confidence == 0.0
