"""L2D immutable source and reactive packing workflow contracts."""

from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("flytekit")

from data_parsing.l2d.dataset import L2DDataset  # noqa: E402
from data_processing.source_revisions import L2D_DATA_REVISION  # noqa: E402
from Platform.pipelines import workflows  # noqa: E402


def test_l2d_source_revision_is_one_immutable_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", L2D_DATA_REVISION)
    assert workflows.L2D_SOURCE_REVISION == L2D_DATA_REVISION
    assert (
        inspect.signature(L2DDataset.__init__)
        .parameters["revision"]
        .default
        == L2D_DATA_REVISION
    )


def test_l2d_reactive_dataset_version_includes_heading_contract_fix():
    assert workflows.L2D_REACTIVE_DATASET_VERSION == "v3.0-reactive-v2"


def test_packed_episode_count_uses_explicit_partition_groups():
    assert workflows._packed_episode_count(0, ["185"]) == 1
    assert workflows._packed_episode_count(3, None) == 3
    assert workflows._packed_episode_count(0, []) == 0


def test_l2d_pack_passes_source_revision_into_dataset():
    source = inspect.getsource(workflows.data_processing.task_function)

    assert source.count("revision=source_revision") >= 2
    assert (
        '"episodes": _packed_episode_count(episodes, group_ids)'
        in source
    )
