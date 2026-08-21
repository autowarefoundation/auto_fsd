"""KITScenes temporal sampling and fixed model-ABI constants."""

from __future__ import annotations

KITSCENES_ABI_HISTORY_STEPS = 64
KITSCENES_ABI_FUTURE_STEPS = 64

KITSCENES_TRAINING_HISTORY_STEPS = 64
KITSCENES_TRAINING_FUTURE_STEPS = 64

KITSCENES_BENCHMARK_HISTORY_STEPS = 40
KITSCENES_BENCHMARK_FUTURE_STEPS = 50


def kitscenes_temporal_contract(
    *,
    benchmark_protocol: bool,
) -> dict[str, int | str]:
    """Return the sampling margins and unchanged tensor ABI."""
    history_steps = (
        KITSCENES_BENCHMARK_HISTORY_STEPS
        if benchmark_protocol
        else KITSCENES_TRAINING_HISTORY_STEPS
    )
    future_steps = (
        KITSCENES_BENCHMARK_FUTURE_STEPS
        if benchmark_protocol
        else KITSCENES_TRAINING_FUTURE_STEPS
    )
    return {
        "mode": (
            "paper_protocol_approximation"
            if benchmark_protocol
            else "training"
        ),
        "sampling_history_steps": history_steps,
        "sampling_future_steps": future_steps,
        "abi_history_steps": KITSCENES_ABI_HISTORY_STEPS,
        "abi_future_steps": KITSCENES_ABI_FUTURE_STEPS,
        "history_padding": "left_zero",
        "future_control_padding": "right_zero",
        "future_gps_padding": "repeat_last",
    }
