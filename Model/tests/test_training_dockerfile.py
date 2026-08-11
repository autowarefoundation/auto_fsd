"""Training image contract tests."""

from pathlib import Path


def test_training_image_supports_g6_and_g7_gpu_families():
    dockerfile = (
        Path(__file__).parents[2]
        / "Platform"
        / "docker"
        / "training"
        / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert (
        "FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
        in dockerfile
    )
    assert "torch.__version__.startswith('2.7.1')" in dockerfile
    assert "torch.version.cuda == '12.8'" in dockerfile
