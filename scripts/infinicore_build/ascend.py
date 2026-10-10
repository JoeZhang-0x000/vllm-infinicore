"""Ascend patches for the locked legacy InfiniCore C API source tree."""

from pathlib import Path

from .patches import prepare_patched_source


def prepare_ascend_source(
    source: Path, build: Path, patch_dir: Path, revision: str, mode: str
) -> tuple[Path, dict | None]:
    if mode not in {"local", "none"}:
        raise ValueError(f"Unsupported Ascend patch mode: {mode}")
    return prepare_patched_source(
        source,
        build,
        patch_dir,
        revision,
        mode,
        component="InfiniCore",
        marker_name=".vllm-infinicore-ascend-patches.json",
        directories=("include", "src"),
    )
