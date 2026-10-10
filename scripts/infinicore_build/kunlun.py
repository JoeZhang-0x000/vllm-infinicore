"""Isolated compatibility patches for the locked legacy Kunlun source."""

from pathlib import Path

from .patches import prepare_patched_source


def prepare_kunlun_source(
    source: Path, build: Path, patch_dir: Path, revision: str, mode: str
) -> tuple[Path, dict | None]:
    if mode not in {"local", "none"}:
        raise ValueError(f"Unsupported Kunlun patch mode: {mode}")
    return prepare_patched_source(
        source,
        build,
        patch_dir,
        revision,
        mode,
        component="InfiniCore",
        marker_name=".vllm-infinicore-kunlun-patches.json",
        ignored_names=(".xmake", "build", "dist"),
    )
