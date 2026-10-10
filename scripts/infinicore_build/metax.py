"""MetaX optimization patches for the modular InfiniOps source tree."""

from pathlib import Path

from .patches import prepare_patched_source


def prepare_ops_source(
    source: Path, build: Path, patch_dir: Path, revision: str, mode: str
) -> tuple[Path, dict | None]:
    return prepare_patched_source(
        source / "submodules" / "InfiniOps",
        build,
        patch_dir,
        revision,
        mode,
        component="InfiniOps",
        marker_name=".vllm-infinicore-patches.json",
    )
