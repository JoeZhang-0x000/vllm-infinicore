"""Fetch pinned upstream patches and apply them to an independent source copy."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .sources import sha256


def _patch_file(patch: dict, mode: str, local: Path, cache: Path) -> Path:
    path = local / patch["file"] if mode == "local" else cache / patch["file"]
    if mode == "upstream" and not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(patch["url"], timeout=30) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot fetch {patch['name']}; use --metax-patches local for offline builds"
            ) from exc
        # Validate before exposing a downloaded file to the source preparation.
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
            temporary.write(data)
            downloaded = Path(temporary.name)
        try:
            if sha256(downloaded) != patch["sha256"]:
                raise RuntimeError(f"Downloaded patch checksum mismatch: {patch['name']}")
            downloaded.replace(path)
        finally:
            downloaded.unlink(missing_ok=True)
    if sha256(path) != patch["sha256"]:
        raise RuntimeError(f"Patch checksum mismatch: {patch['name']}")
    return path


def _verify_patched_files(source: Path, fingerprints: dict[str, str]) -> None:
    for name, expected in fingerprints.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != expected:
            raise RuntimeError(f"Patched source fingerprint mismatch: {name}")


def prepare_ops_source(
    source: Path, build: Path, patch_dir: Path, revision: str, mode: str
) -> tuple[Path, dict | None]:
    original = source / "submodules" / "InfiniOps"
    if mode == "none":
        return original, None

    manifest = json.loads((patch_dir / "manifest.json").read_text())
    if manifest["base_revision"] != revision:
        raise RuntimeError("Patch manifest does not match the locked InfiniOps revision")
    patches = manifest["patches"]
    if not patches:
        raise RuntimeError("Patch manifest is empty")
    files = [_patch_file(patch, mode, patch_dir, build / "patch-cache") for patch in patches]

    sources = build / "sources"
    copied = sources / "InfiniOps"
    marker_name = ".vllm-infinicore-patches.json"
    if copied.exists():
        marker = copied / marker_name
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise RuntimeError("Use a fresh build directory for a different patch set")
        _verify_patched_files(copied, manifest["files"])
        return copied, manifest

    sources.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".infiniops-", dir=sources) as temporary:
        staging = Path(temporary) / "InfiniOps"
        shutil.copytree(original, staging, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        for path in files:
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(path)],
                cwd=staging,
                check=True,
            )
        _verify_patched_files(staging, manifest["files"])
        (staging / marker_name).write_text(json.dumps(manifest, indent=2) + "\n")
        staging.replace(copied)
    return copied, manifest
