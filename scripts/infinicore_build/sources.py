"""Verify an InfiniCore checkout or a fingerprinted source archive."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *arguments], text=True).strip()


def verify_source(source: Path, lock: dict, manifest_path: Path | None) -> None:
    if manifest_path:
        # Fingerprinted archives support experiment containers without git.
        manifest = json.loads(manifest_path.read_text())
        keys = ("repository", "revision", "components")
        if any(manifest.get(key) != lock[key] for key in keys):
            raise RuntimeError("Source manifest does not match the InfiniCore lock")
        if not manifest.get("files"):
            raise RuntimeError("Source manifest has no file fingerprints")
        for name, expected in manifest["files"].items():
            path = (source / name).resolve()
            if not path.is_relative_to(source) or sha256(path) != expected:
                raise RuntimeError(f"Source fingerprint mismatch: {name}")
        return

    revisions = [(source, lock["revision"])] + [
        (source / "submodules" / name, component["revision"])
        for name, component in lock["components"].items()
    ]
    for path, revision in revisions:
        if _git(path, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"Source revision does not match the lock: {path}")
        if _git(path, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"Source checkout has tracked modifications: {path}")
