#!/usr/bin/env python3
"""Build only the locked InfiniCore operators used by the Ascend adapter.

Requires git, CMake, C++17 and an initialized CANN development environment.
No InfiniRT, device management, Python InfiniCore or optional submodules are built.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from infinicore_build.ascend import prepare_ascend_source
from infinicore_build.sources import sha256, verify_git_source

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "vllm_infinicore/infinicore.lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Existing clean checkout at the locked revision"
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--soc", required=True, help="e.g. Ascend910B4; must match the target NPU")
    parser.add_argument("--cann", default=os.getenv("ASCEND_TOOLKIT_HOME"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--ascend-patches", choices=("local", "none"), default="local")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())["legacy_ascend"]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = (args.source or build / "InfiniCore").resolve()
    if not source.exists():
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "--depth=1",
                lock["repository"],
                lock["revision"],
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    verify_git_source(source, lock["revision"])
    operator_source, patches = prepare_ascend_source(
        source, build, ROOT / "scripts/patches/ascend", lock["revision"], args.ascend_patches
    )
    if not args.cann or not Path(args.cann).is_dir():
        parser.error("--cann or ASCEND_TOOLKIT_HOME must point to the CANN toolkit")
    cmake_dir = Path(tempfile.mkdtemp(prefix="cmake-", dir=build))
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "vllm_infinicore/operators/platforms/ascend/csrc"),
            "-B",
            str(cmake_dir),
            f"-DINFINICORE_SOURCE={operator_source}",
            f"-DINFINICORE_REVISION={lock['revision']}",
            f"-DASCEND_CANN_PACKAGE_PATH={args.cann}",
            f"-DSOC_VERSION={args.soc}",
            f"-DVLLM_INFINICORE_ASCEND_STRIDED_ROPE={'ON' if patches else 'OFF'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(cmake_dir), "-j", str(args.jobs)], check=True)
    # CANN's preprocess step rewrites object files in-place and cannot safely
    # reuse them on an incremental rebuild. Use a fresh tree for every build,
    # then atomically replace only the successfully linked output library.
    library = build / "libvllm_infinicore_ascend.so"
    (cmake_dir / library.name).replace(library)
    manifest = dict(
        lock,
        soc=args.soc,
        build_tree=str(cmake_dir),
        cann=str(Path(args.cann).resolve()),
        library=str(library),
        source=str(source),
        operator_source=str(operator_source),
        ascend_patches=patches,
        sha256=sha256(library),
    )
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"VLLM_INFINICORE_ASCEND_LIBRARY={library}")


if __name__ == "__main__":
    main()
