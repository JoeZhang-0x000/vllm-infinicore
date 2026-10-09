#!/usr/bin/env python3
"""Build the locked modular InfiniRT/InfiniOps stack for the C++ bridge.

Requires CMake, Ninja, libclang, nlohmann_json 3.12.0 and the vendor SDK.
MetaX builds fetch the pinned upstream optimization patches by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from infinicore_build.patches import prepare_ops_source
from infinicore_build.sources import sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "vllm_infinicore/infinicore.lock.json"
OPERATORS = (
    "gemm",
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "rotary_embedding",
    "embedding",
    "paged_caching_infinilm",
    "paged_attention_infinilm",
    "paged_attention_prefill_infinilm",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--prefix", type=Path, required=True, help="Use a new prefix for the modular stack"
    )
    parser.add_argument("--platform", choices=("metax", "cuda"), default="metax")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Verified archive fingerprints, when git is unavailable",
    )
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument(
        "--cmake-option", action="append", default=[], help="Additional -D... CMake option"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--cxx11-abi", choices=(0, 1), type=int, default=1, help="Must match the installed PyTorch"
    )
    parser.add_argument(
        "--metax-patches",
        choices=("upstream", "local", "none"),
        help="Pinned upstream patches (MetaX default), local offline copies, or no patches",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    patch_mode = args.metax_patches or ("upstream" if args.platform == "metax" else "none")
    if patch_mode != "none" and args.platform != "metax":
        parser.error("MetaX patches require --platform metax")

    source, prefix = args.source.resolve(), args.prefix.resolve()
    lock = json.loads(LOCK.read_text())
    verify_source(source, lock, args.source_manifest)
    build = (args.build_dir or prefix.parent / "build-infinicore").resolve()
    ops_source, patches = prepare_ops_source(
        source,
        build,
        ROOT / "scripts/patches",
        lock["components"]["InfiniOps"]["revision"],
        patch_mode,
    )
    backend = "METAX" if args.platform == "metax" else "NVIDIA"
    common = [
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        f"-DWITH_{backend}=ON",
        "-DWITH_CPU=OFF",
        "-DAUTO_DETECT_DEVICES=OFF",
        f"-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}",
    ]
    for component in ("InfiniRT", "InfiniOps"):
        options = list(common)
        if component == "InfiniOps":
            options += [
                f"-DINFINI_RT_ROOT={prefix}",
                "-DAUTO_DETECT_BACKENDS=OFF",
                "-DWITH_TORCH=OFF",
                "-DWITH_LINKED=OFF",
                "-DGENERATE_PYTHON_BINDINGS=OFF",
                "-DINFINI_OPS_OPS=" + ",".join(OPERATORS),
            ]
        component_source = (
            ops_source if component == "InfiniOps" else source / "submodules" / component
        )
        tree = build / component
        subprocess.run(
            [
                args.cmake,
                "-S",
                str(component_source),
                "-B",
                str(tree),
                *options,
                *args.cmake_option,
            ],
            check=True,
        )
        subprocess.run([args.cmake, "--build", str(tree), "-j", str(args.jobs)], check=True)
        subprocess.run([args.cmake, "--install", str(tree)], check=True)

    manifest = {
        key: lock[key] for key in ("repository", "revision", "resolved_at", "components", "api")
    }
    manifest.update(
        platform=args.platform,
        source=str(source),
        build_dir=str(build),
        cxx11_abi=args.cxx11_abi,
        operators=OPERATORS,
        maca_path=os.getenv("MACA_PATH"),
        libraries={
            name: sha256(prefix / "lib" / name) for name in ("libinfinirt.so", "libinfiniops.so")
        },
    )
    if patches:
        manifest["upstream_patches"] = patches
        manifest["patch_source"] = patch_mode
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"INFINI_ROOT={prefix}")


if __name__ == "__main__":
    main()
