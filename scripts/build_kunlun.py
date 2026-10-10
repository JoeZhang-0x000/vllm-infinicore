#!/usr/bin/env python3
"""Build the locked Kunlun legacy stack using the vendor SDK and xmake.

The SDK must contain xre/{include,so}, xhpc/{xdnn,xblas}/{include,so},
and xtdk/bin/clang++. NVIDIA cublas headers cannot replace vendor XBLAS headers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from infinicore_build.kunlun import prepare_kunlun_source
from infinicore_build.sources import sha256, verify_git_source

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Clean checkout at the locked revision")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, help="Defaults to BUILD_DIR/install")
    parser.add_argument("--sdk", type=Path, default=os.getenv("KUNLUN_HOME"))
    parser.add_argument("--cuda", type=Path, default=os.getenv("CUDA_HOME", "/usr/local/cuda"))
    parser.add_argument("--xmake", default="xmake")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=0)
    parser.add_argument("--kunlun-patches", choices=("local", "none"), default="local")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.sdk is None:
        parser.error("Supply --sdk or KUNLUN_HOME")
    sdk = args.sdk.resolve()
    required = (
        "xre/include/xpu/runtime.h",
        "xhpc/xdnn/include/xpu/xdnn.h",
        "xhpc/xblas/include/xblas_api.h",
        "xhpc/xblas/include/cublasLt.h",
        "xtdk/bin/clang++",
    )
    for name in required:
        if not (sdk / name).is_file():
            parser.error(f"Incomplete vendor SDK: missing {sdk / name}")
    lock = json.loads((ROOT / "vllm_infinicore/infinicore.lock.json").read_text())["legacy_kunlun"]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = (args.source or build / "InfiniCore").resolve()
    if not source.exists():
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "fetch", "--depth=1", lock["repository"], lock["revision"]],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    verify_git_source(source, lock["revision"])
    if not (source / "third_party/spdlog/include/spdlog/spdlog.h").is_file():
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "submodule",
                "update",
                "--init",
                "--depth=1",
                "third_party/spdlog",
            ],
            check=True,
        )
    verify_git_source(source / "third_party/spdlog", lock["spdlog_revision"])
    copied, patches = prepare_kunlun_source(
        source, build, ROOT / "scripts/patches/kunlun", lock["revision"], args.kunlun_patches
    )
    prefix = (args.prefix or build / "install").resolve()
    environment = dict(
        os.environ,
        INFINI_ROOT=str(prefix),
        KUNLUN_HOME=str(sdk),
        CUDA_HOME=str(args.cuda.resolve()),
        XMAKE_ROOT="y",
    )
    configure = [
        args.xmake,
        "f",
        "-y",
        "-m",
        "release",
        "--kunlun-xpu=true",
        "--cpu=false",
        "--omp=false",
        "--ccl=false",
        "--cudnn=false",
        f"--cxxflags=-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}",
    ]
    subprocess.run(configure, cwd=copied, env=environment, check=True)
    subprocess.run(
        [args.xmake, "build", "-y", f"-j{args.jobs}", "infinicore_cpp_api"],
        cwd=copied,
        env=environment,
        check=True,
    )
    for target in ("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl"):
        subprocess.run(
            [args.xmake, "install", "-y", target], cwd=copied, env=environment, check=True
        )
    libraries = {
        name: sha256(prefix / "lib" / name)
        for name in (
            "libinfinicore_cpp_api.so",
            "libinfiniop.so",
            "libinfinirt.so",
            "libinfiniccl.so",
        )
    }
    manifest = dict(
        lock,
        source=str(source),
        operator_source=str(copied),
        sdk=str(sdk),
        prefix=str(prefix),
        cxx11_abi=args.cxx11_abi,
        patches=patches,
        libraries=libraries,
    )
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"INFINI_ROOT={prefix}")
    print(f"Add {prefix / 'lib'} to LD_LIBRARY_PATH for transitive legacy dependencies.")


if __name__ == "__main__":
    main()
