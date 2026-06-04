from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "/mnt/geogpt-doc-new/default/infinilm-models/DeepSeek-R1-Distill-Qwen-7B"
METAX_PLUGINS = "metax,vllm_infinicore"
NO_METAX_PLUGINS = "infinicore,vllm_infinicore"


def configure_environment() -> dict[str, str]:
    configure_runtime_environment()
    os.environ.setdefault("VLLM_INFINICORE_ENABLE_PATCHES", "1")
    os.environ.setdefault("VLLM_INFINICORE_ROUTES", "all")
    os.environ.setdefault("VLLM_INFINICORE_FORCE_NATIVE_FALLBACK", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_INFINICORE_STRICT_BACKEND", "1")
    os.environ.setdefault("VLLM_PLUGINS", default_plugins())

    return {
        "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS", ""),
        "VLLM_INFINICORE_ENABLE_PATCHES": os.environ.get(
            "VLLM_INFINICORE_ENABLE_PATCHES",
            "",
        ),
        "VLLM_INFINICORE_ROUTES": os.environ.get("VLLM_INFINICORE_ROUTES", ""),
        "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK": os.environ.get(
            "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK",
            "",
        ),
        "VLLM_INFINICORE_STRICT_BACKEND": os.environ.get(
            "VLLM_INFINICORE_STRICT_BACKEND",
            "",
        ),
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            "",
        ),
    }


def configure_runtime_environment() -> None:
    site_packages = "/opt/conda/lib/python3.12/site-packages"
    maca_path = "/opt/maca-3.5.3"
    infini_root = os.path.expanduser("~/.infini")
    torch_lib = f"{site_packages}/torch/lib"

    os.environ.setdefault("MACA_PATH", maca_path)
    os.environ.setdefault("MACA_HOME", maca_path)
    os.environ.setdefault("MACA_ROOT", maca_path)
    os.environ.setdefault("INFINI_ROOT", infini_root)
    os.environ.setdefault("PYTHON_SITE_PACKAGES", site_packages)
    os.environ.setdefault("TORCH_LIB", torch_lib)
    os.environ.setdefault(
        "FLASH_ATTN_2_CUDA_SO",
        f"{site_packages}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so",
    )
    os.environ.setdefault("XMAKE_ROOT", "y")

    prepend_env(
        "PATH",
        [
            "/mnt/geogpt-doc-new/default/xmake_env/bin",
            os.path.expanduser("~/.local/bin"),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            f"{maca_path}/bin",
        ],
    )
    prepend_env(
        "LD_LIBRARY_PATH",
        [
            "/opt/conda/lib",
            torch_lib,
            f"{infini_root}/lib",
            "/root/InfiniCore/python/infinicore/lib",
            "/root/InfiniCore/build/linux/x86_64/release",
            f"{maca_path}/lib",
            f"{maca_path}/lib64",
        ],
    )


def prepend_env(name: str, paths: list[str]) -> None:
    existing = [part for part in os.environ.get(name, "").split(":") if part]
    new_parts = [path for path in paths if path and path not in existing]
    os.environ[name] = ":".join([*new_parts, *existing])


def default_plugins() -> str:
    if env_truthy("VLLM_SMOKE_FORBID_METAX_LOAD") or env_truthy(
        "VLLM_INFINICORE_NO_METAX"
    ):
        return NO_METAX_PLUGINS
    return METAX_PLUGINS


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    environment = configure_environment()
    reexec_with_runtime_environment()

    import vllm_infinicore

    registration = vllm_infinicore.register()
    print("environment", environment, flush=True)
    print("registration", registration, flush=True)

    model = os.environ.get("MODEL", DEFAULT_MODEL)
    print("model", model, flush=True)
    max_model_len = int(os.environ.get("VLLM_SMOKE_MAX_MODEL_LEN", "256"))
    max_tokens = int(os.environ.get("VLLM_SMOKE_MAX_TOKENS", "8"))
    gpu_memory_utilization = float(
        os.environ.get("VLLM_SMOKE_GPU_MEMORY_UTILIZATION", "0.60")
    )
    prompt = os.environ.get("VLLM_SMOKE_PROMPT", "hello")
    tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    distributed_executor_backend = os.environ.get("VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
    ray_status = vllm_infinicore.configure_ray_environment(
        distributed_executor_backend=distributed_executor_backend,
    )
    print("ray_environment", ray_status, flush=True)

    from vllm import LLM, SamplingParams

    enforce_eager = env_truthy("VLLM_SMOKE_ENFORCE_EAGER")
    llm_kwargs = {
        "model": model,
        "dtype": os.environ.get("VLLM_SMOKE_DTYPE", "bfloat16"),
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": enforce_eager,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = distributed_executor_backend
    print("llm_kwargs", llm_kwargs, flush=True)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    for out in outputs:
        completion = out.outputs[0]
        output_token_ids = [int(token) for token in completion.token_ids]
        print("OUTPUT", completion.text.replace("\n", " "), flush=True)
        print("OUTPUT_TOKEN_COUNT", len(output_token_ids), flush=True)
        if len(output_token_ids) != max_tokens:
            raise RuntimeError(
                f"output_token_count={len(output_token_ids)}, expected={max_tokens}"
            )

    vllm_metax_loaded = metax_loaded(
        llm,
        use_ray=distributed_executor_backend == "ray",
    )
    print("vllm_metax_loaded", vllm_metax_loaded, flush=True)
    if env_truthy("VLLM_SMOKE_FORBID_METAX_LOAD") and vllm_metax_loaded:
        raise RuntimeError("vllm_metax_loaded=True")

    print("VLLM_SMOKE_OK", flush=True)
    return 0


def reexec_with_runtime_environment() -> None:
    if os.environ.get("VLLM_REMOTE_SMOKE_BOOTSTRAPPED") == "1":
        return
    os.environ["VLLM_REMOTE_SMOKE_BOOTSTRAPPED"] = "1"
    print("reexec_with_runtime_environment", flush=True)
    os.execvpe(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve())],
        os.environ,
    )


def metax_loaded(llm: Any | None = None, *, use_ray: bool = False) -> bool:
    local_loaded = local_metax_loaded()
    if use_ray and llm is not None:
        try:
            worker_loaded = any(
                bool(item) for item in llm.collective_rpc(worker_metax_loaded)
            )
        except Exception:
            worker_loaded = False
        return local_loaded or worker_loaded
    return local_loaded


def local_metax_loaded() -> bool:
    return any(
        name == "vllm_metax" or name.startswith("vllm_metax.")
        for name in sys.modules
    )


def worker_metax_loaded(_worker: Any) -> bool:
    return local_metax_loaded()


if __name__ == "__main__":
    raise SystemExit(main())
