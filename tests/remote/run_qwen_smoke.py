import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ["VLLM_INFINICORE_ENABLE_PATCHES"] = "1"
os.environ.setdefault("VLLM_INFINICORE_ROUTES", "all")
os.environ.setdefault("VLLM_PLUGINS", "metax,vllm_infinicore")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import vllm_infinicore
registration = vllm_infinicore.register()
print("registration", registration, flush=True)

def main():
    model = os.environ.get("MODEL", "/mnt/geogpt-doc-new/default/infinilm-models/DeepSeek-R1-Distill-Qwen-7B")
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

    llm_kwargs = {
        "model": model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": True,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = distributed_executor_backend
    print("llm_kwargs", llm_kwargs, flush=True)
    llm = LLM(**llm_kwargs)
    outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.0))
    for out in outputs:
        print("OUTPUT", out.outputs[0].text.replace("\n", " "), flush=True)
    print("VLLM_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
