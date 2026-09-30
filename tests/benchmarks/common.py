"""Shared device setup, correctness gates, graph timing, and result storage."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def setup_device(platform, threads=4):
    import torch

    if platform == "ascend":
        import torch_npu  # noqa: F401: registers torch.npu

        api, device, dtype = torch.npu, "npu:0", torch.bfloat16
    elif platform in ("metax", "kunlun"):
        if platform == "kunlun":
            # XPytorch's NVML shim reports zero devices on the P800 runtime.
            torch.cuda._device_count_nvml = lambda: -1
            torch.cuda._cached_device_count = None
        api, device = torch.cuda, "cuda:0"
        dtype = torch.float16 if platform == "kunlun" else torch.bfloat16
    else:
        raise ValueError(f"unsupported platform: {platform}")
    api.set_device(0)
    torch.set_num_threads(threads)
    return api, device, dtype


def cpu(value, *, clone=False):
    """Copy device outputs without losing tuple/list boundaries."""
    if isinstance(value, (tuple, list)):
        return tuple(cpu(part, clone=clone) for part in value)
    result = value.detach().cpu()
    return result.clone() if clone else result


def _parts(value):
    if isinstance(value, (tuple, list)):
        return [part for child in value for part in _parts(child)]
    return [value]


def compare_outputs(actual, expected, *, dtype, exact=False):
    """Check every element on CPU, rejecting missing outputs and nonfinite values."""
    import torch

    left, right = _parts(cpu(actual)), _parts(cpu(expected))
    if not left or len(left) != len(right):
        raise ValueError(f"output counts differ or are empty: {len(left)}, {len(right)}")
    atol = 0.0 if exact else (0.02 if dtype == torch.bfloat16 else 0.005)
    rtol = 0.0 if exact else 0.02
    maximum, failures = 0.0, 0
    for a, b in zip(left, right):
        if a.shape != b.shape or a.numel() == 0:
            raise ValueError(f"output shapes differ or are empty: {a.shape}, {b.shape}")
        a, b = a.flatten(), b.flatten()
        for offset in range(0, a.numel(), 1 << 20):
            x = a[offset:offset + (1 << 20)].float()
            y = b[offset:offset + (1 << 20)].float()
            delta = (x - y).abs()
            finite = torch.isfinite(delta)
            if bool(finite.any()):
                maximum = max(maximum, delta[finite].max().item())
            failures += int((~finite | (delta > atol + rtol * y.abs())).sum())
    return {"passed": failures == 0, "max_abs_error": maximum,
            "failed_elements": failures, "atol": atol, "rtol": rtol}


def measure_graph(fn, reference, *, api, platform, dtype, unroll,
                  repeats=3, target_ms=30.0, warmup=0, output=None,
                  exact=False, nan_sentinel=False):
    """Time graph replay only; validate once before publishing any latency.

    Inputs, descriptors and layout conversion are prepared by the caller. A
    callback may return its output, or write to the supplied output tensor.
    Milliseconds are per operation, including any copies in the callback.
    """
    if unroll < 1 or repeats < 3 or target_ms <= 0:
        raise ValueError("need unroll >= 1, repeats >= 3 and target_ms > 0")
    graph = (api.NPUGraph if platform == "ascend" else api.CUDAGraph)()
    for _ in range(warmup):
        fn()
    api.synchronize()
    with api.graph(graph):
        for _ in range(unroll):
            returned = fn()
            if returned is not None:
                output = returned
    if output is None:
        raise ValueError("graph benchmark requires an output to validate")
    if nan_sentinel:
        for part in _parts(output):
            part.fill_(float("nan"))
    api.synchronize()
    start = time.perf_counter()
    graph.replay()
    api.synchronize()
    warm_ms = (time.perf_counter() - start) * 1000
    graph_check = compare_outputs(output, reference, dtype=dtype, exact=exact)
    if not graph_check["passed"]:
        return {"status": "graph_correctness_failed", "graph_check": graph_check}
    iterations = min(50, max(1, math.ceil(target_ms / max(warm_ms, 0.001))))
    device_ms, wall_ms = [], []
    for _ in range(repeats):
        begin, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        api.synchronize()
        start = time.perf_counter()
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        api.synchronize()
        wall_ms.append((time.perf_counter() - start) * 1000 / (iterations * unroll))
        device_ms.append(begin.elapsed_time(end) / (iterations * unroll))
    supported = all(math.isfinite(t) and t > 0 for t in device_ms)
    return {"status": "passed", "graph_check": graph_check,
            "unroll": unroll, "iterations_per_sample": iterations,
            "graph_device_ms": statistics.median(device_ms) if supported else None,
            "graph_wall_ms": statistics.median(wall_ms),
            "graph_device_samples_ms": device_ms, "graph_wall_samples_ms": wall_ms}


def microseconds(timing):
    """Keep the auxiliary-operator JSON schema while sharing the graph timer."""
    result = {}
    for key, value in timing.items():
        if key.endswith("_ms"):
            value = ([x * 1000 for x in value] if isinstance(value, list) else
                     None if value is None else value * 1000)
            key = key[:-3] + "_us"
        elif key == "iterations_per_sample":
            key = "iterations"
        result[key] = value
    return result


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def source_fingerprints():
    """Include shared helpers in benchmark provenance, not just the caller."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def cache_rope_tables(platform):
    """Opt-in, process-local cache ablation; this is not a production fix."""
    from vllm_infinicore.operators import backend, cpp_bridge
    import torch

    if platform == "ascend":
        from vllm_infinicore.operators.ascend import backend as ascend

        def rotary_embedding(positions, query, key, head_size, rotary_dim, cache, neox):
            supported, reason = ascend.supports_rotary_embedding(positions, head_size, rotary_dim)
            if not supported:
                raise ascend.Unsupported(reason)
            positions = ascend.nd(positions).contiguous()
            cache = ascend.nd(cache.to(device=query.device, dtype=query.dtype))
            cos, sin = (backend._contiguous_rope_table_cached(t) for t in cache.chunk(2, dim=-1))

            def apply(x):
                if x is None:
                    return None
                shaped = ascend.nd(x).contiguous().reshape(positions.numel(), -1, head_size)
                out = torch.empty_like(shaped)
                ascend.launch("RoPE", [out, shaped, positions, sin, cos], (int(neox),))
                return out.reshape(x.shape)

            return apply(query), apply(key)

        ascend.rotary_embedding = rotary_embedding
    else:
        def rotary_embedding(positions, query, key, head_size, cache, neox):
            module = cpp_bridge.module()
            cos, sin = (backend._contiguous_rope_table_cached(t) for t in cache.chunk(2, dim=-1))
            positions = positions.flatten().to(torch.int32)
            if cache.shape[0]:
                positions = positions.clamp(0, cache.shape[0] - 1)

            def apply(x):
                if x is None:
                    return None
                view = x.view(positions.shape[0], -1, head_size)
                out = module.rope_current_stream(view, positions, sin, cos, neox)
                cpp_bridge.record_call(cpp_bridge.ROPE_ROUTE)
                return out.reshape(x.shape)

            return apply(query), apply(key)

        backend._rotary_embedding_cpp_bridge = rotary_embedding
