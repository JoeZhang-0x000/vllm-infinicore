from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOG = logging.getLogger("vllm_infinicore")
_PATCHED = False
_ORIGINALS: dict[str, object] = {}
_COUNTERS: dict[str, int] = {}
_FA2_SHADOW_KV_CACHES: dict[int, Any] = {}
_CONFIG: "PatchConfig | None" = None
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "strict-infinicore.yaml"
_VALID_BACKENDS = {"original", "metax", "infinicore"}


@dataclass(frozen=True)
class OpConfig:
    backend: str


@dataclass(frozen=True)
class PatchConfig:
    version: int
    default_backend: str
    ops: dict[str, OpConfig]
    path: str | None = None


def _enabled() -> bool:
    return os.environ.get("VLLM_INFINICORE_PATCH", "0").lower() in {"1", "true", "yes", "on"}


def _counter(name: str, amount: int = 1) -> None:
    _COUNTERS[name] = _COUNTERS.get(name, 0) + amount
    counter_dir = os.environ.get("VLLM_INFINICORE_COUNTER_DIR")
    if counter_dir:
        path = Path(counter_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{os.getpid()}.json").write_text(
            json.dumps(get_patch_state(), sort_keys=True),
            encoding="utf-8",
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"patch config {path} must contain a YAML mapping")
    return data


def _validate_backend(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a string backend name")
    value = value.strip().lower()
    if value not in _VALID_BACKENDS:
        raise RuntimeError(f"{field} has unsupported backend {value!r}; expected one of {sorted(_VALID_BACKENDS)}")
    return value


def _load_config() -> PatchConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path = Path(os.environ.get("VLLM_INFINICORE_PATCH_CONFIG", _DEFAULT_CONFIG_PATH))
    raw = _load_yaml(path) if path.exists() else {}
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise RuntimeError("patch config defaults must be a mapping")
    default_backend = _validate_backend(defaults.get("backend", "infinicore"), field="defaults.backend")

    raw_ops = raw.get("ops") or {}
    if not isinstance(raw_ops, dict):
        raise RuntimeError("patch config ops must be a mapping")

    ops: dict[str, OpConfig] = {}
    for op_name, op_raw in raw_ops.items():
        if not isinstance(op_name, str):
            raise RuntimeError("patch config op names must be strings")
        if op_raw is None:
            op_raw = {}
        if isinstance(op_raw, str):
            op_raw = {"backend": op_raw}
        if not isinstance(op_raw, dict):
            raise RuntimeError(f"patch config for {op_name} must be a mapping or backend string")
        ops[op_name] = OpConfig(
            backend=_validate_backend(op_raw.get("backend", default_backend), field=f"ops.{op_name}.backend"),
        )

    version = int(raw.get("version", 1))
    if version != 1:
        raise RuntimeError(f"unsupported patch config version {version}; expected 1")

    _CONFIG = PatchConfig(version=version, default_backend=default_backend, ops=ops, path=str(path) if path.exists() else None)
    return _CONFIG


def _config_for(op_name: str) -> OpConfig:
    config = _load_config()
    return config.ops.get(op_name, OpConfig(backend=config.default_backend))


def _configured_backend(op_name: str) -> str:
    if op_name == "RoPE":
        return _config_for("ApplyRotaryEmb").backend
    return _config_for(op_name).backend


def get_patch_state() -> dict[str, Any]:
    config = _load_config()
    return {
        "patched": _PATCHED,
        "enabled": _enabled(),
        "config": {
            "path": config.path,
            "default_backend": config.default_backend,
            "ops": {name: cfg.__dict__ for name, cfg in sorted(config.ops.items())},
        },
        "counters": dict(sorted(_COUNTERS.items())),
    }


def _as_infini(tensor):
    return _as_infini_strided(tensor.contiguous() if not tensor.is_contiguous() else tensor)


def _as_infini_strided(tensor):
    import infinicore
    from infinicore.tensor import to_infinicore_dtype

    device_index = tensor.device.index if tensor.device.index is not None else 0
    infinicore.set_device(infinicore.device(tensor.device.type, device_index))
    return infinicore.strided_from_blob(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        dtype=to_infinicore_dtype(tensor.dtype),
        device=infinicore.device(tensor.device.type, device_index),
    )


def _rms_norm_infini(x, weight, eps):
    import torch
    import infinicore.nn.functional as F

    out = torch.empty_like(x)
    F.rms_norm(_as_infini(x), list(weight.shape), _as_infini(weight), eps, out=_as_infini(out))
    return out


def _silu_and_mul_infini(x):
    import torch
    import infinicore.nn.functional as F

    if os.environ.get("VLLM_INFINICORE_ALLOW_UNSAFE_SILU_AND_MUL", "0").lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "InfiniCore silu_and_mul segfaults on the current MetaX remote runtime; "
            "set VLLM_INFINICORE_ALLOW_UNSAFE_SILU_AND_MUL=1 to bypass this guard"
        )
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    F.silu_and_mul(_as_infini(x), out=_as_infini(out))
    return out


def _linear_infini(x, weight, bias=None):
    import torch
    import infinicore.nn.functional as F

    out_shape = x.shape[:-1] + (weight.shape[0],)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    F.linear(_as_infini(x), _as_infini(weight), None if bias is None else _as_infini(bias), out=_as_infini(out))
    return out


def _apply_rotary_infini(self, x, cos, sin):
    import torch
    import infinicore
    import infinicore.nn.functional as F
    from infinicore.nn.functional import RopeAlgo

    origin_shape = x.shape
    origin_dtype = x.dtype
    x_work = x.unsqueeze(0) if len(origin_shape) == 3 else x
    if self.enable_fp32_compute:
        x_work = x_work.float()
        cos = cos.float()
        sin = sin.float()
    bs, seq_len = x_work.shape[0], x_work.shape[1]
    pos = torch.arange(seq_len, device=x.device, dtype=torch.int32).view(1, seq_len).expand(bs, seq_len).contiguous()
    out = torch.empty_like(x_work)
    algo = RopeAlgo.GPT_NEOX if self.is_neox_style else RopeAlgo.GPT_J
    F.rope(_as_infini(x_work), infinicore.from_torch(pos), _as_infini(sin.contiguous()), _as_infini(cos.contiguous()), algo, out=_as_infini(out))
    if len(origin_shape) == 3:
        out = out.squeeze(0)
    if self.enable_fp32_compute:
        out = out.to(origin_dtype)
    return out


def _rope_infini(self, positions, query, key=None):
    import torch
    import infinicore
    import infinicore.nn.functional as F
    from infinicore.nn.functional import RopeAlgo

    if not self.is_neox_style:
        raise RuntimeError("InfiniCore RoPE adapter currently supports only NeoX-style RoPE")
    if self.rotary_dim % 2:
        raise RuntimeError(f"InfiniCore RoPE adapter requires an even rotary_dim, got {self.rotary_dim}")

    self._match_cos_sin_cache_dtype(query)
    cos_table, sin_table = self.cos_sin_cache.chunk(2, dim=-1)
    positions = positions.flatten().to(device=query.device, dtype=torch.int32).view(1, -1).contiguous()
    algo = RopeAlgo.GPT_NEOX

    def apply_one(x):
        if x is None:
            return None
        origin_dtype = x.dtype
        x_view = x.view(positions.shape[1], -1, self.head_size)
        rotary = x_view[..., : self.rotary_dim]
        if self.enable_fp32_compute:
            rotary = rotary.float()
            cos = cos_table.float()
            sin = sin_table.float()
        else:
            cos = cos_table
            sin = sin_table
        work = rotary.unsqueeze(0).contiguous()
        out = torch.empty_like(work)
        F.rope(
            _as_infini(work),
            infinicore.from_torch(positions),
            _as_infini(sin.contiguous()),
            _as_infini(cos.contiguous()),
            algo,
            out=_as_infini(out),
        )
        rotated = out.squeeze(0)
        if self.enable_fp32_compute:
            rotated = rotated.to(origin_dtype)
        if self.rotary_dim == self.head_size:
            return rotated.reshape_as(x)
        return torch.cat((rotated, x_view[..., self.rotary_dim :]), dim=-1).reshape_as(x)

    return apply_one(query), apply_one(key)


def _embedding_infini(input_, weight):
    import torch
    import infinicore.nn.functional as F

    out = torch.empty(input_.shape + (weight.shape[1],), dtype=weight.dtype, device=weight.device)
    F.embedding(_as_infini(input_), _as_infini(weight), out=_as_infini(out))
    return out


def _pa_cache_views(kv_cache, key):
    key_cache, value_cache = kv_cache.unbind(0)
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(f"expected 4D paged KV cache tensors, got {key_cache.shape}/{value_cache.shape}")
    num_kv_heads = key.shape[1]
    if key_cache.shape[1] == num_kv_heads:
        return _as_infini_strided(key_cache), _as_infini_strided(value_cache)
    if key_cache.shape[2] == num_kv_heads:
        return _as_infini_strided(key_cache.permute(0, 2, 1, 3)), _as_infini_strided(value_cache.permute(0, 2, 1, 3))
    raise RuntimeError(f"cannot infer KV cache layout from key={key.shape}, cache={key_cache.shape}")


def _fa2_cache_views(kv_cache, key):
    key_cache, value_cache = kv_cache.unbind(0)
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(f"expected 4D paged KV cache tensors, got {key_cache.shape}/{value_cache.shape}")
    num_kv_heads = key.shape[1]
    if key_cache.shape[2] == num_kv_heads:
        return _as_infini_strided(key_cache), _as_infini_strided(value_cache)
    if key_cache.shape[1] == num_kv_heads:
        return _as_infini_strided(key_cache.permute(0, 2, 1, 3)), _as_infini_strided(value_cache.permute(0, 2, 1, 3))
    raise RuntimeError(f"cannot infer FA2 KV cache layout from key={key.shape}, cache={key_cache.shape}")


def _fa2_vllm_cache_views(kv_cache, key):
    key_cache, value_cache = kv_cache.unbind(0)
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(f"expected 4D paged KV cache tensors, got {key_cache.shape}/{value_cache.shape}")
    num_kv_heads = key.shape[1]
    if key_cache.shape[2] == num_kv_heads:
        return _as_infini_strided(key_cache), _as_infini_strided(value_cache)
    if key_cache.shape[1] == num_kv_heads:
        return _as_infini_strided(key_cache.permute(0, 2, 1, 3)), _as_infini_strided(value_cache.permute(0, 2, 1, 3))
    raise RuntimeError(f"cannot infer FA2 KV cache layout from key={key.shape}, cache={key_cache.shape}")


def _debug_fa2_attention(label: str, **values: Any) -> None:
    if os.environ.get("VLLM_INFINICORE_DEBUG_ATTENTION", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    parts = []
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        stride = getattr(value, "stride", None)
        if shape is not None:
            stride_value = stride() if callable(stride) else stride
            parts.append(f"{name}=shape:{tuple(shape)} stride:{tuple(stride_value) if stride_value is not None else None}")
        else:
            parts.append(f"{name}={value!r}")
    LOG.warning("FA2 attention debug %s: %s", label, ", ".join(parts))


def _prefill_total_lens(attn_metadata):
    import torch

    cu_prefix_kv_lens = getattr(attn_metadata, "cu_prefix_kv_lens", None)
    if cu_prefix_kv_lens is not None:
        return (cu_prefix_kv_lens[1:] - cu_prefix_kv_lens[:-1]).to(torch.int64)

    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        raise RuntimeError("missing seq_lens for paged attention prefill")
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
    if seq_lens.shape[0] >= num_decodes + num_prefills:
        return seq_lens[num_decodes:num_decodes + num_prefills]
    if seq_lens.shape[0] == num_prefills:
        return seq_lens
    raise RuntimeError(f"cannot derive prefill total lengths from seq_lens={seq_lens.shape}, num_decodes={num_decodes}, num_prefills={num_prefills}")


def _store_kv_cache_infini(kv_cache, key, value, slot_mapping):
    import infinicore

    key_cache, value_cache = _fa2_cache_views(kv_cache, key)
    key_work = key.contiguous()
    value_work = value.contiguous()
    _debug_fa2_attention("store", kv_cache=kv_cache, key=key, value=value, slot_mapping=slot_mapping)
    infinicore.paged_caching(
        key_cache,
        value_cache,
        _as_infini(key_work),
        _as_infini(value_work),
        _as_infini(slot_mapping.flatten()),
    )


def _paged_attention_prefill_infini(self, query, key, kv_cache, attn_metadata, output):
    import infinicore

    key_cache, value_cache = _pa_cache_views(kv_cache, key)
    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    q = query[num_decode_tokens:num_actual_tokens]
    if q.numel() == 0:
        return
    out = output[num_decode_tokens:num_actual_tokens].view(q.shape)
    infinicore.paged_attention_prefill(
        _as_infini(q),
        key_cache,
        value_cache,
        _as_infini(attn_metadata.prefill_block_table),
        _as_infini(_prefill_total_lens(attn_metadata)),
        _as_infini(attn_metadata.prefill_query_start_loc),
        _as_infini(self.alibi_slopes) if self.alibi_slopes is not None else None,
        self.scale,
        out=_as_infini(out),
    )


def _paged_attention_decode_infini(self, query, key, kv_cache, attn_metadata, output):
    import infinicore

    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    if num_decode_tokens == 0:
        return
    if num_decode_tokens != num_decodes:
        raise RuntimeError(f"speculative decode is not supported by InfiniCore PA adapter: tokens={num_decode_tokens}, decodes={num_decodes}")

    key_cache, value_cache = _pa_cache_views(kv_cache, key)
    q = query[:num_decode_tokens]
    out = output[:num_decode_tokens].view(q.shape)
    infinicore.paged_attention(
        _as_infini(q),
        key_cache,
        value_cache,
        _as_infini(attn_metadata.decode_block_table),
        _as_infini(attn_metadata.decode_seq_lens),
        _as_infini(self.alibi_slopes) if self.alibi_slopes is not None else None,
        self.scale,
        out=_as_infini(out),
    )


def _mha_varlen_fa2_infini(self, query, key, kv_cache, attn_metadata, output):
    import infinicore

    raw_key_cache, raw_value_cache = kv_cache.unbind(0)
    key_cache, value_cache = _fa2_cache_views(kv_cache, key)
    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    q = query[num_decode_tokens:num_actual_tokens]
    if q.numel() == 0:
        return
    out = output[num_decode_tokens:num_actual_tokens].view(q.shape)
    _debug_fa2_attention(
        "prefill",
        query=query,
        key=key,
        q=q,
        output=output,
        out=out,
        kv_cache=kv_cache,
        raw_key_cache=raw_key_cache,
        raw_value_cache=raw_value_cache,
        num_heads=getattr(self, "num_heads", None),
        num_kv_heads=getattr(self, "num_kv_heads", None),
        head_size=getattr(self, "head_size", None),
        prefill_block_table=attn_metadata.prefill_block_table,
        prefill_query_start_loc=attn_metadata.prefill_query_start_loc,
        cu_prefix_kv_lens=attn_metadata.cu_prefix_kv_lens,
        max_query_len=int(attn_metadata.max_query_len),
        prefill_max_seq_len=int(attn_metadata.prefill_max_seq_len),
    )
    infinicore.mha_varlen(
        _as_infini(q),
        key_cache,
        value_cache,
        _as_infini(attn_metadata.prefill_query_start_loc),
        _as_infini(attn_metadata.cu_prefix_kv_lens),
        _as_infini(attn_metadata.prefill_block_table),
        int(attn_metadata.max_query_len),
        int(attn_metadata.prefill_max_seq_len),
        _as_infini(self.alibi_slopes) if self.alibi_slopes is not None else None,
        self.scale,
        out=_as_infini(out),
    )


def _mha_kvcache_fa2_infini(self, query, key, kv_cache, attn_metadata, output):
    import infinicore

    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    if num_decode_tokens == 0:
        return
    if num_decodes <= 0 or num_decode_tokens % num_decodes != 0:
        raise RuntimeError(f"unsupported InfiniCore FA2 decode shape: tokens={num_decode_tokens}, decodes={num_decodes}")

    raw_key_cache, raw_value_cache = kv_cache.unbind(0)
    key_cache, value_cache = _fa2_cache_views(kv_cache, key)
    q = query[:num_decode_tokens].view(num_decodes, num_decode_tokens // num_decodes, *query.shape[1:])
    out = output[:num_decode_tokens].view(q.shape)
    _debug_fa2_attention(
        "decode",
        query=query,
        q=q,
        output=output,
        out=out,
        kv_cache=kv_cache,
        raw_key_cache=raw_key_cache,
        raw_value_cache=raw_value_cache,
        num_heads=getattr(self, "num_heads", None),
        num_kv_heads=getattr(self, "num_kv_heads", None),
        head_size=getattr(self, "head_size", None),
        decode_block_table=attn_metadata.decode_block_table,
        decode_seq_lens=attn_metadata.decode_seq_lens,
    )
    infinicore.mha_kvcache(
        _as_infini(q),
        key_cache,
        value_cache,
        _as_infini(attn_metadata.decode_seq_lens),
        _as_infini(attn_metadata.decode_block_table),
        _as_infini(self.alibi_slopes) if self.alibi_slopes is not None else None,
        self.scale,
        out=_as_infini(out),
    )


def _pa_unsupported_reason(self, attn_metadata, output_scale, output_block_scale) -> str | None:
    if output_scale is not None or output_block_scale is not None:
        return "fused output quantization is not supported by InfiniCore PA adapter"
    if getattr(attn_metadata, "use_cascade", False):
        return "cascade attention is not supported by InfiniCore PA adapter"
    if getattr(self, "dcp_world_size", 1) > 1:
        return "DCP attention is not supported by InfiniCore PA adapter"
    attn_type = getattr(self, "attn_type", None)
    attn_type_name = getattr(attn_type, "name", str(attn_type))
    if "ENCODER" in attn_type_name:
        return "encoder attention is not supported by InfiniCore PA adapter"
    sliding_window = getattr(self, "sliding_window", None)
    if sliding_window is not None:
        values = sliding_window if isinstance(sliding_window, (tuple, list)) else (sliding_window,)
        if any(int(item) >= 0 for item in values):
            return "sliding-window attention is not supported by InfiniCore PA adapter"
    logits_soft_cap = getattr(self, "logits_soft_cap", None)
    if logits_soft_cap not in (None, 0, 0.0):
        return "logits soft cap is not supported by InfiniCore PA adapter"
    if getattr(self, "sinks", None) is not None:
        return "attention sinks are not supported by InfiniCore PA adapter"
    if getattr(self, "kv_cache_dtype", "auto").startswith("fp8"):
        return "FP8 KV cache is not supported by InfiniCore PA adapter"
    return None


def _call_backend(op_name: str, backend: str, call_original: Callable[[], Any], call_infinicore: Callable[[], Any] | None = None) -> Any:
    if backend in {"original", "metax"}:
        return call_original()
    if call_infinicore is None:
        raise RuntimeError(f"{op_name} has no InfiniCore adapter")
    _counter(f"{op_name}.{backend}.attempt")
    result = call_infinicore()
    _counter(f"{op_name}.{backend}.hit")
    return result


def _route(op_name: str, call_original: Callable[[], Any], call_infinicore: Callable[[], Any] | None = None) -> Any:
    backend = _configured_backend(op_name)
    try:
        return _call_backend(op_name, backend, call_original, call_infinicore)
    except Exception:
        _counter(f"{op_name}.{backend}.failure")
        raise


def _make_apply_rotary_patch(base_cls):
    class Patched(base_cls):
        def forward_oot(self, x, cos, sin):
            return _route(
                "ApplyRotaryEmb",
                lambda: super(Patched, self).forward_oot(x, cos, sin),
                lambda: _apply_rotary_infini(self, x, cos, sin),
            )

    Patched.__name__ = f"InfiniCore{base_cls.__name__}"
    Patched.__qualname__ = Patched.__name__
    Patched.__module__ = __name__
    Patched.name = getattr(base_cls, "name", "ApplyRotaryEmb")
    return Patched


def _patch_rotary_embedding() -> None:
    for module_name in (
        "vllm.model_executor.layers.rotary_embedding.base",
        "vllm.model_executor.layers.rotary_embedding",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, "RotaryEmbedding", None)
        if cls is None:
            continue
        original = cls.forward_oot
        if getattr(original, "_vllm_infinicore_patched", False):
            continue
        _ORIGINALS.setdefault(f"{module_name}.RotaryEmbedding.forward_oot", original)

        def forward_oot(self, positions, query, key=None, _original=original):
            return _route(
                "RoPE",
                lambda: _original(self, positions, query, key),
                lambda: _rope_infini(self, positions, query, key),
            )

        forward_oot._vllm_infinicore_patched = True
        cls.forward_oot = forward_oot
        LOG.warning("vllm-infinicore patch installed: %s.RotaryEmbedding.forward_oot", module_name)


def _make_silu_and_mul_patch(base_cls):
    class Patched(base_cls):
        def forward_oot(self, x):
            return _route(
                "SiluAndMul",
                lambda: super(Patched, self).forward_oot(x),
                lambda: _silu_and_mul_infini(x),
            )

    Patched.__name__ = f"InfiniCore{base_cls.__name__}"
    Patched.__qualname__ = Patched.__name__
    Patched.__module__ = __name__
    Patched.name = getattr(base_cls, "name", "SiluAndMul")
    return Patched


def _make_rms_norm_patch(op_name: str, base_cls):
    class Patched(base_cls):
        def forward_oot(self, x, residual=None):
            if self.variance_size_override is not None or not self.has_weight:
                return super(Patched, self).forward_oot(x, residual)
            weight = self.weight.data
            if op_name == "GemmaRMSNorm":
                weight = 1.0 + weight

            def original():
                return super(Patched, self).forward_oot(x, residual)

            def infini():
                if residual is not None:
                    merged = x + residual
                    return _rms_norm_infini(merged, weight, self.variance_epsilon), merged.to(x.dtype)
                return _rms_norm_infini(x, weight, self.variance_epsilon)

            return _route(op_name, original, infini)

    Patched.__name__ = f"InfiniCore{base_cls.__name__}"
    Patched.__qualname__ = Patched.__name__
    Patched.__module__ = __name__
    Patched.name = getattr(base_cls, "name", op_name)
    return Patched


def _install_registry_patch(name: str, patched_cls: type) -> None:
    from vllm.model_executor.custom_op import op_registry_oot

    if name not in _ORIGINALS:
        _ORIGINALS[name] = op_registry_oot[name]
    op_registry_oot[name] = patched_cls
    LOG.warning("vllm-infinicore patch installed: %s -> %s", name, patched_cls)


def _patch_embedding_and_linear() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    original_embedding = UnquantizedEmbeddingMethod.embedding
    _ORIGINALS.setdefault("UnquantizedEmbeddingMethod.embedding", original_embedding)

    def embedding(self, layer, input_):
        return _route("Embedding", lambda: original_embedding(self, layer, input_), lambda: _embedding_infini(input_, layer.weight))

    UnquantizedEmbeddingMethod.embedding = embedding

    original_apply = UnquantizedEmbeddingMethod.apply
    _ORIGINALS.setdefault("UnquantizedEmbeddingMethod.apply", original_apply)

    def lm_head(self, layer, x, bias=None, **kwargs):
        def infini():
            output = _linear_infini(x, layer.weight, bias)
            residual = kwargs.get("residual")
            return output + residual if residual is not None else output

        return _route("LMHead", lambda: original_apply(self, layer, x, bias, **kwargs), infini)

    UnquantizedEmbeddingMethod.apply = lm_head

    original_linear = UnquantizedLinearMethod.apply
    _ORIGINALS.setdefault("UnquantizedLinearMethod.apply", original_linear)

    def matmul(self, layer, x, bias=None, **kwargs):
        def infini():
            output = _linear_infini(x, layer.weight, bias)
            residual = kwargs.get("residual")
            return output + residual if residual is not None else output

        return _route("MatMul", lambda: original_linear(self, layer, x, bias, **kwargs), infini)

    UnquantizedLinearMethod.apply = matmul


def _patch_attention() -> None:
    try:
        flash_attn = importlib.import_module("vllm_metax.v1.attention.backends.flash_attn")
    except Exception as exc:
        LOG.warning("attention patch skipped: %s", exc)
        return

    impl_cls = getattr(flash_attn, "FlashAttentionImpl", None)
    if impl_cls is None:
        LOG.warning("attention patch skipped: FlashAttentionImpl not found")
        return

    original = impl_cls.forward
    if getattr(original, "_vllm_infinicore_patched", False):
        return
    _ORIGINALS.setdefault("vllm_metax.FlashAttentionImpl.forward", original)

    original_update = getattr(impl_cls, "do_kv_cache_update", None)
    if original_update is not None:
        _ORIGINALS.setdefault("vllm_metax.FlashAttentionImpl.do_kv_cache_update", original_update)

        def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
            return _route(
                "StoreKVCache",
                lambda: original_update(self, layer, key, value, kv_cache, slot_mapping),
                lambda: _store_kv_cache_infini(kv_cache, key, value, slot_mapping),
            )

        do_kv_cache_update._vllm_infinicore_patched = True
        impl_cls.do_kv_cache_update = do_kv_cache_update

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        def call_original():
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

        if output is None or attn_metadata is None:
            return call_original()
        unsupported_reason = _pa_unsupported_reason(self, attn_metadata, output_scale, output_block_scale)
        if unsupported_reason is not None:
            raise RuntimeError(unsupported_reason)
        if int(getattr(attn_metadata, "num_prefills", 0)) > 0:
            _route(
                "MhaVarlenFA2",
                call_original,
                lambda: _mha_varlen_fa2_infini(self, query, key, kv_cache, attn_metadata, output),
            )
        if int(getattr(attn_metadata, "num_decodes", 0)) > 0:
            num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
            num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
            if num_decode_tokens == num_decodes:
                _route(
                    "PagedAttentionDecode",
                    call_original,
                    lambda: _paged_attention_decode_infini(self, query, key, kv_cache, attn_metadata, output),
                )
            else:
                _route(
                    "MhaKVCacheFA2",
                    call_original,
                    lambda: _mha_kvcache_fa2_infini(self, query, key, kv_cache, attn_metadata, output),
                )
        return output

    forward._vllm_infinicore_patched = True
    impl_cls.forward = forward


def apply_patches() -> bool:
    global _PATCHED
    if not _enabled():
        LOG.warning("VLLM_INFINICORE_PATCH is not enabled; no patches applied")
        return False
    if _PATCHED:
        return True

    _load_config()
    import vllm_metax
    from vllm.model_executor.custom_op import op_registry_oot

    vllm_metax.register_customized()
    required = ["RMSNorm", "GemmaRMSNorm", "SiluAndMul", "ApplyRotaryEmb"]
    missing = [name for name in required if name not in op_registry_oot]
    if missing:
        raise RuntimeError(f"Missing expected OOT registrations: {missing}")

    _install_registry_patch("RMSNorm", _make_rms_norm_patch("RMSNorm", op_registry_oot["RMSNorm"]))
    _install_registry_patch("GemmaRMSNorm", _make_rms_norm_patch("GemmaRMSNorm", op_registry_oot["GemmaRMSNorm"]))
    _install_registry_patch("SiluAndMul", _make_silu_and_mul_patch(op_registry_oot["SiluAndMul"]))
    _install_registry_patch("ApplyRotaryEmb", _make_apply_rotary_patch(op_registry_oot["ApplyRotaryEmb"]))
    _patch_rotary_embedding()
    _patch_embedding_and_linear()
    _patch_attention()

    _PATCHED = True
    return True
