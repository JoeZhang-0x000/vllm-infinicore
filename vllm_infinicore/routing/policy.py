"""Measured, opt-in mixed routing for the validated vendor versions.

Explicit operator lists (including ``all``) still force the selected kernels.
The ``recommended`` profile keeps attention computation native and only
enables auxiliary kernels supported by the measurements in docs/.
"""
from __future__ import annotations

import os

RECOMMENDED_ROUTES = {
    "ascend": ("RMSNorm", "Embedding", "MatMul", "LMHead", "StoreKVCache"),
    "metax": ("MatMul", "LMHead", "StoreKVCache"),
    # The old Kunlun build failed the 2K down-projection comparison. Its native
    # 2K-token cache writes also failed the independent CPU reference;
    # retain the verified InfiniCore store for all sizes on this backend.
    "kunlun": ("LMHead", "StoreKVCache"),
}


def recommended_routes(backend: str | None) -> tuple[str, ...]:
    if backend not in RECOMMENDED_ROUTES:
        raise ValueError(f"no measured recommended profile for backend {backend!r}")
    return RECOMMENDED_ROUTES[backend]


def recommended_selected() -> bool:
    return "recommended" in {
        token.strip().lower()
        for token in os.environ.get("VLLM_INFINICORE_ROUTES", "").split(",")
    }


def store_token_limit(backend: str) -> int | None:
    # 4/16/32-token decode stores were measured; large prefill stores on these
    # two backends were 6–32x slower. Shape selection happens before launch and
    # is fixed during graph capture, without a device synchronization.
    if recommended_selected() and backend in {"ascend", "metax"}:
        return 32
    return None
