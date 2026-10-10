#!/usr/bin/env python3
"""Check or apply the measured BHLD cache fix for kunlun_ops 0.1.58+ee39020a.

Run with the experiment's Python interpreter. --apply keeps a byte-for-byte
backup and accepts only the tested source. Apply to an isolated environment
shared by native and InfiniCore comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ORIGINAL_SHA256 = "27bd11cd623d942d4ec2bea2e4feea0411cd8bc08b06f54a35b8d509734628ca"
PATCHED_SHA256 = "fe8437d8133da80a8ac0760d1000c44b47b6ac1edf2afc982eb5cba11da74bc3"
NEEDLE = """    if key_cache.dtype is torch.int8 and force_sdnn:
        raise ValueError("reshape and cache flash use sdnn do not support quant")
"""
REPLACEMENT = (
    """    # Experiment environment fix for kunlun_ops 0.1.58+ee39020a.
    # Its Flash BHLD V-store corrupts multi-token writes. This is the vendor's
    # public store_paged_kv_cache implementation, independently checked on CPU.
    if (not BLHD_LAYOUT and not force_sdnn and quant_mode == 0
            and key_cache.dtype is not torch.int8):
        return xpu_flash_ops.store_paged_kv_cache(
            key, key_cache, slot_mapping, value, value_cache, k_max, v_max)

"""
    + NEEDLE
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Patch the active isolated environment"
    )
    args = parser.parse_args()
    module = importlib.util.find_spec("kunlun_ops")
    if module is None or module.origin is None:
        parser.error("kunlun_ops is not installed in this interpreter's environment")
    path = Path(module.origin).with_name("_cache.py")
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest not in {ORIGINAL_SHA256, PATCHED_SHA256}:
        raise ValueError(f"Untested vendor cache source: {path}, SHA256={digest}")
    backup = path.with_suffix(".py.original")
    if args.apply and digest == ORIGINAL_SHA256:
        if backup.exists():
            raise FileExistsError(f"Preserve/reconcile the existing backup: {backup}")
        source = original.decode()
        if source.count(NEEDLE) != 1:
            raise ValueError("Vendor cache patch context does not match")
        patched = source.replace(NEEDLE, REPLACEMENT).encode()
        if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
            raise ValueError("Patched cache checksum mismatch")
        shutil.copy2(path, backup)
        path.write_bytes(patched)
        digest = PATCHED_SHA256
    print(
        json.dumps(
            dict(
                path=str(path), backup=str(backup), sha256=digest, patched=digest == PATCHED_SHA256
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
