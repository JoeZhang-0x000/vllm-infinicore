from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from vllm_infinicore import platform

ROOT = Path(__file__).resolve().parents[1]


class PlatformPluginTests(unittest.TestCase):
    def test_register_platform_returns_infinicore_platform_class_path(self) -> None:
        self.assertEqual(
            platform.register_platform(),
            "vllm_infinicore.platform.InfiniCorePlatform",
        )

    def test_platform_entry_point_does_not_import_vllm_or_torch(self) -> None:
        code = """
import sys
from vllm_infinicore.platform import register_platform
print(register_platform())
print("torch" in sys.modules)
print("vllm" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.stdout.strip().splitlines(),
            [
                "vllm_infinicore.platform.InfiniCorePlatform",
                "False",
                "False",
            ],
        )


if __name__ == "__main__":
    unittest.main()
