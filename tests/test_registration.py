"""Registration must remain usable before torch and vendor plugins are loaded."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def test_disabled_registration_is_lazy_for_every_backend(self):
        for backend in ("", "ascend", "cuda", "kunlun", "metax"):
            with self.subTest(backend=backend):
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("VLLM_INFINICORE_")
                }
                env.update(
                    VLLM_INFINICORE_OPERATOR_BACKEND=backend,
                    VLLM_INFINICORE_ENABLE_PATCHES="0",
                    VLLM_INFINICORE_ROUTES="all",
                    VLLM_INFINICORE_ASCEND_LIBRARY="/unused/during/registration.so",
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; import vllm_infinicore as plugin; "
                        "result = plugin.register(); "
                        "assert result.route_count == 9; "
                        "assert not result.patching_enabled; "
                        "assert not result.installed_routes; "
                        "assert plugin.register() is result; "
                        "plugin.unregister(); "
                        "assert not any(name.split('.')[0] in "
                        "{'torch', 'vllm', 'vllm_ascend', 'vllm_metax', 'vllm_kunlun'} "
                        "for name in sys.modules)",
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
