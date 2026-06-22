from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

from vllm_infinicore.ops import cpp_bridge


class CppBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ENABLE_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ROUTES_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_DISABLE_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_TARGET_ENV, None)
        os.environ.pop(cpp_bridge.FLASH_DECODE_NUM_SPLITS_ENV, None)
        cpp_bridge.reset_bridge_call_counts()

    def tearDown(self) -> None:
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ENABLE_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ROUTES_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_DISABLE_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_TARGET_ENV, None)
        os.environ.pop(cpp_bridge.FLASH_DECODE_NUM_SPLITS_ENV, None)
        cpp_bridge.reset_bridge_call_counts()

    def test_flash_decode_enabled_by_default(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "cuda"

        self.assertEqual(
            cpp_bridge.selected_routes(),
            (
                cpp_bridge.FLASH_DECODE_ROUTE,
                cpp_bridge.MATMUL_ROUTE,
            ),
        )
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.FLASH_DECODE_ROUTE))
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.MATMUL_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.STORE_KV_CACHE_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.RMS_NORM_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.SILU_AND_MUL_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.ROPE_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.LM_HEAD_ROUTE))

    def test_musa_defaults_prefer_current_stream_routes(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "musa"

        self.assertEqual(cpp_bridge.selected_routes(), cpp_bridge.MUSA_DEFAULT_ROUTES)
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE))
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.EMBEDDING_ROUTE))
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.PREFILL_ROUTE))
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.LM_HEAD_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.FLASH_DECODE_ROUTE))

    def test_can_be_explicitly_disabled(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_DISABLE_ENV] = "1"

        self.assertEqual(cpp_bridge.selected_routes(), ())
        with self.assertRaisesRegex(cpp_bridge.CppBridgeError, "disabled"):
            cpp_bridge.module()

    def test_route_selection(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_ENABLE_ENV] = "1"
        os.environ[cpp_bridge.CPP_BRIDGE_ROUTES_ENV] = "PagedAttentionDecode,LMHead"

        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE))
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.LM_HEAD_ROUTE))

    def test_flash_decode_route_selection(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "cuda"
        os.environ[cpp_bridge.CPP_BRIDGE_ENABLE_ENV] = "1"
        os.environ[cpp_bridge.CPP_BRIDGE_ROUTES_ENV] = "PagedAttentionDecodeFlash"

        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.FLASH_DECODE_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE))

    def test_all_routes_include_flash_decode(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "cuda"
        os.environ[cpp_bridge.CPP_BRIDGE_ENABLE_ENV] = "1"
        os.environ[cpp_bridge.CPP_BRIDGE_ROUTES_ENV] = "all"

        self.assertIn(cpp_bridge.FLASH_DECODE_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.MATMUL_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.RMS_NORM_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.SILU_AND_MUL_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.ROPE_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.STORE_KV_CACHE_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.EMBEDDING_ROUTE, cpp_bridge.selected_routes())
        self.assertIn(cpp_bridge.PREFILL_ROUTE, cpp_bridge.selected_routes())

    def test_musa_all_routes_excludes_flash_decode(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "musa"
        os.environ[cpp_bridge.CPP_BRIDGE_ENABLE_ENV] = "1"
        os.environ[cpp_bridge.CPP_BRIDGE_ROUTES_ENV] = "all"

        self.assertEqual(
            cpp_bridge.selected_routes(),
            tuple(sorted(cpp_bridge.MUSA_DEFAULT_ROUTES)),
        )
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.FLASH_DECODE_ROUTE))

    def test_musa_allows_explicit_flash_decode_route(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "musa"
        os.environ[cpp_bridge.CPP_BRIDGE_ENABLE_ENV] = "1"
        os.environ[cpp_bridge.CPP_BRIDGE_ROUTES_ENV] = cpp_bridge.FLASH_DECODE_ROUTE

        self.assertEqual(
            cpp_bridge.selected_routes(),
            (cpp_bridge.FLASH_DECODE_ROUTE,),
        )
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.FLASH_DECODE_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE))

    def test_unknown_route_is_rejected_before_compile(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_ENABLE_ENV] = "1"
        os.environ[cpp_bridge.CPP_BRIDGE_ROUTES_ENV] = "Unknown"

        with mock.patch.object(cpp_bridge, "_compile_bridge") as compile_bridge:
            with self.assertRaisesRegex(cpp_bridge.CppBridgeError, "unsupported"):
                cpp_bridge.module()

        compile_bridge.assert_not_called()

    def test_call_counts(self) -> None:
        cpp_bridge.record_call(cpp_bridge.DECODE_ROUTE)
        cpp_bridge.record_call(cpp_bridge.DECODE_ROUTE)

        self.assertEqual(cpp_bridge.bridge_call_counts()[cpp_bridge.DECODE_ROUTE], 2)

    def test_flash_decode_num_splits_defaults_to_auto(self) -> None:
        self.assertEqual(cpp_bridge.flash_decode_num_splits(), 0)

    def test_flash_decode_num_splits_env(self) -> None:
        os.environ[cpp_bridge.FLASH_DECODE_NUM_SPLITS_ENV] = "4"

        self.assertEqual(cpp_bridge.flash_decode_num_splits(), 4)

    def test_flash_decode_num_splits_rejects_negative_values(self) -> None:
        os.environ[cpp_bridge.FLASH_DECODE_NUM_SPLITS_ENV] = "-1"

        with self.assertRaisesRegex(cpp_bridge.CppBridgeError, "must be >= 0"):
            cpp_bridge.flash_decode_num_splits()

    def test_invalid_bridge_target_is_rejected(self) -> None:
        os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = "bogus"

        with self.assertRaisesRegex(cpp_bridge.CppBridgeError, "must be one of"):
            cpp_bridge.bridge_target()

    def test_cuda_like_bridge_target_aliases_are_normalized(self) -> None:
        cases = {
            "cuda": cpp_bridge.CUDA_TARGET,
            "metax": cpp_bridge.CUDA_TARGET,
            "muxi": cpp_bridge.CUDA_TARGET,
            "musa": cpp_bridge.MUSA_TARGET,
            "moore": cpp_bridge.MUSA_TARGET,
        }

        for raw_target, expected in cases.items():
            with self.subTest(raw_target=raw_target):
                os.environ[cpp_bridge.CPP_BRIDGE_TARGET_ENV] = raw_target
                self.assertEqual(cpp_bridge.bridge_target(), expected)

    def test_musa_build_config_adds_musa_paths_and_omits_cuda_flags(self) -> None:
        package_dir = Path("/usr/local/lib/python3.10/dist-packages/torch_musa")
        musa_root = Path("/usr/local/musa-4.2")
        with mock.patch.dict(
            os.environ,
            {
                cpp_bridge.CPP_BRIDGE_TARGET_ENV: "musa",
                "INFINI_ROOT": "/opt/infini",
            },
            clear=True,
        ):
            with (
                mock.patch.object(
                    cpp_bridge,
                    "_torch_musa_package_dirs",
                    return_value=(package_dir,),
                ),
                mock.patch.object(cpp_bridge, "_musa_roots", return_value=(musa_root,)),
            ):
                config = cpp_bridge._bridge_build_config()

        self.assertEqual(config["target"], "musa")
        self.assertIn(
            "/usr/local/lib/python3.10/dist-packages",
            config["extra_include_paths"],
        )
        self.assertIn(
            "/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible",
            config["extra_include_paths"],
        )
        self.assertNotIn(
            "/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible/include",
            config["extra_include_paths"],
        )
        self.assertIn("/usr/local/musa-4.2/include", config["extra_include_paths"])
        self.assertIn("-DENABLE_MUSA_API", config["extra_cflags"])
        self.assertIn("-DENABLE_MOORE_API", config["extra_cflags"])
        self.assertNotIn("-DENABLE_METAX_API", config["extra_cflags"])
        self.assertNotIn("-DENABLE_FLASH_ATTN", config["extra_cflags"])
        self.assertIn("-lmusa_python", config["extra_ldflags"])
        self.assertIn("-lmusa_kernels", config["extra_ldflags"])
        self.assertIn("-lmusart", config["extra_ldflags"])
        self.assertFalse(
            any("flash_attn_2_cuda" in flag for flag in config["extra_ldflags"])
        )

    def test_cuda_build_config_keeps_metax_flash_flags(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                cpp_bridge.CPP_BRIDGE_TARGET_ENV: "cuda",
                "INFINI_ROOT": "/opt/infini",
                "MACA_PATH": "/opt/maca",
                "FLASH_ATTN_2_CUDA_SO": "/tmp/flash_attn_2_cuda.so",
            },
            clear=True,
        ):
            config = cpp_bridge._bridge_build_config()

        self.assertEqual(config["target"], "cuda")
        self.assertIn("-DENABLE_FLASH_ATTN", config["extra_cflags"])
        self.assertIn("-DENABLE_METAX_API", config["extra_cflags"])
        self.assertNotIn("-DENABLE_MUSA_API", config["extra_cflags"])
        self.assertIn("/tmp/flash_attn_2_cuda.so", config["extra_ldflags"])


if __name__ == "__main__":
    unittest.main()
