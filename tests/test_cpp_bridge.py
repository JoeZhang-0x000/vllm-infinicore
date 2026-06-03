from __future__ import annotations

import os
import unittest
from unittest import mock

from vllm_infinicore.ops import cpp_bridge


class CppBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ENABLE_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ROUTES_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_DISABLE_ENV, None)
        cpp_bridge.reset_bridge_call_counts()

    def tearDown(self) -> None:
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ENABLE_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_ROUTES_ENV, None)
        os.environ.pop(cpp_bridge.CPP_BRIDGE_DISABLE_ENV, None)
        cpp_bridge.reset_bridge_call_counts()

    def test_decode_enabled_by_default(self) -> None:
        self.assertEqual(cpp_bridge.selected_routes(), (cpp_bridge.DECODE_ROUTE,))
        self.assertTrue(cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE))
        self.assertFalse(cpp_bridge.enabled_for(cpp_bridge.LM_HEAD_ROUTE))

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


if __name__ == "__main__":
    unittest.main()
