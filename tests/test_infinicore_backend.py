from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS dev env may not ship torch.
    torch = None

if torch is not None:
    from vllm_infinicore.ops import infinicore_backend
else:
    infinicore_backend = None


@unittest.skipIf(torch is None, "torch is not installed")
class InfiniCoreBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        infinicore_backend.clear_tensor_wrapper_cache()
        infinicore_backend.clear_stream_cache()
        infinicore_backend.reset_backend_call_counts()

    def tearDown(self) -> None:
        infinicore_backend.clear_tensor_wrapper_cache()
        infinicore_backend.clear_stream_cache()
        infinicore_backend.reset_backend_call_counts()

    def test_tensor_wrapper_cache_reuses_matching_views(self) -> None:
        base = torch.arange(32, dtype=torch.float32).view(4, 8)
        view = base[:, ::2]
        wrapped = SimpleNamespace()

        with mock.patch.object(
            infinicore_backend,
            "_as_infini_strided",
            return_value=wrapped,
        ) as create:
            first = infinicore_backend._as_infini_strided_cached(view)
            second = infinicore_backend._as_infini_strided_cached(view)

        self.assertIs(first, wrapped)
        self.assertIs(second, wrapped)
        create.assert_called_once_with(view)

    def test_tensor_wrapper_cache_distinguishes_shape_and_stride(self) -> None:
        base = torch.arange(32, dtype=torch.float32).view(4, 8)
        row_major = base[:, :4]
        strided = base[:, ::2]

        with mock.patch.object(
            infinicore_backend,
            "_as_infini_strided",
            side_effect=[SimpleNamespace(name="row-major"), SimpleNamespace(name="strided")],
        ) as create:
            first = infinicore_backend._as_infini_strided_cached(row_major)
            second = infinicore_backend._as_infini_strided_cached(strided)

        self.assertEqual(first.name, "row-major")
        self.assertEqual(second.name, "strided")
        self.assertEqual(create.call_count, 2)

    def test_ray_store_kv_cache_is_enabled_by_default(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = True
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_RAY_BACKEND": "1",
            },
            clear=True,
        ):
            self.assertTrue(infinicore_backend.store_kv_cache_backend_enabled(tensor))

    def test_ray_store_kv_cache_can_be_explicitly_disabled(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = True
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_RAY_BACKEND": "1",
                "VLLM_INFINICORE_DISABLE_RAY_STORE_KV_CACHE": "1",
            },
            clear=True,
        ):
            self.assertFalse(infinicore_backend.store_kv_cache_backend_enabled(tensor))

    def test_ray_backend_enabled_uses_ray_marker(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_INFINICORE_RAY_BACKEND": "1"}, clear=True):
            self.assertTrue(infinicore_backend.ray_backend_enabled())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(infinicore_backend.ray_backend_enabled())

if __name__ == "__main__":
    unittest.main()
