from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from vllm_infinicore.ops import infinicore_backend


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

if __name__ == "__main__":
    unittest.main()
