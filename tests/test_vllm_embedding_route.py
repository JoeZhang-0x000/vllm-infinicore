from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on torch install
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from vllm_infinicore.ops import vllm_embedding
except Exception as exc:  # pragma: no cover - depends on vLLM install
    vllm_embedding = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class VllmEmbeddingRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            self.skipTest(f"torch unavailable: {TORCH_IMPORT_ERROR}")
        if vllm_embedding is None:
            self.skipTest(f"vLLM embedding route unavailable: {IMPORT_ERROR}")

    def test_tensor_parallel_embedding_uses_native_route(self) -> None:
        class Layer(torch.nn.Module):
            tp_size = 2

            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.ones(4, 2)

        calls: list[str] = []

        def original(_self, layer, input_):
            calls.append("native")
            return torch.nn.functional.embedding(input_, layer.weight)

        with (
            mock.patch.object(vllm_embedding, "_ORIGINAL_EMBEDDING", original),
            mock.patch.object(
                torch.ops.vllm_infinicore,
                "embedding",
                side_effect=AssertionError("custom op should not be called"),
                create=True,
            ),
        ):
            result = vllm_embedding._patched_embedding(
                object(),
                Layer(),
                torch.tensor([0, 1], dtype=torch.long),
            )

        self.assertEqual(calls, ["native"])
        self.assertEqual(tuple(result.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
