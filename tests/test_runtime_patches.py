from __future__ import annotations

import unittest

from vllm_infinicore import runtime_patches


class RuntimePatchTests(unittest.TestCase):
    def test_dummy_run_source_uses_real_request_count(self) -> None:
        source = """
            @torch.inference_mode()
            def _dummy_run(self):
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                )
        """

        patched = runtime_patches._dummy_run_real_reqs_source(source)

        self.assertIsNotNone(patched)
        assert patched is not None
        self.assertIn("num_reqs=num_reqs,", patched)
        self.assertNotIn("num_reqs=num_reqs_padded,", patched)

    def test_dummy_run_source_returns_none_when_pattern_missing(self) -> None:
        source = """
            def _dummy_run(self):
                return self._build_attention_metadata(num_reqs=num_reqs)
        """

        self.assertIsNone(runtime_patches._dummy_run_real_reqs_source(source))


if __name__ == "__main__":
    unittest.main()
