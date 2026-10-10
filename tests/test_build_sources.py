"""Reproducible patch preparation without vendor SDKs or upstream downloads."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.infinicore_build.ascend import prepare_ascend_source
from scripts.infinicore_build.metax import prepare_ops_source

REVISION = "0" * 40


class PatchPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, platform):
        source = self.root / platform / "checkout"
        original = source / "submodules/InfiniOps" if platform == "metax" else source
        for directory in ("src", "include", "tests"):
            (original / directory).mkdir(parents=True)
        (original / "src/value.txt").write_text("before\n")
        (original / "tests/keep.txt").write_text("test source\n")
        patch_dir = self.root / platform / "patches" / platform
        patch_dir.mkdir(parents=True)
        patch = patch_dir / "change.patch"
        patch.write_text("--- a/src/value.txt\n+++ b/src/value.txt\n@@ -1 +1 @@\n-before\n+after\n")
        manifest = {
            "base_revision": REVISION,
            "patches": [
                {
                    "name": "change",
                    "file": patch.name,
                    "url": patch.as_uri(),
                    "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                }
            ],
            "files": {"src/value.txt": hashlib.sha256(b"after\n").hexdigest()},
        }
        (patch_dir / "manifest.json").write_text(json.dumps(manifest))
        build = self.root / platform / "build"
        prepare = prepare_ops_source if platform == "metax" else prepare_ascend_source
        return prepare, source, original, build, patch_dir, manifest

    def test_local_patches_preserve_checkout_and_reuse_verified_copy(self):
        for platform in ("metax", "ascend"):
            with self.subTest(platform=platform):
                prepare, source, original, build, patch_dir, manifest = self.fixture(platform)
                copied, applied = prepare(source, build, patch_dir, REVISION, "local")
                self.assertEqual(applied, manifest)
                self.assertEqual((original / "src/value.txt").read_text(), "before\n")
                self.assertEqual((copied / "src/value.txt").read_text(), "after\n")
                self.assertTrue((copied / "include").is_dir())
                self.assertEqual((copied / "tests").exists(), platform == "metax")
                self.assertEqual(
                    prepare(source, build, patch_dir, REVISION, "local"), (copied, manifest)
                )

    def test_none_uses_original_without_loading_manifest(self):
        for platform in ("metax", "ascend"):
            with self.subTest(platform=platform):
                prepare, source, original, build, _, _ = self.fixture(platform)
                self.assertEqual(
                    prepare(source, build, self.root / "missing", REVISION, "none"),
                    (original, None),
                )
                self.assertFalse(build.exists())

    def test_wrong_revision_and_patch_bytes_are_rejected_before_copy(self):
        for platform in ("metax", "ascend"):
            with self.subTest(platform=platform):
                prepare, source, original, build, patch_dir, _ = self.fixture(platform)
                with self.assertRaisesRegex(RuntimeError, "locked .* revision"):
                    prepare(source, build, patch_dir, "1" * 40, "local")
                (patch_dir / "change.patch").write_text("corrupt patch\n")
                with self.assertRaisesRegex(RuntimeError, "Patch checksum mismatch"):
                    prepare(source, build, patch_dir, REVISION, "local")
                self.assertFalse((build / "sources").exists())
                self.assertEqual((original / "src/value.txt").read_text(), "before\n")

    def test_cached_sources_reject_mutations_and_changed_patch_set(self):
        for platform in ("metax", "ascend"):
            with self.subTest(platform=platform):
                prepare, source, _, build, patch_dir, manifest = self.fixture(platform)
                copied, _ = prepare(source, build, patch_dir, REVISION, "local")
                (copied / "src/value.txt").write_text("changed\n")
                with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                    prepare(source, build, patch_dir, REVISION, "local")
                (copied / "src/value.txt").write_text("after\n")
                manifest["note"] = "a different patch set"
                (patch_dir / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "fresh build directory"):
                    prepare(source, build, patch_dir, REVISION, "local")

    def test_upstream_download_is_verified_and_cached_offline(self):
        prepare, source, _, build, patch_dir, manifest = self.fixture("metax")
        copied, _ = prepare(source, build, patch_dir, REVISION, "upstream")
        cached = build / "patch-cache/change.patch"
        self.assertEqual(
            hashlib.sha256(cached.read_bytes()).hexdigest(), manifest["patches"][0]["sha256"]
        )
        (patch_dir / "change.patch").unlink()
        self.assertEqual(
            prepare(source, build, patch_dir, REVISION, "upstream"), (copied, manifest)
        )

    def test_invalid_download_is_not_published_to_cache(self):
        prepare, source, original, build, patch_dir, _ = self.fixture("metax")
        (patch_dir / "change.patch").write_text("invalid download\n")
        with self.assertRaisesRegex(RuntimeError, "Downloaded patch checksum mismatch"):
            prepare(source, build, patch_dir, REVISION, "upstream")
        self.assertEqual(list((build / "patch-cache").iterdir()), [])
        self.assertFalse((build / "sources").exists())
        self.assertEqual((original / "src/value.txt").read_text(), "before\n")


if __name__ == "__main__":
    unittest.main()
