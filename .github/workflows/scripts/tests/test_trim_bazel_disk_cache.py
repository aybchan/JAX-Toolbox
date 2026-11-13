import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("trim_cache", Path(__file__).resolve().parents[1] / "trim-bazel-disk-cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrimCacheTest(unittest.TestCase):
    def test_caps_total_bytes_and_keeps_newest_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            for name, size, mtime in [("cas/old", 7, 1), ("ac/new", 3, 2), ("cas/new", 5, 3)]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * size)
                os.utime(path, (mtime, mtime))
            self.assertEqual(module.trim_cache(root, 8), 8)
            self.assertFalse((root / "cas/old").exists())
            self.assertEqual(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()), 8)
            self.assertEqual((root / "cas/new").read_bytes(), b"x" * 5)

    def test_drops_partial_writes_even_when_under_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tmp").mkdir()
            (root / "tmp/incomplete").write_bytes(b"pending")
            (root / "cas").mkdir()
            (root / "cas/valid").write_bytes(b"complete")
            self.assertEqual(module.trim_cache(root, 1024), 8)
            self.assertFalse((root / "tmp/incomplete").exists())
            self.assertTrue((root / "cas").is_dir())

    def test_rejects_symlink_without_deleting_outside_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.write_text("keep")
            (root / "escape").symlink_to(outside)
            with self.assertRaises(ValueError):
                module.trim_cache(root, 0)
            self.assertEqual(outside.read_text(), "keep")

    def test_zero_limit_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            self.assertEqual(module.trim_cache(root, 0), 0)
            root.mkdir()
            (root / "entry").write_bytes(b"cached")
            self.assertEqual(module.trim_cache(root, 0), 0)
            self.assertTrue(root.is_dir())


if __name__ == "__main__":
    unittest.main()
