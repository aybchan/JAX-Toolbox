import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
import zipfile

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / 'install-windows-cuda.py'
spec = importlib.util.spec_from_file_location('windows_cuda', SCRIPT)
cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda)


class WindowsCudaTests(unittest.TestCase):
    def test_selects_arm64_cuda13_archive(self):
        wanted = {'relative_path': 'cudnn/windows-arm64/package.zip', 'sha256': 'abc'}
        manifest = {'cudnn': {'windows-arm64': {'cuda13': wanted}, 'linux-sbsa': {}}}
        self.assertEqual(cuda.select_package(manifest, 'cudnn'), wanted)

    def test_rejects_linux_only_component(self):
        with self.assertRaisesRegex(ValueError, 'Windows ARM64'):
            cuda.select_package({'nvshmem': {'linux-sbsa': {}}}, 'nvshmem')

    def test_bad_checksum_prevents_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / 'source.zip'
            source.write_bytes(b'not the expected archive')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                cuda.download_verified(source.as_uri(), root / 'download.zip', '0' * 64)
            self.assertFalse((root / 'download.zip').exists())

    def test_verified_archive_preserves_arm64_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / 'source.zip'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('package/bin/arm64/cudart64_13.dll', b'arm64 dll')
                archive.writestr('package/lib/arm64/cudart.lib', b'import lib')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            archive = cuda.download_verified(source.as_uri(), root / 'download.zip', digest)
            cuda.install_archive(archive, root / 'cuda')
            self.assertEqual((root / 'cuda/bin/arm64/cudart64_13.dll').read_bytes(), b'arm64 dll')
            self.assertEqual((root / 'cuda/lib/arm64/cudart.lib').read_bytes(), b'import lib')

    def test_rejects_archive_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / 'source.zip'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('package/../escape.dll', b'bad')
            with self.assertRaises(ValueError):
                cuda.install_archive(source, root / 'cuda')
            self.assertFalse((root / 'escape.dll').exists())


if __name__ == '__main__':
    unittest.main()
