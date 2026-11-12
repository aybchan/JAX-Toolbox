"""Exercise the exact Python helper functions shipped in the JAX patch.

Windows directory handles are modeled, while package discovery/import order and
filesystem copying use real Python behavior. Native DLL loading is a CI check.
"""
import gc
import importlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
import weakref

PATCH = Path(__file__).resolve().parents[3] / 'patches/jax-windows-arm64-cuda.patch'


def patched_function(filename, name):
    lines = PATCH.read_text().splitlines()
    start = lines.index('+++ b/' + filename) + 1
    body = []
    for line in lines[start:]:
        if line.startswith(('diff --git ', '--- ', '+++ ')):
            break
        if line.startswith((' ', '+')):
            body.append(line[1:])
    start = next((i for i, line in enumerate(body) if line.startswith('def ' + name + '(')), None)
    if start is None:
        raise AssertionError(f'{name} is not present in the JAX patch')
    end = start + 1
    while end < len(body) and (not body[end] or body[end].startswith(' ')):
        end += 1
    return '\n'.join(body[start:end]) + '\n'


class Handle:
    pass


class NcclPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / 'jax_cuda13_plugin'
        self.old_path = sys.path[:]
        sys.path.insert(0, str(self.root))
        self.old_modules = {name: value for name, value in sys.modules.items()
                            if name == '_nccl_test_support' or name.startswith('jax_cuda13_plugin')}
        for name in self.old_modules:
            del sys.modules[name]
        self.addCleanup(self.restore_imports)
        self.calls, self.handles = [], []
        self.support = ModuleType('_nccl_test_support')
        self.os = SimpleNamespace(name='nt', environ={}, add_dll_directory=self.register)
        self.support.__dict__.update(os=self.os, pathlib=importlib.import_module('pathlib'),
                                     importlib=importlib, _windows_cuda_dll_directories={})
        exec(patched_function('jaxlib/plugin_support.py', 'initialize_windows_cuda_dll_search_path'),
             self.support.__dict__)
        sys.modules['_nccl_test_support'] = self.support

    def restore_imports(self):
        for name in list(sys.modules):
            if name == '_nccl_test_support' or name.startswith('jax_cuda13_plugin'):
                del sys.modules[name]
        sys.modules.update(self.old_modules)
        sys.path[:] = self.old_path
        importlib.invalidate_caches()

    def register(self, directory):
        self.calls.append(directory)
        handle = Handle()
        self.handles.append(weakref.ref(handle))
        return handle

    def create_package(self, initializer):
        self.package.mkdir()
        (self.package / '__init__.py').write_text(initializer)
        (self.package / 'nccl.dll').write_bytes(b'fixture, never executed')
        importlib.invalidate_caches()

    def test_pjrt_first_discovers_bundle_without_executing_package(self):
        self.create_package("raise RuntimeError('discovery must not execute package')\n")
        self.support.initialize_windows_cuda_dll_search_path()
        self.assertEqual(self.calls, [str(self.package.resolve())])
        self.assertNotIn('jax_cuda13_plugin', sys.modules)

    def test_package_initialization_registers_before_extension_import(self):
        self.create_package('from _nccl_test_support import initialize_windows_cuda_dll_search_path\n'
                            'initialize_windows_cuda_dll_search_path()\nfrom . import _versions\n')
        (self.package / '_versions.py').write_text(
            'from pathlib import Path\nfrom _nccl_test_support import _windows_cuda_dll_directories\n'
            "if Path(__file__).resolve().parent not in _windows_cuda_dll_directories:\n"
            "    raise ImportError('bundled NCCL is not discoverable before the extension')\n")
        importlib.import_module('jax_cuda13_plugin')
        self.support.initialize_windows_cuda_dll_search_path()
        self.assertEqual(self.calls, [str(self.package.resolve())])
        gc.collect()
        self.assertTrue(all(handle() is not None for handle in self.handles))

    def test_external_sdk_fallback_and_non_windows_behavior(self):
        sdk = self.root / 'sdk/bin'
        sdk.mkdir(parents=True)
        self.os.environ['NCCL_PATH'] = str(sdk.parent)
        self.support.initialize_windows_cuda_dll_search_path()
        self.assertEqual(self.calls, [str(sdk.resolve())])
        self.os.name = 'posix'
        self.support.initialize_windows_cuda_dll_search_path()
        self.assertEqual(len(self.calls), 1)

    def test_bundled_nccl_avoids_ambiguous_external_dll_search(self):
        self.create_package('')
        sdk = self.root / 'sdk/bin'
        sdk.mkdir(parents=True)
        self.os.environ['NCCL_PATH'] = str(sdk.parent)
        self.support.initialize_windows_cuda_dll_search_path()
        self.assertEqual(self.calls, [str(self.package.resolve())])

    def test_wheel_assembly_uses_declared_inputs_and_requires_both_files(self):
        namespace = {'build_utils': SimpleNamespace(is_windows=lambda: True)}
        exec(patched_function('jaxlib/tools/build_gpu_kernels_wheel.py', '_copy_windows_nccl_runtime'), namespace)
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'nccl.dll').write_bytes(b'actual declared DLL bytes')
        (runtime / 'LICENSE.nccl.txt').write_text('actual declared license')
        self.package.mkdir()

        def copy_files(src_files, dst_dir):
            for name in src_files:
                shutil.copyfile(self.root / name, dst_dir / Path(name).name)

        namespace['_copy_windows_nccl_runtime'](self.package, copy_files, ['declared inputs'])
        self.assertEqual((self.package / 'nccl.dll').read_bytes(), (runtime / 'nccl.dll').read_bytes())
        self.assertEqual((self.package / 'LICENSE.nccl.txt').read_bytes(), (runtime / 'LICENSE.nccl.txt').read_bytes())
        (runtime / 'LICENSE.nccl.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            namespace['_copy_windows_nccl_runtime'](self.package, copy_files, ['declared inputs'])
        namespace['build_utils'].is_windows = lambda: False
        namespace['_copy_windows_nccl_runtime'](self.package, copy_files, ['declared inputs'])


if __name__ == '__main__':
    unittest.main()
