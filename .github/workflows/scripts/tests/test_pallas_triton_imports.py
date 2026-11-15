"""Exercise the patched public module without requiring host JAX native binaries.

Set JAX_TEST_SOURCE to the pinned JAX checkout after applying
jax-windows-arm64.patch and jax-windows-arm64-cuda.patch, then run this file
or include it in unittest discovery. No source is downloaded or installed.

Only host-independent providers are stubbed, except CompilerParams which is
loaded from the exact pinned source. The finder simulates the unavailable native
extension at its two eager dependency entry points, or supplies identity-stable
native APIs to verify deferred exports. This does not execute Pallas kernels.
"""
import ast
import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

PUBLIC = 'jax.experimental.pallas.triton'
NATIVE = {'jax._src.pallas.triton.pallas_call_registration', 'jax._src.pallas.triton.primitives'}
EXTENSION = 'jaxlib.mlir._mlir_libs._triton_ext'
# Public API contract at pinned JAX 17c608ec2206772f573ae986fe8da097a34b0c39.
# Keep this independent of the implementation so dropped exports fail a test.
EXPORTS = {
    'dot': ('jax._src.pallas.primitives', 'dot'),
    'CompilerParams': ('jax._src.pallas.triton.core', 'CompilerParams'),
    'CUSTOM_CALL_TARGET_NAME': ('jax._src.pallas.triton.pallas_call_registration', 'CUSTOM_CALL_TARGET_NAME'),
    'approx_tanh': ('jax._src.pallas.triton.primitives', 'approx_tanh'),
    'atomic_add': ('jax._src.pallas.triton.primitives', 'atomic_add'),
    'atomic_and': ('jax._src.pallas.triton.primitives', 'atomic_and'),
    'atomic_cas': ('jax._src.pallas.triton.primitives', 'atomic_cas'),
    'atomic_max': ('jax._src.pallas.triton.primitives', 'atomic_max'),
    'atomic_min': ('jax._src.pallas.triton.primitives', 'atomic_min'),
    'atomic_or': ('jax._src.pallas.triton.primitives', 'atomic_or'),
    'atomic_xchg': ('jax._src.pallas.triton.primitives', 'atomic_xchg'),
    'atomic_xor': ('jax._src.pallas.triton.primitives', 'atomic_xor'),
    'debug_barrier': ('jax._src.pallas.triton.primitives', 'debug_barrier'),
    'elementwise_inline_asm': ('jax._src.pallas.triton.primitives', 'elementwise_inline_asm'),
    'load': ('jax._src.pallas.triton.primitives', 'load'),
    'max_contiguous': ('jax._src.pallas.triton.primitives', 'max_contiguous'),
    'store': ('jax._src.pallas.triton.primitives', 'store'),
    'GpuVersion': ('jax._src.pallas.triton.gpu_info', 'GpuVersion'),
    'get_gpu_info': ('jax._src.pallas.triton.gpu_info', 'get_gpu_info'),
    'get_gpu_info_from_version': ('jax._src.pallas.triton.gpu_info', 'get_gpu_info_from_version'),
    'is_gpu_device': ('jax._src.pallas.triton.gpu_info', 'is_gpu_device'),
    'GpuInfo': ('jax._src.pallas.triton.gpu_info', 'GpuInfo'),
}


class NativeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, available):
        self.available = available
        self.attempts = []
        self.values = {name: object() for name, (module, _) in EXPORTS.items() if module in NATIVE}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in NATIVE:
            return None
        self.attempts.append(fullname)
        if not self.available:
            raise ModuleNotFoundError(f"No module named '{EXTENSION}'", name=EXTENSION)
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        for name, (provider, exported) in EXPORTS.items():
            if provider == module.__name__:
                setattr(module, exported, self.values[name])


class LazyTritonImportsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = os.environ.get('JAX_TEST_SOURCE')
        if not source_path:
            raise unittest.SkipTest('Set JAX_TEST_SOURCE to the patched pinned JAX source root')
        cls.source_root = Path(source_path).resolve()
        cls.public_source = cls.source_root / 'jax/experimental/pallas/triton.py'
        required = (cls.public_source,
                    cls.source_root / 'jax/_src/pallas/triton/core.py',
                    cls.source_root / 'jax/_src/pallas/pallas_call.py')
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f'Required JAX source file does not exist: {path}')

    def load_public(self, native_available=False):
        self.enterContext(mock.patch.dict(sys.modules))
        for name in list(sys.modules):
            if name == 'jax' or name.startswith('jax.'):
                del sys.modules[name]
        for name in ('jax', 'jax._src', 'jax._src.pallas', 'jax._src.pallas.triton',
                     'jax.experimental', 'jax.experimental.pallas'):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        self.eager_values = {}
        for name, (provider, exported) in EXPORTS.items():
            if provider in NATIVE or exported == 'CompilerParams':
                continue
            module = sys.modules.setdefault(provider, types.ModuleType(provider))
            value = object()
            setattr(module, exported, value)
            self.eager_values[name] = value
        core_path = self.source_root / 'jax/_src/pallas/triton/core.py'
        core_spec = importlib.util.spec_from_file_location('jax._src.pallas.triton.core', core_path)
        core = importlib.util.module_from_spec(core_spec)
        sys.modules[core_spec.name] = core
        core_spec.loader.exec_module(core)
        self.finder = NativeFinder(native_available)
        self.enterContext(mock.patch.object(sys, 'meta_path', [self.finder, *sys.meta_path]))
        spec = importlib.util.spec_from_file_location(PUBLIC, self.public_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[PUBLIC] = module
        spec.loader.exec_module(module)
        return module

    def test_import_and_real_compiler_params_without_native_extension(self):
        module = self.load_public()
        self.assertEqual(module.CompilerParams().num_warps, None)
        self.assertEqual(module.CompilerParams(num_warps=8).num_warps, 8)
        imported = {}
        exec('from jax.experimental.pallas.triton import CompilerParams, get_gpu_info', imported)
        self.assertIs(imported['CompilerParams'], module.CompilerParams)
        self.assertIs(imported['get_gpu_info'], self.eager_values['get_gpu_info'])
        self.assertEqual(self.finder.attempts, [])

    def test_all_export_names_and_dir_are_preserved_without_loading_native(self):
        module = self.load_public()
        self.assertEqual(set(module.__all__), set(EXPORTS))
        self.assertEqual(len(module.__all__), len(EXPORTS))
        self.assertTrue(set(EXPORTS).issubset(dir(module)))
        self.assertEqual(self.finder.attempts, [])

    def test_missing_native_extension_propagates_for_each_native_provider(self):
        module = self.load_public()
        for name in ('atomic_add', 'CUSTOM_CALL_TARGET_NAME'):
            with self.subTest(name=name):
                with self.assertRaises(ModuleNotFoundError) as error:
                    getattr(module, name)
                self.assertEqual(error.exception.name, EXTENSION)
                self.assertNotIn(name, vars(module))
        self.assertEqual(set(self.finder.attempts), NATIVE)

    def test_native_exports_keep_identity_cache_and_star_import(self):
        module = self.load_public(native_available=True)
        self.assertEqual(self.finder.attempts, [])
        namespace = {}
        exec('from jax.experimental.pallas.triton import *', namespace)
        self.assertEqual(set(namespace) - {'__builtins__'}, set(EXPORTS))
        for name, value in self.finder.values.items():
            self.assertIs(namespace[name], value)
            self.assertIs(getattr(module, name), value)
            self.assertIs(vars(module)[name], value)
        self.assertEqual(set(self.finder.attempts), NATIVE)
        self.assertEqual(len(self.finder.attempts), 2)

    def test_existing_noninterpret_dispatch_still_imports_registration(self):
        module = self.load_public(native_available=True)
        self.assertEqual(self.finder.attempts, [])
        # Execute the unchanged pinned dispatch function with a fake MLIR boundary.
        # This verifies deferred registration, not native GPU compilation.
        dispatch_path = self.source_root / 'jax/_src/pallas/pallas_call.py'
        node = next(node for node in ast.parse(dispatch_path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_pallas_call_lowering')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        dispatch_ast = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        lowered = object()
        def get_rule(params_type, platform):
            self.assertEqual(self.finder.attempts, ['jax._src.pallas.triton.pallas_call_registration'])
            self.assertIs(params_type, module.CompilerParams)
            self.assertEqual(platform, 'gpu')
            return lambda ctx, *args, **kwargs: lowered
        def lower_per_platform(ctx, name, rules, default, effects, *args, **kwargs):
            return rules['cuda'](ctx, *args, **kwargs)
        namespace = {
            'mlir': types.SimpleNamespace(lower_per_platform=lower_per_platform),
            'pallas_core': types.SimpleNamespace(get_lowering_rule=get_rule),
            'effects': types.SimpleNamespace(no_effects=()),
        }
        exec(compile(dispatch_ast, str(dispatch_path), 'exec'), namespace)
        for params in (None, module.CompilerParams()):
            with self.subTest(compiler_params=params):
                result = namespace['_pallas_call_lowering'](
                    object(), interpret=False, jaxpr=types.SimpleNamespace(constvars=()),
                    compiler_params=params)
                self.assertIs(result, lowered)

    def test_unknown_name_does_not_attempt_native_import(self):
        module = self.load_public()
        with self.assertRaisesRegex(AttributeError, 'does_not_exist'):
            module.does_not_exist
        self.assertEqual(self.finder.attempts, [])


if __name__ == '__main__':
    unittest.main()
