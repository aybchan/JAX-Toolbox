"""Portable PE/package checks; native DLL calls are mocked only in import tests."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import zipfile

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / 'verify-windows-nccl.py'
API = {'ncclGetVersion', 'ncclAllReduce', 'ncclAllGather', 'ncclReduceScatter',
       'ncclBroadcast', 'ncclSend', 'ncclRecv', 'ncclGroupStart', 'ncclGroupEnd',
       'ncclCommInitRankConfig'}


def import_library_fixture(machine=0xaa64, library='nccl.dll', symbols=API):
    archive = bytearray(b'!<arch>\n')
    for symbol in sorted(symbols):
        strings = symbol.encode() + b'\0' + library.encode() + b'\0'
        member = struct.pack('<HHHHIIHH', 0, 0xffff, 0, machine, 0, len(strings), 0, 4) + strings
        header = f'{"nccl.dll/":16}{0:<12}{0:<6}{0:<6}{0:<8}{len(member):<10}`\n'
        archive.extend(header.encode() + member)
        if len(member) % 2:
            archive.extend(b'\n')
    return bytes(archive)


def pe_fixture(exports=(), imports=None, machine=0xaa64, forwarded=False):
    """Small PE32+ layout with actual import/export directories and an RVA gap."""
    data = bytearray(8192)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<HHIIIHH', data, 0x84, machine, 1, 0, 0, 0, 240, 0x2022)
    optional = 0x98
    struct.pack_into('<H', data, optional, 0x20b)
    struct.pack_into('<I', data, optional + 60, 0x200)
    struct.pack_into('<I', data, optional + 108, 16)
    section = optional + 240
    data[section:section + 8] = b'.rdata\0\0'
    struct.pack_into('<IIII', data, section + 8, 0x1e00, 0x1000, 0x1e00, 0x200)

    def rva(offset):
        return offset + 0xe00

    cursor = 0x900

    def string(value):
        nonlocal cursor
        offset = cursor
        encoded = value.encode('ascii') + b'\0'
        data[offset:offset + len(encoded)] = encoded
        cursor += len(encoded)
        return rva(offset)

    if exports:
        exports = sorted(exports)
        struct.pack_into('<II', data, optional + 112, rva(0x200), 0x600)
        struct.pack_into('<IIHHIIIIIII', data, 0x200, 0, 0, 0, 0, 0, 1,
                         len(exports), len(exports), rva(0x300), rva(0x400), rva(0x500))
        for i, name in enumerate(exports):
            struct.pack_into('<I', data, 0x300 + 4 * i,
                             rva(0x250) if forwarded else rva(0x1800))
            struct.pack_into('<I', data, 0x400 + 4 * i, string(name))
            struct.pack_into('<H', data, 0x500 + 2 * i, i)
    if imports:
        struct.pack_into('<II', data, optional + 120, rva(0x600), 20 * (len(imports) + 1))
        for i, (library, symbols) in enumerate(imports.items()):
            thunk = 0x1000 + i * 0x100
            struct.pack_into('<IIIII', data, 0x600 + 20 * i,
                             rva(thunk), 0, 0, string(library), rva(thunk))
            for j, symbol in enumerate(symbols):
                hint = cursor
                struct.pack_into('<H', data, cursor, 0)
                cursor += 2
                string(symbol)
                struct.pack_into('<Q', data, thunk + j * 8, rva(hint))
    return bytes(data)


class NcclVerificationTests(unittest.TestCase):
    def setUp(self):
        if not SCRIPT.exists():
            self.fail('NCCL verifier is not implemented')
        spec = importlib.util.spec_from_file_location('windows_nccl', SCRIPT)
        self.nccl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.nccl)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.sdk = self.root / 'sdk'
        for directory in ('bin', 'lib', 'include'):
            (self.sdk / directory).mkdir(parents=True)
        (self.sdk / 'bin/nccl.dll').write_bytes(pe_fixture(API))
        (self.sdk / 'lib/nccl.lib').write_bytes(import_library_fixture())
        (self.sdk / 'include/nccl.h').write_text(
            '#define NCCL_MAJOR 2\n#define NCCL_MINOR 32\n#define NCCL_PATCH 6\n')
        (self.sdk / 'LICENSE.txt').write_text('Apache License fixture\n')
        (self.sdk / 'include/nccl_device.h').write_text('// device API fixture\n')
        (self.sdk / 'include/nccl_device').mkdir()
        (self.sdk / 'include/nccl_device/gin_win_stub.h').write_text('// Windows GIN declaration fixture\n')
        (self.sdk / 'windows_cuda_versions.json').write_text('{"cuda_nccl":"2.32.6"}')
        (self.sdk / 'build-info.json').write_text(json.dumps({
            'source_commit': 'ad2d192c1f279aefcaf340f67e07880b65a1116b',
            'version': '2.32.6',
            'cuda_architectures': '75-real;80-real;90-real;100-real;110-real;120',
            'dll_sha256': hashlib.sha256((self.sdk / 'bin/nccl.dll').read_bytes()).hexdigest(),
            'import_library_sha256': hashlib.sha256((self.sdk / 'lib/nccl.lib').read_bytes()).hexdigest(),
        }))

    def wheels(self, *, imports=None, bundled=None):
        directory = self.root / 'wheels'
        directory.mkdir(exist_ok=True)
        with zipfile.ZipFile(directory / 'jax_cuda13_plugin-0.0.0-cp314-cp314-win_arm64.whl', 'w') as wheel:
            wheel.writestr('jax_cuda13_plugin/nccl.dll',
                          bundled if bundled is not None else (self.sdk / 'bin/nccl.dll').read_bytes())
            wheel.writestr('jax_cuda13_plugin/LICENSE.nccl.txt', (self.sdk / 'LICENSE.txt').read_bytes())
        with zipfile.ZipFile(directory / 'jax_cuda13_pjrt-0.0.0-py3-none-win_arm64.whl', 'w') as wheel:
            wheel.writestr('jax_plugins/xla_cuda13/xla_cuda_plugin.so',
                          pe_fixture({'GetPjrtApi'}, imports=imports if imports is not None else
                                     {'nccl.dll': sorted(API),
                                      'cudnn64_9.dll': ['cudnnGetProperty', 'cudnnGetVersion']}))
        return directory

    def test_arm64_sdk_and_real_import_export_tables(self):
        sdk = self.nccl.inspect_sdk(self.sdk, 23206)
        self.assertEqual(sdk['version'], 23206)
        report = self.nccl.inspect_wheels(self.wheels(), self.sdk, 23206)
        self.assertEqual(set(report['pjrt_nccl_imports']), API)
        self.assertEqual(report['pjrt_cudnn_imports'], ['cudnnGetProperty', 'cudnnGetVersion'])
        self.assertEqual(report['nccl_sha256'], sdk['nccl_sha256'])

    def test_missing_pjrt_cudnn_property_import_rejected(self):
        # A Windows weak reference omitted this import in the released wheel,
        # so cuDNN initialization crashed even though cudnnGetVersion resolved.
        for cudnn_imports in ({}, {'cudnn64_9.dll': ['cudnnGetVersion']},
                              {'cudnn64_8.dll': ['cudnnGetProperty']}):
            with self.subTest(cudnn_imports=cudnn_imports):
                imports = {'nccl.dll': sorted(API), **cudnn_imports}
                with self.assertRaisesRegex(ValueError, 'cudnnGetProperty'):
                    self.nccl.inspect_wheels(self.wheels(imports=imports), self.sdk, 23206)

    def test_wrong_architecture_rejected(self):
        (self.sdk / 'bin/nccl.dll').write_bytes(pe_fixture(API, machine=0x8664))
        with self.assertRaisesRegex(ValueError, 'ARM64'):
            self.nccl.inspect_sdk(self.sdk, 23206)

    def test_missing_or_forwarded_collective_export_rejected(self):
        for data in (pe_fixture(API - {'ncclAllReduce'}), pe_fixture(API, forwarded=True)):
            with self.subTest(forwarded=data == pe_fixture(API, forwarded=True)):
                (self.sdk / 'bin/nccl.dll').write_bytes(data)
                with self.assertRaises(ValueError):
                    self.nccl.inspect_sdk(self.sdk, 23206)

    def test_missing_sdk_file_and_wrong_header_version_rejected(self):
        library = self.sdk / 'lib/nccl.lib'
        library.unlink()
        with self.assertRaisesRegex(ValueError, 'nccl.lib'):
            self.nccl.inspect_sdk(self.sdk, 23206)
        library.write_bytes(import_library_fixture())
        (self.sdk / 'include/nccl.h').write_text(
            '#define NCCL_MAJOR 2\n#define NCCL_MINOR 31\n#define NCCL_PATCH 6\n')
        with self.assertRaisesRegex(ValueError, 'version'):
            self.nccl.inspect_sdk(self.sdk, 23206)

    def test_presence_without_pjrt_collective_linkage_rejected(self):
        for imports in ({'kernel32.dll': ['GetLastError']}, {'nccl.dll': ['ncclGetVersion']}):
            with self.subTest(imports=imports):
                with self.assertRaisesRegex(ValueError, 'ncclAllReduce|nccl.dll'):
                    self.nccl.inspect_wheels(self.wheels(imports=imports), self.sdk, 23206)

    def test_unresolved_import_and_stale_bundle_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ncclMissingApi'):
            self.nccl.inspect_wheels(self.wheels(imports={'nccl.dll': ['ncclAllReduce', 'ncclMissingApi']}),
                                     self.sdk, 23206)
        changed = bytearray((self.sdk / 'bin/nccl.dll').read_bytes())
        changed[-1] = 1
        with self.assertRaisesRegex(ValueError, 'SHA256|hash|differs'):
            self.nccl.inspect_wheels(self.wheels(bundled=changed), self.sdk, 23206)

    def test_truncated_and_unmapped_pe_rejected(self):
        for data in (b'MZ', pe_fixture(API)[:0x480]):
            (self.sdk / 'bin/nccl.dll').write_bytes(data)
            with self.assertRaises(ValueError):
                self.nccl.inspect_sdk(self.sdk, 23206)

    def test_native_query_cannot_silently_pass_on_other_platforms(self):
        import os
        if os.name == 'nt':
            self.skipTest('Portable fixture must never be executed as a Windows DLL')
        with self.assertRaisesRegex(ValueError, 'Windows'):
            self.nccl.query_version(self.sdk / 'bin/nccl.dll', 23206)

    def test_stale_source_or_hash_in_sdk_provenance_rejected(self):
        path = self.sdk / 'build-info.json'
        original = json.loads(path.read_text())
        for field, value in (('source_commit', '0' * 40), ('dll_sha256', '0' * 64),
                             ('import_library_sha256', '0' * 64), ('version', '2.31.6'),
                             ('cuda_architectures', '120')):
            with self.subTest(field=field):
                path.write_text(json.dumps(dict(original, **{field: value})))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    self.nccl.inspect_sdk(self.sdk, 23206)

    def test_wrong_import_library_architecture_dll_or_api_rejected(self):
        for data in (import_library_fixture(machine=0x8664),
                     import_library_fixture(library='other.dll'),
                     import_library_fixture(symbols={'ncclGetVersion'}), b'!<arch>\n'):
            with self.subTest(data=data[:40]):
                (self.sdk / 'lib/nccl.lib').write_bytes(data)
                provenance = json.loads((self.sdk / 'build-info.json').read_text())
                provenance['import_library_sha256'] = hashlib.sha256(data).hexdigest()
                (self.sdk / 'build-info.json').write_text(json.dumps(provenance))
                with self.assertRaisesRegex(ValueError, 'ARM64|import|nccl.dll'):
                    self.nccl.inspect_sdk(self.sdk, 23206)

    def test_missing_device_headers_rejected(self):
        (self.sdk / 'include/nccl_device/gin_win_stub.h').unlink()
        with self.assertRaisesRegex(ValueError, 'gin_win_stub.h'):
            self.nccl.inspect_sdk(self.sdk, 23206)

    @contextmanager
    def installed_package(self, initializer):
        """Real isolated package imports; the DLL load is a test-only stand-in."""
        site = self.root / 'site-packages'
        package = site / 'jax_cuda13_plugin'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text(initializer)
        (package / 'nccl.dll').write_bytes((self.sdk / 'bin/nccl.dll').read_bytes())
        support = site / 'jaxlib'
        support.mkdir()
        (support / '__init__.py').write_text('')
        (support / 'plugin_support.py').write_text(
            'import _nccl_installed_fixture as state\n'
            'def initialize_windows_cuda_dll_search_path():\n'
            '    state.registered = True\n')
        state = ModuleType('_nccl_installed_fixture')
        state.registered = False
        names = ('jax_cuda13_plugin', 'jaxlib', '_nccl_installed_fixture')
        saved = {key: value for key, value in sys.modules.items()
                 if any(key == name or key.startswith(name + '.') for name in names)}
        for key in saved:
            del sys.modules[key]
        sys.modules['_nccl_installed_fixture'] = state
        sys.path.insert(0, str(site))
        importlib.invalidate_caches()

        def native_query(path, version, *, by_name=False):
            self.assertTrue(by_name)
            self.assertEqual(path, package / 'nccl.dll')
            if not state.registered:
                raise OSError('Fixture package did not register its DLL directory')
            return {'loaded_library': str(path.resolve()), 'runtime_version': version}

        environment = {'NCCL_PATH': str(self.sdk), 'LOCAL_NCCL_PATH': str(self.sdk),
                       'PATH': os.pathsep.join((str(self.sdk / 'bin'), str(self.root / 'other')))}
        try:
            with mock.patch.dict(os.environ, environment, clear=True), \
                 mock.patch.object(self.nccl, 'os', SimpleNamespace(
                     name='nt', environ=os.environ, pathsep=os.pathsep)), \
                 mock.patch.object(self.nccl, 'query_version', side_effect=native_query) as query:
                yield package, query
        finally:
            sys.path.remove(str(site))
            for key in list(sys.modules):
                if any(key == name or key.startswith(name + '.') for name in names):
                    del sys.modules[key]
            sys.modules.update(saved)
            importlib.invalidate_caches()

    def test_installed_package_initializer_registers_before_native_load(self):
        with self.installed_package(
                'from jaxlib.plugin_support import initialize_windows_cuda_dll_search_path\n'
                'initialize_windows_cuda_dll_search_path()\n') as (package, query):
            result = self.nccl.inspect_installed(self.sdk, 23206)
            self.assertTrue('jax_cuda13_plugin' in sys.modules, 'CUDA package initializer was not executed')
            self.assertEqual(result['nccl_sha256'], self.nccl.inspect_sdk(self.sdk, 23206)['nccl_sha256'])
            query.assert_called_once_with(package / 'nccl.dll', 23206, by_name=True)
            self.assertNotIn('NCCL_PATH', os.environ)
            self.assertNotIn('LOCAL_NCCL_PATH', os.environ)
            self.assertEqual(os.environ['PATH'], str(self.root / 'other'))

    def test_installed_empty_initializer_cannot_pass_via_direct_helper_call(self):
        with self.installed_package(''):
            with self.assertRaisesRegex(OSError, 'did not register'):
                self.nccl.inspect_installed(self.sdk, 23206)

    def test_installed_broken_initializer_is_not_bypassed(self):
        with self.installed_package("raise RuntimeError('broken installed initializer')\n") as (_, query):
            with self.assertRaisesRegex(RuntimeError, 'broken installed initializer'):
                self.nccl.inspect_installed(self.sdk, 23206)
            query.assert_not_called()

    def test_installed_hash_must_match_verified_sdk_before_loading(self):
        with self.installed_package('') as (package, query):
            dll = package / 'nccl.dll'
            dll.write_bytes(dll.read_bytes() + b'different installed payload')
            with self.assertRaisesRegex(ValueError, 'SHA256.*SDK'):
                self.nccl.inspect_installed(self.sdk, 23206)
            query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
