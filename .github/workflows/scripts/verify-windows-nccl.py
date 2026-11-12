#!/usr/bin/env python3
"""Verify the real Windows ARM64 NCCL SDK, wheel linkage, and installed DLL.

The SDK and installed modes execute only NCCL's version API, which does not
initialize CUDA. Wheel inspection is portable. None of these checks establishes
GPU collective correctness or performance.
"""
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import zipfile

REQUIRED_EXPORTS = {
    'ncclGetVersion', 'ncclCommInitRankConfig', 'ncclGroupStart', 'ncclGroupEnd',
    'ncclAllReduce', 'ncclAllGather', 'ncclReduceScatter', 'ncclBroadcast',
    'ncclSend', 'ncclRecv',
}
MAX_ENTRIES = 131072
EXPECTED_SOURCE = 'ad2d192c1f279aefcaf340f67e07880b65a1116b'
EXPECTED_ARCHITECTURES = '75-real;80-real;90-real;100-real;110-real;120'


def require(condition, message):
    if not condition:
        raise ValueError(message)


class PE:
    """Bounded PE32+ import/export reader; accepts bytes or a read-only mmap."""
    def __init__(self, data, label):
        self.data, self.label = data, label
        require(len(data) >= 64 and data[:2] == b'MZ', f'{label}: missing PE DOS header')
        header = self.unpack('<I', 0x3c)[0]
        require(header >= 64 and data[header:header + 4] == b'PE\0\0', f'{label}: invalid PE header')
        machine, count = self.unpack('<HH', header + 4)
        require(machine == 0xaa64, f'{label}: expected ARM64 PE, found machine {machine:#x}')
        optional_size = self.unpack('<H', header + 20)[0]
        self.optional = header + 24
        require(optional_size >= 120 and self.optional + optional_size <= len(data),
                f'{label}: truncated optional header')
        require(self.unpack('<H', self.optional)[0] == 0x20b, f'{label}: expected PE32+')
        self.directory_count = self.unpack('<I', self.optional + 108)[0]
        require(self.directory_count <= (optional_size - 112) // 8, f'{label}: invalid directory count')
        self.header_size = self.unpack('<I', self.optional + 60)[0]
        require(1 <= count <= 96, f'{label}: invalid section count')
        self.sections = []
        for index in range(count):
            offset = self.optional + optional_size + index * 40
            virtual_size, address, raw_size, raw = self.unpack('<IIII', offset + 8)
            require(raw + raw_size <= len(data), f'{label}: truncated section')
            self.sections.append((address, virtual_size, raw, raw_size))

    def unpack(self, fmt, offset):
        require(0 <= offset <= len(self.data) - struct.calcsize(fmt), f'{self.label}: truncated PE field')
        return struct.unpack_from(fmt, self.data, offset)

    def locate(self, rva, count=1):
        if 0 <= rva < self.header_size:
            end = min(self.header_size, len(self.data))
            require(rva + count <= end, f'{self.label}: header RVA overflow')
            return rva, end
        for address, virtual_size, raw, raw_size in self.sections:
            if address <= rva < address + max(virtual_size, raw_size):
                relative = rva - address
                require(relative + count <= raw_size, f'{self.label}: unbacked RVA')
                return raw + relative, raw + raw_size
        raise ValueError(f'{self.label}: unmapped RVA {rva:#x}')

    def string(self, rva):
        offset, end = self.locate(rva)
        end = self.data.find(b'\0', offset, min(end, offset + 4096))
        require(end >= 0, f'{self.label}: unterminated PE string')
        return self.data[offset:end].decode('ascii')

    def directory(self, index):
        return self.unpack('<II', self.optional + 112 + index * 8) if index < self.directory_count else (0, 0)

    def exports(self):
        rva, size = self.directory(0)
        if not rva:
            return {}
        require(size >= 40, f'{self.label}: invalid export directory')
        offset, _ = self.locate(rva, 40)
        fields = self.unpack('<IIHHIIIIIII', offset)
        function_count, name_count = fields[6:8]
        require(0 < function_count <= MAX_ENTRIES and name_count <= MAX_ENTRIES,
                f'{self.label}: invalid export count')
        functions, _ = self.locate(fields[8], 4 * function_count)
        names, _ = self.locate(fields[9], 4 * name_count)
        ordinals, _ = self.locate(fields[10], 2 * name_count)
        result = {}
        for index in range(name_count):
            name = self.string(self.unpack('<I', names + 4 * index)[0])
            ordinal = self.unpack('<H', ordinals + 2 * index)[0]
            require(ordinal < function_count and name not in result, f'{self.label}: invalid export ordinal/name')
            address = self.unpack('<I', functions + 4 * ordinal)[0]
            if address:
                self.locate(address)
            result[name] = bool(address and not rva <= address < rva + size)
        return result

    def imports(self):
        rva, size = self.directory(1)
        if not rva:
            return {}
        require(20 <= size <= 20 * MAX_ENTRIES, f'{self.label}: invalid import directory')
        offset, _ = self.locate(rva, size)
        result = {}
        for index in range(size // 20):
            fields = self.unpack('<IIIII', offset + index * 20)
            if not any(fields):
                return result
            library = self.string(fields[3]).lower()
            require(library not in result, f'{self.label}: duplicate DLL import descriptor')
            thunk_rva = fields[0] or fields[4]
            thunk, end = self.locate(thunk_rva, 8)
            symbols = []
            for symbol_index in range(min(MAX_ENTRIES, (end - thunk) // 8)):
                address = self.unpack('<Q', thunk + 8 * symbol_index)[0]
                if not address:
                    break
                symbols.append(f'ordinal:{address & 0xffff}' if address >> 63 else self.string(address + 2))
            else:
                raise ValueError(f'{self.label}: unterminated import thunk')
            result[library] = symbols
        raise ValueError(f'{self.label}: unterminated import directory')


def binary_info(path):
    require(path.is_file() and path.stat().st_size > 0, f'Missing or empty native library: {path}')
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pe = PE(data, str(path))
        return {'exports': pe.exports(), 'imports': pe.imports()}


def check_exports(exports, required, label):
    missing = sorted(name for name in required if not exports.get(name, False))
    require(not missing, f'{label}: missing, empty or forwarded exports: {missing}')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_import_library(path):
    """Read the ARM64 short-import records emitted by MSVC/LLVM linkers.

    Format: https://learn.microsoft.com/windows/win32/debug/pe-format#import-library-format
    Linker symbol indexes and the archive long-name table are metadata, not COFF.
    """
    data = path.read_bytes()
    require(data[:8] == b'!<arch>\n', 'nccl.lib is not a COFF import-library archive')
    offset, symbols = 8, set()
    while offset < len(data):
        require(offset + 60 <= len(data), 'Truncated COFF import archive header')
        header = data[offset:offset + 60]
        require(header[58:] == b'`\n', 'Invalid COFF import archive header')
        size = int(header[48:58].strip())
        start, end = offset + 60, offset + 60 + size
        require(size >= 0 and end <= len(data), 'Truncated COFF import archive member')
        name = header[:16].strip()
        if name not in (b'/', b'//', b'/SYM64/'):
            require(size >= 20, 'Truncated COFF import object')
            member = data[start:end]
            if member[:4] == b'\0\0\xff\xff':
                _, _, version, machine, _, string_size, _, flags = struct.unpack_from('<HHHHIIHH', member)
                require(version == 0 and machine == 0xaa64, 'Expected an ARM64 COFF import object')
                require(0 < string_size <= size - 20, 'Invalid COFF import string size')
                strings = member[20:20 + string_size].split(b'\0')
                require(len(strings) >= 3 and strings[0] and strings[1], 'Invalid COFF import strings')
                require(strings[1].lower() == b'nccl.dll', 'COFF import references a DLL other than nccl.dll')
                name_type = (flags >> 2) & 7
                require(name_type in (1, 2, 3), 'NCCL import must resolve by symbol name')
                symbol = strings[0].decode('ascii')
                if name_type in (2, 3):
                    symbol = symbol.removeprefix('_').removeprefix('@').removeprefix('?')
                if name_type == 3:
                    symbol = symbol.split('@', 1)[0]
                if flags & 3 == 0:  # IMPORT_OBJECT_CODE
                    symbols.add(symbol)
            else:
                require(struct.unpack_from('<H', member)[0] == 0xaa64,
                        'Expected an ARM64 COFF import descriptor')
        offset = end + (size & 1)
    require(REQUIRED_EXPORTS <= symbols,
            f'COFF import library lacks required NCCL APIs: {sorted(REQUIRED_EXPORTS - symbols)}')
    return sorted(symbols)


def inspect_sdk(root, expected_version):
    for relative in ('bin/nccl.dll', 'lib/nccl.lib', 'include/nccl.h', 'LICENSE.txt',
                     'include/nccl_device.h', 'include/nccl_device/gin_win_stub.h',
                     'windows_cuda_versions.json', 'build-info.json'):
        path = root / relative
        require(path.is_file() and path.stat().st_size > 0, f'Missing SDK file: {path}')
    header = (root / 'include/nccl.h').read_text()
    parts = []
    for part in ('MAJOR', 'MINOR', 'PATCH'):
        match = re.search(r'^\s*#define\s+NCCL_' + part + r'\s+(\d+)\s*$', header, re.MULTILINE)
        require(match is not None, f'NCCL header is missing {part} version')
        parts.append(int(match[1]))
    version = parts[0] * 10000 + parts[1] * 100 + parts[2]
    require(version == expected_version, f'NCCL header version {version} != {expected_version}')
    manifest = json.loads((root / 'windows_cuda_versions.json').read_text())
    require(manifest.get('cuda_nccl') == '.'.join(map(str, parts)), 'NCCL manifest version differs from header')
    import_symbols = inspect_import_library(root / 'lib/nccl.lib')
    info = binary_info(root / 'bin/nccl.dll')
    check_exports(info['exports'], REQUIRED_EXPORTS, 'NCCL SDK')
    digest = sha256(root / 'bin/nccl.dll')
    provenance = json.loads((root / 'build-info.json').read_text())
    expected = {'source_commit': EXPECTED_SOURCE, 'version': '.'.join(map(str, parts)),
                'cuda_architectures': EXPECTED_ARCHITECTURES,
                'dll_sha256': digest, 'import_library_sha256': sha256(root / 'lib/nccl.lib')}
    for field, value in expected.items():
        require(provenance.get(field) == value, f'NCCL provenance {field} differs from expected source or SDK')
    return {'sdk_root': str(root.resolve()), 'version': version,
            'nccl_sha256': digest, 'source_commit': EXPECTED_SOURCE,
            'exports': sorted(info['exports']), 'imports': info['imports'],
            'import_library_symbols': import_symbols}


def inspect_wheels(directory, sdk_root, expected_version):
    sdk = inspect_sdk(sdk_root, expected_version)
    payloads = {}
    with tempfile.TemporaryDirectory(prefix='nccl-wheel-check-') as temporary:
        for project, member in (('plugin', 'jax_cuda13_plugin/nccl.dll'),
                                ('pjrt', 'jax_plugins/xla_cuda13/xla_cuda_plugin.so')):
            matches = list(directory.glob(f'jax_cuda13_{project}-*-win_arm64.whl'))
            require(len(matches) == 1, f'Expected one Windows ARM64 CUDA {project} wheel, found {matches}')
            with zipfile.ZipFile(matches[0]) as wheel:
                names = wheel.namelist()
                require(len(names) == len(set(names)), f'{matches[0]}: duplicate wheel member')
                bundled = [name for name in names if name.lower().endswith('/nccl.dll')]
                require(bundled == ([member] if project == 'plugin' else []),
                        f'{matches[0]}: NCCL must have one copy in jax_cuda13_plugin')
                if project == 'plugin':
                    require(wheel.read('jax_cuda13_plugin/LICENSE.nccl.txt') == (sdk_root / 'LICENSE.txt').read_bytes(),
                            'Bundled NCCL license differs from SDK')
                info = wheel.getinfo(member)
                require(0 < info.file_size <= 2 * 1024**3, f'{member}: invalid native payload size')
                target = Path(temporary) / f'{project}.dll'
                with wheel.open(info) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                payloads[project] = binary_info(target)
                if project == 'plugin':
                    require(sha256(target) == sdk['nccl_sha256'], 'Bundled NCCL SHA256 differs from tested SDK')
        check_exports(payloads['pjrt']['exports'], {'GetPjrtApi'}, 'PJRT')
        symbols = payloads['pjrt']['imports'].get('nccl.dll', [])
        require('ncclAllReduce' in symbols, 'PJRT must import nccl.dll and the real ncclAllReduce collective')
        check_exports(payloads['plugin']['exports'], set(symbols) | REQUIRED_EXPORTS, 'Bundled NCCL')
    return {'wheel_dir': str(directory.resolve()), 'version': expected_version,
            'nccl_sha256': sdk['nccl_sha256'], 'pjrt_nccl_imports': sorted(symbols)}


def query_version(path, expected_version, *, by_name=False):
    require(os.name == 'nt', 'Native NCCL version checks require Windows')
    # Keep directory handles alive through loading and both API calls.
    with ExitStack() as stack:
        for variable in ('CUDA_PATH', 'CUDNN_PATH', 'NCCL_PATH'):
            root = os.environ.get(variable)
            if root:
                for relative in ('bin', 'bin/arm64', 'lib/arm64'):
                    directory = Path(root) / relative
                    if directory.is_dir():
                        stack.enter_context(os.add_dll_directory(str(directory.resolve())))
        name = path.name if by_name else str(path.resolve())
        library = ctypes.CDLL(name, winmode=0x1000 if by_name else 0x1100)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        filename = kernel32.GetModuleFileNameW
        filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        filename.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32768)
        length = filename(library._handle, buffer, len(buffer))
        require(0 < length < len(buffer), f'GetModuleFileNameW failed: {ctypes.get_last_error()}')
        loaded = Path(buffer.value).resolve()
        require(loaded == path.resolve(), f'Loaded unexpected NCCL DLL: {loaded}, expected {path}')
        get_version = library.ncclGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        version = ctypes.c_int()
        result = get_version(ctypes.byref(version))
        require(result == 0 and version.value == expected_version,
                f'ncclGetVersion returned status {result}, version {version.value}; expected {expected_version}')
        require(get_version(None) == 4, 'ncclGetVersion(NULL) must return ncclInvalidArgument')
        return {'loaded_library': str(loaded), 'runtime_version': version.value}


def inspect_installed(expected_version):
    require(os.name == 'nt', 'Installed NCCL loading checks require Windows')
    roots = {Path('C:/nccl').resolve()}
    for variable in ('NCCL_PATH', 'LOCAL_NCCL_PATH'):
        value = os.environ.pop(variable, None)
        if value:
            roots.add(Path(value).resolve())
    sdk_directories = {root / relative for root in roots for relative in ('bin', 'bin/arm64', 'lib/arm64')}
    os.environ['PATH'] = os.pathsep.join(entry for entry in os.environ.get('PATH', '').split(os.pathsep)
                                       if entry and Path(entry).resolve() not in sdk_directories)
    spec = importlib.util.find_spec('jax_cuda13_plugin')
    require(spec is not None and spec.submodule_search_locations, 'CUDA plugin package is not installed')
    candidates = [Path(directory) / 'nccl.dll' for directory in spec.submodule_search_locations]
    require(len(candidates) == 1 and candidates[0].is_file(), 'Installed CUDA plugin is missing its NCCL DLL')
    from jaxlib.plugin_support import initialize_windows_cuda_dll_search_path
    initialize_windows_cuda_dll_search_path()
    result = query_version(candidates[0], expected_version, by_name=True)
    result['nccl_sha256'] = sha256(candidates[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    for mode in ('sdk', 'wheels', 'installed'):
        command = modes.add_parser(mode)
        command.add_argument('--expected-version', type=int, default=23206)
        if mode in ('sdk', 'wheels'):
            command.add_argument('--sdk-root', type=Path, required=True)
        if mode == 'wheels':
            command.add_argument('--wheel-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'sdk':
        result = inspect_sdk(args.sdk_root, args.expected_version)
        result.update(query_version(args.sdk_root / 'bin/nccl.dll', args.expected_version))
    elif args.mode == 'wheels':
        result = inspect_wheels(args.wheel_dir, args.sdk_root, args.expected_version)
    else:
        result = inspect_installed(args.expected_version)
    result.update(status='PASS', mode=args.mode, gpu_collective_execution_tested=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
