"""Install pinned NVIDIA Windows ARM64 redistributions for the wheel build."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile


def select_package(manifest, component):
    package = manifest[component].get('windows-arm64')
    if not package:
        raise ValueError(f'{component} has no Windows ARM64 redistribution')
    return package.get('cuda13', package)


def download_verified(url, destination, expected_sha256):
    destination = Path(destination)
    partial = destination.with_suffix('.partial')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f'SHA256 mismatch for {url}')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def install_archive(archive_path, destination):
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or any(p in ('.', '..') or ':' in p or '\\' in p for p in path.parts):
                raise ValueError(f'Invalid archive path: {entry.filename}')
            if entry.is_dir() or len(path.parts) < 2:
                continue
            target = destination.joinpath(*path.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--cuda-path', type=Path, default=Path('C:/cuda'))
    parser.add_argument('--cudnn-path', type=Path, default=Path('C:/cudnn'))
    args = parser.parse_args()
    lock = json.loads(args.manifest.read_text())
    for product, destination in [('cuda', args.cuda_path), ('cudnn', args.cudnn_path)]:
        with tempfile.TemporaryDirectory(prefix=f'{product}-arm64-') as tmp:
            for component in lock[product]:
                package = select_package(lock[product], component)
                url = f'https://developer.download.nvidia.com/compute/{product}/redist/' + package['relative_path']
                print(f'Installing {component} ({package["size"]} bytes) from {url}', flush=True)
                archive = download_verified(url, Path(tmp) / f'{component}.zip', package['sha256'])
                install_archive(archive, destination)
                archive.unlink()
    versions = {'cuda': {'name': 'CUDA SDK', 'version': lock['cuda_version']}}
    versions.update({name: {'version': entry['version']} for name, entry in lock['cuda'].items()})
    (args.cuda_path / 'version.json').write_text(json.dumps(versions, indent=2))
    repo_names = {
        'cccl': 'cuda_cccl', 'libcublas': 'cuda_cublas', 'libcufft': 'cuda_cufft',
        'libcurand': 'cuda_curand', 'libcusolver': 'cuda_cusolver', 'libcusparse': 'cuda_cusparse',
        'libnvjitlink': 'cuda_nvjitlink', 'libnvfatbin': 'cuda_nvfatbin',
        'libnvptxcompiler': 'cuda_nvptxcompiler', 'libnvvm': 'cuda_nvvm', 'cuda_nvml_dev': 'cuda_nvml',
    }
    repo_versions = {repo_names.get(name, name): entry['version'] for name, entry in lock['cuda'].items()}
    (args.cuda_path / 'windows_cuda_versions.json').write_text(json.dumps(repo_versions, indent=2))
    (args.cudnn_path / 'windows_cuda_versions.json').write_text(json.dumps({'cuda_cudnn': lock['cudnn']['cudnn']['version']}))
    required = [args.cuda_path / 'bin/nvcc.exe', args.cuda_path / 'bin/ptxas.exe',
                args.cuda_path / 'lib/arm64/cuda.lib', args.cuda_path / 'lib/arm64/cudart.lib',
                args.cudnn_path / 'include/cudnn.h']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing Windows ARM64 CUDA build input: {path}')
    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as env:
            env.write(f'CUDA_PATH={args.cuda_path}\nCUDNN_PATH={args.cudnn_path}\nLOCAL_CUDA_PATH={args.cuda_path}\nLOCAL_CUDNN_PATH={args.cudnn_path}\n')
    if 'GITHUB_PATH' in os.environ:
        with open(os.environ['GITHUB_PATH'], 'a', encoding='utf-8') as env:
            for path in [args.cuda_path / 'bin', args.cuda_path / 'bin/arm64', args.cuda_path / 'lib/arm64', args.cudnn_path / 'bin', args.cudnn_path / 'bin/arm64']:
                if path.is_dir():
                    env.write(f'{path}\n')
    print(f'Installed CUDA {lock["cuda_version"]} and cuDNN {lock["cudnn_version"]} for Windows ARM64', flush=True)


if __name__ == '__main__':
    main()
