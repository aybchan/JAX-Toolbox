#!/usr/bin/env python3
"""Publish one independently verified Actions ZIP without repacking or executing it."""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.parse import quote, urljoin, urlsplit
import zipfile

REPOSITORY = 'aybchan/JAX-Toolbox'
REPOSITORY_ID = 1407207656
BRANCH = 'feat/windows-build-cuda-nccl'
RUN_ID = 43485135810
ARTIFACT_ID = 14082227290
HEAD_SHA = 'ad73a58a62738ae977ee01b20c9fff0d90730845'
ZIP_SHA256 = 'd8a16975104f17fb79eaddaf2bfe8738268553f01d0251753b5d8ad87c10a1c0'
ZIP_SIZE = 405560067
ARTIFACT_NAME = 'jax-windows-arm64-cuda13-nccl-py3.14'
ASSET_NAME = ARTIFACT_NAME + '-ad73a58a.zip'
RELEASE_ID = 444290320
RELEASE_TAG = 'jax-win-arm64-cuda-nccl'
API_PREFIX = '/repos/' + REPOSITORY
DOWNLOAD_PATH = f'{API_PREFIX}/actions/artifacts/{ARTIFACT_ID}/zip'
RELEASE_PATH = f'{API_PREFIX}/releases/{RELEASE_ID}'
UPLOAD_TEMPLATE = 'https://uploads.github.com' + RELEASE_PATH + '/assets{?name,label}'
UPLOAD_PATH = RELEASE_PATH + '/assets?name=' + quote(ASSET_NAME, safe='')
PUBLIC_URL = f'https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{ASSET_NAME}'
API_VERSION = '2026-03-10'
WHEELS = {
    'jax-0.11.2.dev0+selfbuilt-py3-none-any.whl',
    'jaxlib-0.11.2.dev0+selfbuilt-cp314-cp314-win_arm64.whl',
    'jax_cuda13_plugin-0.11.2.dev0+selfbuilt-cp314-cp314-win_arm64.whl',
    'jax_cuda13_pjrt-0.11.2.dev0+selfbuilt-py3-none-win_arm64.whl',
}


class PublishError(Exception):
    """An error safe to print without credentials or signed download URLs."""


def require(condition, message):
    if not condition:
        raise PublishError(message)


class GitHubClient:
    def __init__(self, token, connection_factory=http.client.HTTPSConnection):
        require(bool(token) and '\r' not in token and '\n' not in token, 'Invalid runtime token')
        self.token = token
        self.connection_factory = connection_factory

    @contextmanager
    def response(self, host, method, path, *, authenticated=False, body=None,
                 headers=None, error_path=None):
        # HTTPConnection never follows redirects or retries an upload.
        label = error_path or path.split('?', 1)[0]
        request_headers = {'Accept': 'application/vnd.github+json',
                           'X-GitHub-Api-Version': API_VERSION,
                           'User-Agent': 'jax-windows-wheel-publisher'}
        if authenticated:
            require(host in {'api.github.com', 'uploads.github.com'}, 'Invalid authenticated host')
            request_headers['Authorization'] = 'Bearer ' + self.token
        request_headers.update(headers or {})
        connection = self.connection_factory(host, timeout=300)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            yield connection.getresponse()
        except (OSError, http.client.HTTPException, ValueError):
            raise PublishError(f'Request failed: {label}') from None
        finally:
            connection.close()

    @staticmethod
    def read_json(response, expected_status, error_path):
        require(response.status == expected_status, f'HTTP {response.status}: {error_path}')
        data = response.read(4 * 1024**2 + 1)
        require(len(data) <= 4 * 1024**2, f'Oversized API response: {error_path}')
        try:
            return json.loads(data)
        except (ValueError, UnicodeError):
            raise PublishError(f'Invalid API response: {error_path}') from None

    def api_json(self, path):
        require(path.startswith(API_PREFIX + '/'), 'Invalid API path')
        with self.response('api.github.com', 'GET', path, authenticated=True) as response:
            return self.read_json(response, 200, path.split('?', 1)[0])

    def download_artifact(self, target):
        url = 'https://api.github.com' + DOWNLOAD_PATH
        authenticated = True
        for _ in range(6):
            parsed = urlsplit(url)
            require(parsed.scheme == 'https' and bool(parsed.hostname)
                    and not parsed.username and not parsed.password
                    and parsed.port in (None, 443) and not parsed.fragment,
                    'Unsafe artifact download redirect')
            path = parsed.path + ('?' + parsed.query if parsed.query else '')
            with self.response(parsed.hostname, 'GET', path, authenticated=authenticated,
                               error_path=DOWNLOAD_PATH) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader('Location')
                    require(bool(location), 'Missing artifact download redirect')
                    url = urljoin(url, location)
                    # Never forward the API token, even if a later redirect returns to GitHub.
                    authenticated = False
                    continue
                require(response.status == 200, f'HTTP {response.status}: {DOWNLOAD_PATH}')
                size = 0
                with target.open('wb') as output:
                    while chunk := response.read(1024**2):
                        size += len(chunk)
                        require(size <= ZIP_SIZE, 'Artifact download exceeds pinned size')
                        output.write(chunk)
                return
        raise PublishError('Too many artifact download redirects')

    def upload_asset(self, path):
        require(path.stat().st_size == ZIP_SIZE, 'Upload size differs from pinned artifact')
        # The target is fixed here; the API-provided template is validated separately.
        with path.open('rb') as body, self.response(
                'uploads.github.com', 'POST', UPLOAD_PATH, authenticated=True, body=body,
                headers={'Content-Type': 'application/zip', 'Content-Length': str(ZIP_SIZE)},
                error_path=RELEASE_PATH + '/assets') as response:
            return self.read_json(response, 201, RELEASE_PATH + '/assets')


def validate_run(run):
    for key, expected in {'id': RUN_ID, 'event': 'push', 'status': 'completed',
                          'conclusion': 'success', 'head_sha': HEAD_SHA,
                          'head_branch': BRANCH}.items():
        require(run.get(key) == expected, f'Workflow run mismatch: {key}')
    for key in ('repository', 'head_repository'):
        repository = run.get(key, {})
        require(repository.get('id') == REPOSITORY_ID
                and repository.get('full_name') == REPOSITORY
                and repository.get('private') is False, f'Workflow repository mismatch: {key}')


def validate_artifact(artifact):
    for key, expected in {'id': ARTIFACT_ID, 'name': ARTIFACT_NAME,
                          'size_in_bytes': ZIP_SIZE, 'digest': 'sha256:' + ZIP_SHA256}.items():
        require(artifact.get(key) == expected, f'Artifact mismatch: {key}')
    require(artifact.get('expired') is False, 'Artifact is expired or lacks expiry metadata')
    run = artifact.get('workflow_run', {})
    for key, expected in {'id': RUN_ID, 'head_sha': HEAD_SHA, 'head_branch': BRANCH,
                          'repository_id': REPOSITORY_ID,
                          'head_repository_id': REPOSITORY_ID}.items():
        require(run.get(key) == expected, f'Artifact workflow mismatch: {key}')


def validate_release(release):
    require(release.get('id') == RELEASE_ID and release.get('tag_name') == RELEASE_TAG,
            'Release identity mismatch')
    require(release.get('draft') is False and release.get('prerelease') is True,
            'Expected an existing public prerelease')
    require(release.get('upload_url') == UPLOAD_TEMPLATE, 'Release upload URL mismatch')


def verify_archive(path):
    require(path.stat().st_size == ZIP_SIZE, 'Downloaded ZIP size mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    require(digest.hexdigest() == ZIP_SHA256, 'Downloaded ZIP SHA256 mismatch')
    # The digest binds the prior independent ARM64/NCCL/cuDNN verification.
    # Inspect filenames and provenance only; never import or execute artifact code.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP member names')
        wheel_names = [PurePosixPath(name).name for name in names if name.endswith('.whl')]
        require(len(wheel_names) == 4 and set(wheel_names) == WHEELS, 'Unexpected wheel filename set')
        sources = [name for name in names if PurePosixPath(name).name == 'source.txt']
        require(len(sources) == 1, 'Expected one source.txt record')
        require(archive.getinfo(sources[0]).file_size <= 64 * 1024, 'Oversized source.txt record')
        text = archive.read(sources[0]).decode('utf-8-sig')
        heads = [line.partition(': ')[2] for line in text.splitlines()
                 if line.startswith('JAX-Toolbox workflow commit: ')]
        require(heads == [HEAD_SHA], 'ZIP workflow commit mismatch')


def validate_asset(asset):
    require(isinstance(asset.get('id'), int) and asset['id'] > 0, 'Invalid release asset ID')
    for key, expected in {'name': ASSET_NAME, 'size': ZIP_SIZE,
                          'digest': 'sha256:' + ZIP_SHA256, 'state': 'uploaded',
                          'browser_download_url': PUBLIC_URL}.items():
        require(asset.get(key) == expected, f'Release asset mismatch: {key}; refusing overwrite')


def existing_assets(client):
    matches = []
    for page in range(1, 21):
        assets = client.api_json(f'{RELEASE_PATH}/assets?per_page=100&page={page}')
        require(isinstance(assets, list), 'Invalid release asset listing')
        matches.extend(asset for asset in assets if asset.get('name') == ASSET_NAME)
        if len(assets) < 100:
            require(len(matches) <= 1, 'Duplicate release asset names')
            return matches
    raise PublishError('Release asset pagination limit exceeded')


def publish(client):
    validate_run(client.api_json(f'{API_PREFIX}/actions/runs/{RUN_ID}'))
    validate_artifact(client.api_json(f'{API_PREFIX}/actions/artifacts/{ARTIFACT_ID}'))
    validate_release(client.api_json(RELEASE_PATH))
    with tempfile.TemporaryDirectory(prefix='jax-windows-release-') as directory:
        archive = Path(directory) / ASSET_NAME
        client.download_artifact(archive)
        verify_archive(archive)
        matches = existing_assets(client)
        if matches:
            validate_asset(matches[0])
            return 'reused'
        # One attempt only. An ambiguous network failure must be inspected on rerun.
        validate_asset(client.upload_asset(archive))
        return 'uploaded'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, value in [('--run-id', RUN_ID), ('--artifact-id', ARTIFACT_ID),
                        ('--head-sha', HEAD_SHA), ('--zip-sha256', ZIP_SHA256),
                        ('--zip-size', ZIP_SIZE), ('--release-id', RELEASE_ID)]:
        parser.add_argument(flag, type=type(value), choices=[value], required=True)
    return parser.parse_args(argv)


def runtime_token():
    require(os.environ.get('GITHUB_ACTIONS') == 'true'
            and os.environ.get('GITHUB_REPOSITORY') == REPOSITORY
            and os.environ.get('GITHUB_REF') == 'refs/heads/' + BRANCH
            and os.environ.get('GITHUB_EVENT_NAME') in {'push', 'workflow_dispatch'},
            'Publisher must run in the pinned repository and branch on GitHub Actions')
    token = os.environ.get('GH_TOKEN', '')
    require(bool(token), 'Missing GitHub Actions runtime token')
    return token


def main():
    parse_args()
    try:
        result = publish(GitHubClient(runtime_token()))
        summary = (f'Windows ARM64 CUDA wheels {result}: [{ASSET_NAME}]({PUBLIC_URL})\n\n'
                   f'SHA256: `{ZIP_SHA256}`\n\nSize: {ZIP_SIZE} bytes\n')
        print(summary)
        summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_path:
            with open(summary_path, 'a', encoding='utf-8') as output:
                output.write(summary)
        return 0
    except PublishError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # Never print network exception details, API bodies, or signed URLs.
        print(f'Publication failed ({type(error).__name__}).', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
