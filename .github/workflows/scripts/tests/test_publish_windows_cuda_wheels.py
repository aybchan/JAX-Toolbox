"""Exercise pinned publication and HTTP boundaries without network or credentials."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / 'publish-windows-cuda-wheels.py'


class Response(io.BytesIO):
    def __init__(self, status=200, data=b'', headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)


class Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.sent_bodies = []

    def __call__(self, host, timeout):
        owner = self
        class Connection:
            def request(self, method, path, body=None, headers=None):
                owner.requests.append((host, method, path, body, dict(headers)))
                owner.sent_bodies.append(body.read() if hasattr(body, 'read') else body)
            def getresponse(self):
                response = next(owner.responses)
                if isinstance(response, Exception):
                    raise response
                return response
            def close(self):
                pass
        return Connection()


class PublisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SCRIPT.is_file(), 'The pinned publisher implementation is missing'
        spec = importlib.util.spec_from_file_location('windows_wheel_publisher', SCRIPT)
        cls.pub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.pub)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = self.pub
        self.run = {'id': p.RUN_ID, 'event': 'push', 'status': 'completed',
                    'conclusion': 'success', 'head_sha': p.HEAD_SHA,
                    'head_branch': p.BRANCH,
                    'repository': {'id': p.REPOSITORY_ID, 'full_name': p.REPOSITORY, 'private': False},
                    'head_repository': {'id': p.REPOSITORY_ID, 'full_name': p.REPOSITORY, 'private': False}}
        self.artifact = {'id': p.ARTIFACT_ID, 'name': p.ARTIFACT_NAME,
                         'size_in_bytes': p.ZIP_SIZE, 'digest': 'sha256:' + p.ZIP_SHA256,
                         'expired': False, 'workflow_run': {'id': p.RUN_ID,
                         'head_sha': p.HEAD_SHA, 'head_branch': p.BRANCH,
                         'repository_id': p.REPOSITORY_ID, 'head_repository_id': p.REPOSITORY_ID}}
        self.release = {'id': p.RELEASE_ID, 'tag_name': p.RELEASE_TAG,
                        'draft': False, 'prerelease': True, 'upload_url': p.UPLOAD_TEMPLATE}
        self.asset = {'id': 123, 'name': p.ASSET_NAME, 'size': p.ZIP_SIZE,
                      'digest': 'sha256:' + p.ZIP_SHA256, 'state': 'uploaded',
                      'browser_download_url': p.PUBLIC_URL}

    def client(self, responses):
        transport = Transport(responses)
        return self.pub.GitHubClient('dummy-test-token', connection_factory=transport), transport

    def fixture(self, *, head=None, extra=None, duplicate_source=False):
        p = self.pub
        path = self.root / 'fixture.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('source.txt', 'JAX-Toolbox workflow commit: ' + (head or p.HEAD_SHA) + '\r\n')
            for name in p.WHEELS:
                archive.writestr(name, b'wheel fixture; never executed')
            if extra:
                archive.writestr(extra, b'extra')
            if duplicate_source:
                archive.writestr('nested/source.txt', 'other')
        return path

    def verify_fixture(self, path):
        with mock.patch.multiple(self.pub, ZIP_SIZE=path.stat().st_size,
                                 ZIP_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()):
            self.pub.verify_archive(path)

    def test_expected_run_artifact_and_release(self):
        self.pub.validate_run(self.run)
        self.pub.validate_artifact(self.artifact)
        self.pub.validate_release(self.release)

    def test_run_identity_and_success_are_required(self):
        for key, value in [('id', 1), ('event', 'pull_request'), ('status', 'in_progress'),
                           ('conclusion', 'failure'), ('head_sha', 'a' * 40),
                           ('head_branch', 'main')]:
            with self.subTest(key=key), self.assertRaises(self.pub.PublishError):
                self.pub.validate_run(dict(self.run, **{key: value}))
        for key in ['repository', 'head_repository']:
            altered = copy.deepcopy(self.run)
            altered[key]['full_name'] = 'attacker/fork'
            with self.subTest(key=key), self.assertRaises(self.pub.PublishError):
                self.pub.validate_run(altered)

    def test_artifact_pins_are_required(self):
        for key, value in [('id', 1), ('name', 'different'), ('size_in_bytes', 1),
                           ('digest', 'sha256:' + '0' * 64), ('expired', True)]:
            with self.subTest(key=key), self.assertRaises(self.pub.PublishError):
                self.pub.validate_artifact(dict(self.artifact, **{key: value}))
        for key, value in [('id', 1), ('head_sha', 'a' * 40), ('head_branch', 'main'),
                           ('repository_id', 1), ('head_repository_id', 1)]:
            altered = copy.deepcopy(self.artifact)
            altered['workflow_run'][key] = value
            with self.subTest(key=key), self.assertRaises(self.pub.PublishError):
                self.pub.validate_artifact(altered)

    def test_release_identity_and_upload_target_are_required(self):
        for key, value in [('id', 1), ('tag_name', 'other'), ('draft', True),
                           ('prerelease', False), ('upload_url', 'https://attacker.example/upload')]:
            with self.subTest(key=key), self.assertRaises(self.pub.PublishError):
                self.pub.validate_release(dict(self.release, **{key: value}))

    def test_archive_hash_size_and_contents(self):
        self.verify_fixture(self.fixture())
        with self.assertRaises(self.pub.PublishError):
            self.pub.verify_archive(self.fixture())
        for options in [{'head': 'a' * 40}, {'extra': 'unexpected.whl'}, {'duplicate_source': True}]:
            with self.subTest(options=options), self.assertRaises(self.pub.PublishError):
                self.verify_fixture(self.fixture(**options))

    def test_download_redirect_drops_auth_and_preserves_bytes(self):
        payload = b'PK\x00\xff\r\nexact bytes'
        client, transport = self.client([
            Response(302, headers={'Location': 'https://blob.example/archive?sig=secret'}),
            Response(200, payload),
        ])
        target = self.root / 'download.zip'
        client.download_artifact(target)
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(transport.requests[0][4]['Authorization'], 'Bearer dummy-test-token')
        self.assertNotIn('Authorization', transport.requests[1][4])
        self.assertEqual(transport.requests[1][0], 'blob.example')

    def test_auth_stays_absent_if_redirect_returns_to_api(self):
        client, transport = self.client([
            Response(302, headers={'Location': 'https://blob.example/archive?sig=secret'}),
            Response(302, headers={'Location': 'https://api.github.com/redirected'}),
            Response(200, b'zip'),
        ])
        client.download_artifact(self.root / 'download.zip')
        self.assertNotIn('Authorization', transport.requests[1][4])
        self.assertNotIn('Authorization', transport.requests[2][4])

    def test_unsafe_redirect_is_rejected_before_second_request(self):
        for url in ['http://blob.example/archive', 'https://user:password@blob.example/x',
                    'https://blob.example:8443/x', 'file:///tmp/archive']:
            client, transport = self.client([Response(302, headers={'Location': url})])
            with self.subTest(url=url), self.assertRaises(self.pub.PublishError):
                client.download_artifact(self.root / 'download.zip')
            self.assertEqual(len(transport.requests), 1)

    def test_http_error_never_discloses_signed_url_body_or_token(self):
        client, _ = self.client([
            Response(302, headers={'Location': 'https://blob.example/archive?sig=secret'}),
            Response(403, b'token dummy-test-token; signed secret URL'),
        ])
        with self.assertRaises(self.pub.PublishError) as error:
            client.download_artifact(self.root / 'download.zip')
        message = str(error.exception)
        self.assertIn('403', message)
        self.assertIn('/actions/artifacts/', message)
        for secret in ['secret', 'dummy-test-token', 'blob.example', '?sig']:
            self.assertNotIn(secret, message)

    def test_download_size_is_bounded(self):
        client, _ = self.client([Response(200, b'12345')])
        with mock.patch.object(self.pub, 'ZIP_SIZE', 4), self.assertRaises(self.pub.PublishError):
            client.download_artifact(self.root / 'download.zip')

    def test_upload_is_single_raw_post_to_fixed_target(self):
        p = self.pub
        path = self.root / 'artifact.zip'
        path.write_bytes(b'exact raw bytes')
        reply = dict(self.asset, size=path.stat().st_size)
        client, transport = self.client([Response(201, json.dumps(reply).encode())])
        with mock.patch.object(p, 'ZIP_SIZE', path.stat().st_size):
            self.assertEqual(client.upload_asset(path), reply)
        self.assertEqual(len(transport.requests), 1)
        host, method, endpoint, body, headers = transport.requests[0]
        self.assertEqual((host, method, endpoint), ('uploads.github.com', 'POST', p.UPLOAD_PATH))
        self.assertTrue(hasattr(body, 'read'))
        self.assertEqual(transport.sent_bodies, [b'exact raw bytes'])
        self.assertEqual(headers['Content-Length'], str(path.stat().st_size))
        self.assertEqual(headers['Content-Type'], 'application/zip')
        self.assertEqual(headers['X-GitHub-Api-Version'], '2026-03-10')
        self.assertEqual(headers['Authorization'], 'Bearer dummy-test-token')

    def test_upload_failure_does_not_retry(self):
        path = self.root / 'artifact.zip'
        path.write_bytes(b'zip')
        for response in [Response(403, b'dummy-test-token'), OSError('signed-secret-url')]:
            client, transport = self.client([response])
            with mock.patch.object(self.pub, 'ZIP_SIZE', 3), self.assertRaises(self.pub.PublishError) as error:
                client.upload_asset(path)
            self.assertEqual(len(transport.requests), 1)
            self.assertNotIn('dummy-test-token', str(error.exception))
            self.assertNotIn('signed-secret-url', str(error.exception))

    def test_only_exact_existing_asset_can_be_reused(self):
        self.pub.validate_asset(self.asset)
        for key, value in [('name', 'other.zip'), ('size', 1), ('digest', None),
                           ('digest', 'sha256:' + '0' * 64), ('state', 'new'),
                           ('browser_download_url', 'https://attacker.example/file')]:
            with self.subTest(key=key, value=value), self.assertRaises(self.pub.PublishError):
                self.pub.validate_asset(dict(self.asset, **{key: value}))

    def fake_client(self, assets, upload=None):
        p = self.pub
        client = mock.Mock()
        client.api_json.side_effect = [self.run, self.artifact, self.release, assets]
        client.upload_asset.return_value = upload or self.asset
        return client

    def test_publish_reuses_without_upload(self):
        client = self.fake_client([self.asset])
        with mock.patch.object(self.pub, 'verify_archive'):
            self.assertEqual(self.pub.publish(client), 'reused')
        client.upload_asset.assert_not_called()
        client.download_artifact.assert_called_once()

    def test_publish_creates_only_after_verification(self):
        client = self.fake_client([])
        events = []
        client.download_artifact.side_effect = lambda path: events.append('download')
        client.upload_asset.side_effect = lambda path: events.append('upload') or self.asset
        with mock.patch.object(self.pub, 'verify_archive', side_effect=lambda path: events.append('verify')):
            self.assertEqual(self.pub.publish(client), 'uploaded')
        self.assertEqual(events, ['download', 'verify', 'upload'])

    def test_mismatched_existing_asset_fails_without_upload_or_delete(self):
        client = self.fake_client([dict(self.asset, digest='sha256:' + '0' * 64)])
        with mock.patch.object(self.pub, 'verify_archive'), self.assertRaises(self.pub.PublishError):
            self.pub.publish(client)
        client.upload_asset.assert_not_called()
        self.assertFalse(any(call[0].startswith('delete') for call in client.method_calls))

    def test_duplicate_existing_assets_are_rejected(self):
        client = self.fake_client([self.asset, self.asset])
        with mock.patch.object(self.pub, 'verify_archive'), self.assertRaises(self.pub.PublishError):
            self.pub.publish(client)
        client.upload_asset.assert_not_called()

    def test_archive_verification_failure_blocks_publication(self):
        client = self.fake_client([])
        with mock.patch.object(self.pub, 'verify_archive', side_effect=self.pub.PublishError('bad ZIP')):
            with self.assertRaises(self.pub.PublishError):
                self.pub.publish(client)
        client.upload_asset.assert_not_called()

    def test_upload_response_digest_is_checked(self):
        client = self.fake_client([], dict(self.asset, digest=None))
        with mock.patch.object(self.pub, 'verify_archive'), self.assertRaises(self.pub.PublishError):
            self.pub.publish(client)
        client.upload_asset.assert_called_once()

    def test_cli_requires_exact_pins(self):
        p = self.pub
        args = ['--run-id', str(p.RUN_ID), '--artifact-id', str(p.ARTIFACT_ID),
                '--head-sha', p.HEAD_SHA, '--zip-sha256', p.ZIP_SHA256,
                '--zip-size', str(p.ZIP_SIZE), '--release-id', str(p.RELEASE_ID)]
        p.parse_args(args)
        for position in range(1, len(args), 2):
            altered = args[:]
            altered[position] = '1'
            with self.subTest(position=position), mock.patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                p.parse_args(altered)

    def test_runtime_guard_precedes_token_access(self):
        with mock.patch.dict(self.pub.os.environ, {}, clear=True), self.assertRaises(self.pub.PublishError):
            self.pub.runtime_token()
        valid = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': self.pub.REPOSITORY,
                 'GITHUB_REF': 'refs/heads/' + self.pub.BRANCH,
                 'GITHUB_EVENT_NAME': 'push', 'GH_TOKEN': 'dummy-test-token'}
        with mock.patch.dict(self.pub.os.environ, valid, clear=True):
            self.assertEqual(self.pub.runtime_token(), 'dummy-test-token')


if __name__ == '__main__':
    unittest.main()
