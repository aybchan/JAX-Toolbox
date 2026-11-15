"""Exercise the patched C++ ISA query with captured Windows ptxas output.

The subprocess boundary is a test double; this does not execute Windows ptxas.
The production query body is compiled, including its pipe configuration and
failure guards. The unchanged PTX parser is replaced with a byte-exact oracle.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


PATCH = Path(__file__).resolve().parents[3] / 'patches/xla-windows-arm64-cuda.patch'
SOURCE = 'xla/stream_executor/cuda/subprocess_compilation_provider.cc'
# Pinned XLA 91888df6ce85102c30220e41d952065925e10886, lines 117-142.
ORIGINAL = '''absl::StatusOr<int> SubprocessCompilationProvider::GetLatestPtxIsaVersion()
    const {
  std::vector<std::string> ptxas_args = {path_to_ptxas_, "--input-as-string",
                                         ".version 99.99"};
  tsl::SubProcess ptxas_info_dumper;
  ptxas_info_dumper.SetProgram(path_to_ptxas_, ptxas_args);
  ptxas_info_dumper.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!ptxas_info_dumper.Start()) {
    return absl::InternalError("Failed to launch ptxas");
  }
  std::string stderr_output;
  int exit_status = ptxas_info_dumper.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  if (exit_status == 0) {
    return absl::InternalError("ptxas succeeded where it was expected to fail");
  }

  return GetLatestPtxIsaVersionFromUnsupportedVersionErrorLog(stderr_output);
}

std::string SubprocessCompilationProvider::name() const {
  return absl::StrFormat("SubprocessCompilationProvider(ptxas: %s, nvlink: %s)",
                         path_to_ptxas_, path_to_nvlink_);
}

}  // namespace stream_executor::cuda
'''
# Captured on Windows ARM64, CUDA 13.4.92: stdout has 101 bytes, stderr is empty.
WINDOWS_STDOUT = ("ptxas application ptx input, line 1; fatal   : Unsupported "
                  ".version 99.99; current version is '9.4'\r\n")

HARNESS = r'''
#include <iostream>
#include <string>
#include <vector>
namespace absl {
struct Status { std::string message; };
Status InternalError(std::string message) { return {message}; }
template<class T> struct StatusOr {
  bool ok; T value{}; std::string message;
  StatusOr(T value) : ok(true), value(value) {}
  StatusOr(Status status) : ok(false), message(status.message) {}
};
}
namespace fixture {
std::string out, err, expected;
bool launch = true;
int exit_code = -1;
}
namespace tsl {
enum { CHAN_STDOUT = 1, CHAN_STDERR = 2, ACTION_PIPE = 1 };
class SubProcess {
  bool out_pipe = false, err_pipe = false;
 public:
  void SetProgram(const std::string& path, const std::vector<std::string>& args) {
    if (args != std::vector<std::string>{path, "--input-as-string", ".version 99.99"})
      throw "ISA probe arguments changed";
  }
  void SetChannelAction(int channel, int action) {
    if (channel == CHAN_STDOUT) out_pipe = action == ACTION_PIPE;
    if (channel == CHAN_STDERR) err_pipe = action == ACTION_PIPE;
  }
  bool Start() { return fixture::launch; }
  int Communicate(const std::string*, std::string* out, std::string* err) {
    if (out_pipe && out) *out = fixture::out;
    if (err_pipe && err) *err = fixture::err;
    return fixture::exit_code;
  }
};
}
namespace stream_executor::cuda {
absl::StatusOr<int> GetLatestPtxIsaVersionFromUnsupportedVersionErrorLog(
    const std::string& log) {
  if (log.empty() || log != fixture::expected)
    return absl::InternalError("Compiler diagnostic was lost or changed");
  return 94;
}
class SubprocessCompilationProvider {
  std::string path_to_ptxas_ = "fake path/ptxas";
 public:
  absl::StatusOr<int> GetLatestPtxIsaVersion() const;
};
QUERY_BODY
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string mode = argv[1];
  const std::string captured = WINDOWS_FIXTURE;
  if (mode == "stdout" || mode == "both" || mode == "unexpected_success")
    fixture::out = captured;
  if (mode == "stderr") fixture::err = captured;
  if (mode == "both") fixture::err = "additional compiler diagnostic\n";
  if (mode == "launch_failure") fixture::launch = false;
  if (mode == "unexpected_success") fixture::exit_code = 0;
  fixture::expected = fixture::out + fixture::err;
  auto result = stream_executor::cuda::SubprocessCompilationProvider().GetLatestPtxIsaVersion();
  const bool should_pass = mode == "stdout" || mode == "stderr" || mode == "both";
  if (result.ok != should_pass || (result.ok && result.value != 94)) {
    std::cerr << mode << ": " << result.message << "\n";
    return 1;
  }
  return 0;
}
'''


class WindowsPtxIsaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('c++')
        if not compiler:
            raise unittest.SkipTest('A C++17 compiler is required for the production query regression')
        cls.temporary = tempfile.TemporaryDirectory(prefix='ptx-isa-query-')
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        source = root / SOURCE
        source.parent.mkdir(parents=True)
        source.write_text(ORIGINAL, encoding='utf-8')
        sections = re.split(r'(?=^diff --git )', PATCH.read_text(encoding='utf-8'), flags=re.MULTILINE)
        patch = next((part for part in sections if part.startswith(f'diff --git a/{SOURCE} ')), '')
        if patch:
            subprocess.run(['git', 'apply', '-'], cwd=root, input=patch,
                           text=True, check=True, capture_output=True)
        cls.patch = patch
        cls.patched_source = source.read_bytes()
        # Compile the actual patched method; only dependencies outside that method are doubled.
        query = source.read_text(encoding='utf-8').split('\nstd::string SubprocessCompilationProvider::name()')[0]
        program = root / 'probe.cc'
        program.write_text(HARNESS.replace('QUERY_BODY', query).replace(
            'WINDOWS_FIXTURE', json.dumps(WINDOWS_STDOUT)), encoding='utf-8')
        cls.executable = root / ('probe.exe' if os.name == 'nt' else 'probe')
        subprocess.run([compiler, '-std=c++17', str(program), '-o', str(cls.executable)],
                       text=True, check=True, capture_output=True, timeout=60)

    def check_mode(self, mode):
        result = subprocess.run([str(self.executable), mode], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_patch_applies_to_windows_crlf_checkout(self):
        root = Path(self.temporary.name) / 'crlf'
        source = root / SOURCE
        source.parent.mkdir(parents=True)
        source.write_bytes(ORIGINAL.replace('\n', '\r\n').encode('utf-8'))
        result = subprocess.run(
            ['git', 'apply', '-'], cwd=root,
            input=self.patch.replace('\n', '\r\n').encode('utf-8'),
            capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8'))
        self.assertEqual(source.read_bytes().replace(b'\r\n', b'\n'),
                         self.patched_source.replace(b'\r\n', b'\n'))

    def test_windows_stdout_version_diagnostic_is_retained(self):
        self.check_mode('stdout')

    def test_existing_stderr_version_diagnostic_is_retained(self):
        self.check_mode('stderr')

    def test_both_output_channels_reach_the_version_parser(self):
        self.check_mode('both')

    def test_empty_diagnostic_stays_an_error(self):
        self.check_mode('empty')

    def test_unexpected_success_stays_an_error(self):
        self.check_mode('unexpected_success')

    def test_launch_failure_stays_an_error(self):
        self.check_mode('launch_failure')


if __name__ == '__main__':
    unittest.main()
