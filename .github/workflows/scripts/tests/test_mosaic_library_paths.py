"""Regress Mosaic's actual library-list formatter against pinned MLIR options.

Set JAX_TEST_SOURCE to the pinned JAX checkout with the repository's base and
CUDA patches applied. Set MLIR_PASS_REGISTRY_SOURCE to PassRegistry.cpp from
LLVM 7024b9e1b423b3c3c6ac76ab6a73cb2c9e4ef842 (SHA256 checked below).
An optional CXX selects a C++17 compiler; otherwise clang++ or g++ is used.

The compiled harness uses the unchanged upstream option-parser function bodies
with a std::string_view-backed StringRef shim. It tests the two option parsing
layers and exact path values, not the complete MLIR pipeline parser, pass
execution, library loading, or GPU execution. In particular, unmatched braces
in paths are outside this regression's scope because the outer pipeline parser
counts braces even inside quotes.
"""
import ast
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

LLVM_COMMIT = '7024b9e1b423b3c3c6ac76ab6a73cb2c9e4ef842'
PASS_REGISTRY_SHA256 = '2745fe4bc20da2fc80c6ed4710d8ce2f6c4e11958cd763a3eb7ffb0e5bd09b84'
CAPTURED_LIBDEVICE = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4/nvvm/libdevice/libdevice.10.bc'

# Host-independent type adapters only. Parser bodies are read from the
# checksum-pinned fixture; the production formatter is read from patched JAX.
STRING_REF_SHIM = r"""
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
class StringRef {
  std::string_view v;
public:
  static constexpr size_t npos=std::string_view::npos;
  StringRef()=default;
  StringRef(std::string_view s):v(s){}
  StringRef(const std::string &s):v(s){}
  size_t size()const{return v.size();} bool empty()const{return v.empty();}
  char operator[](size_t n)const{return v[n];} char front()const{return v.front();} char back()const{return v.back();}
  size_t find_first_of(char c,size_t n=0)const{return v.find_first_of(c,n);}
  StringRef take_front(size_t n)const{return v.substr(0,n);}
  StringRef drop_front(size_t n=1)const{return v.substr(n);}
  StringRef drop_back(size_t n=1)const{return v.substr(0,v.size()-n);}
  StringRef ltrim()const{auto n=v.find_first_not_of(" \t\n\v\f\r");return n==npos?StringRef():StringRef(v.substr(n));}
  StringRef trim()const{auto s=ltrim();auto n=s.v.find_last_not_of(" \t\n\v\f\r");return n==npos?StringRef():StringRef(s.v.substr(0,n+1));}
  std::string str()const{return std::string(v);}
};
[[noreturn]] void llvm_unreachable(const char*){std::abort();}
using LogicalResult=bool;
LogicalResult success(){return true;} LogicalResult failure(){return false;}
bool failed(LogicalResult r){return !r;}
template<class T>using function_ref=std::function<T>;
namespace llvm {namespace cl {struct Option {};}}
namespace detail {namespace pass_options {LogicalResult parseCommaSeparatedList(llvm::cl::Option&,StringRef,StringRef,function_ref<LogicalResult(StringRef)>);}}
"""

HARNESS_MAIN = r"""
int main(int argc, char **argv) {
  if (argc < 3) return 10;
  auto [key, value, remainder] = parseNextArg(StringRef(std::string_view(argv[1])));
  if (key.str() != "libraries-to-link" || !remainder.empty()) {
    std::cerr << "Unconsumed options: " << remainder.str() << "\n";
    return 1;
  }
  llvm::cl::Option unused;
  std::vector<std::string> actual;
  auto status = detail::pass_options::parseCommaSeparatedList(
      unused, key, value, [&](StringRef path) {
        actual.push_back(path.str());
        return success();
      });
  std::vector<std::string> expected(argv + 2, argv + argc);
  if (!status || actual != expected) {
    for (const auto &path : actual) std::cerr << "Parsed path: " << path << "\n";
    return 2;
  }
  return 0;
}
"""


def extract_formatter(source):
    # Extract both formatting inputs from production rather than reproducing
    # the proposed quoting in the test. Removing either fix must fail tests.
    matches = re.findall(
        r'^\s*(mosaic-gpu-module-to-assembly\{libraries-to-link=.*\}),\s*$',
        source, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError('Expected one actual Mosaic library pass declaration')
    option_format = matches[0].removeprefix('mosaic-gpu-module-to-assembly{').removesuffix('}')
    joins = re.findall(r'absl::StrJoin\(libraries_to_link,\s*("(?:[^"\\]|\\.)*")\)', source)
    if len(joins) != 1 or option_format.count('%3$s') != 1:
        raise ValueError('Expected one actual libraries_to_link formatter')
    delimiter = ast.literal_eval(joins[0])  # This C++ literal contains only quotes/commas.
    return lambda paths: option_format.replace('%3$s', delimiter.join(paths))


class MosaicLibraryPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get('JAX_TEST_SOURCE')
        parser = os.environ.get('MLIR_PASS_REGISTRY_SOURCE')
        if not source or not parser:
            raise unittest.SkipTest('Set JAX_TEST_SOURCE and MLIR_PASS_REGISTRY_SOURCE to pinned source files')
        cpp = (Path(source) / 'jaxlib/mosaic/gpu/custom_call.cc').read_text(encoding='utf-8')
        cls.format_option = staticmethod(extract_formatter(cpp))
        parser_bytes = Path(parser).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(parser_bytes).hexdigest() != PASS_REGISTRY_SHA256:
            raise ValueError('PassRegistry.cpp differs from pinned LLVM ' + LLVM_COMMIT)
        parser_source = parser_bytes.decode('utf-8')
        tokenizer = parser_source[parser_source.index('static size_t findChar'):
                                  parser_source.index('/// Out of line virtual function')]
        argument_parser = parser_source[
            parser_source.index('static std::tuple<StringRef, StringRef, StringRef>\nparseNextArg'):
            parser_source.index('LogicalResult detail::PassOptions::parseFromString')]
        compiler = shlex.split(os.environ['CXX']) if os.environ.get('CXX') else [
            shutil.which('clang++') or shutil.which('g++')]
        if not compiler[0]:
            raise unittest.SkipTest('A C++17 compiler is required for pinned option-parser tests')
        temporary = Path(cls.enterClassContext(tempfile.TemporaryDirectory(prefix='mosaic-library-paths-')))
        harness = temporary / 'parser.cc'
        harness.write_text(STRING_REF_SHIM + tokenizer + argument_parser + HARNESS_MAIN, encoding='utf-8')
        cls.executable = temporary / ('parser.exe' if os.name == 'nt' else 'parser')
        built = subprocess.run([*compiler, '-std=c++17', str(harness), '-o', str(cls.executable)],
                               capture_output=True, text=True, timeout=60)
        if built.returncode:
            raise AssertionError('Pinned parser harness did not compile: ' + built.stderr)

    def run_parser(self, option, paths):
        return subprocess.run([str(self.executable), option, *paths],
                              capture_output=True, text=True, timeout=5)

    def assert_roundtrip(self, paths):
        result = self.run_parser(self.format_option(paths), paths)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_captured_windows_libdevice_path(self):
        self.assert_roundtrip([CAPTURED_LIBDEVICE])

    def test_two_libraries_keep_spaces_commas_and_literal_backslashes(self):
        self.assert_roundtrip([CAPTURED_LIBDEVICE,
                               r'C:\Program Files, shared\NVSHMEM\libnvshmem_device.bc'])

    def test_apostrophes_and_balanced_braces_are_preserved(self):
        self.assert_roundtrip([r"C:\User's Files\{CUDA}\libdevice.bc",
                               r'C:\CUDA\{tool,kit}\nvshmem.bc'])

    def test_linux_paths_with_one_or_two_libraries(self):
        for paths in [
            ['/opt/cuda/nvvm/libdevice/libdevice.10.bc'],
            ['/opt/cuda/nvvm/libdevice/libdevice.10.bc', '/opt/nvshmem/lib/device.bc'],
            ['/opt/cuda kit,local/libdevice.bc', "/opt/user's nvshmem/lib/device.bc"],
        ]:
            with self.subTest(paths=paths):
                self.assert_roundtrip(paths)

    def test_original_unquoted_path_is_rejected(self):
        result = self.run_parser('libraries-to-link=' + CAPTURED_LIBDEVICE, [CAPTURED_LIBDEVICE])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Files', result.stderr)

    def test_one_outer_quote_does_not_preserve_comma_in_path(self):
        paths = [r'C:\Program Files, shared\libdevice.bc', r'C:\NVSHMEM\device.bc']
        result = self.run_parser('libraries-to-link="' + ','.join(paths) + '"', paths)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
