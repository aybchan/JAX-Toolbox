# JAX wheels for Windows ARM64

The `Build JAX Windows ARM64 wheels` workflow builds CPU `jax` and `jaxlib`
wheels on `windows-11-arm`, using native ARM64 Python 3.14. `windows-latest`
is an x64 runner and cannot exercise this native ARM64 build.

The source is pinned to `17c608ec2206772f573ae986fe8da097a34b0c39` from
[jax-ml/jax#40345](https://github.com/jax-ml/jax/pull/40345). This includes the
PR's Windows ARM64 toolchain, Python wheel tags and XLA patches.

Four build fixes are applied by `.github/patches/jax-windows-arm64.patch`:

- Attach the PR's six XLA patches to its `archive_override`; the PR adds
  these files but does not apply them.
- Replace the malformed Python toolchain patch with a minimal patch that
  adds Windows ARM64 to the `rules_python` 2.2.0 pip hub.
- Allow `aarch64` as well as `ARM64` in the Python 3.14 SciPy dependency
  marker. `rules_python` uses `aarch64` when resolving Windows ARM64 wheels.
- Remove an extra Mosaic GPU serialization argument that is not present in
  the pinned LLVM API. This code is also compiled for the CPU wheel.

Python 3.14 is used because the PR's 3.12/3.13 dependency locks omit NumPy
for Windows ARM64. Native Bazel 8.7.0 is downloaded with the PR's SHA256
verification; the runner supplies LLVM, Visual Studio's ARM64 C++ tools
and MSYS2.

Bazel's Windows JNI library is built separately by
`.github/workflows/scripts/build-windows-jni.ps1` from its 8.7.0 source,
pinned to `a6d8d66737d56766c0f5377c62ccf2ee562a1a47`. The helper backports
the string-length fix from
[bazelbuild/bazel#30606](https://github.com/bazelbuild/bazel/pull/30606)
([`b247857d`](https://github.com/bazelbuild/bazel/commit/b247857d9934ba9b6fb8c8eff9531f0fd18ec725))
and checks that the resulting DLL is ARM64. Bazel loads it from `C:/b/jni`
through `java.library.path`, retaining the pinned Bazel version.

Bazel runs in batch mode so failures appear in the build log. The workflow
removes unused named configurations from the checked-out JAX `.bazelrc`
to stay within Windows' command-line length limit. It retains global
settings, imports and the five Windows/Clang configurations used by this
build.

The pure Python `jax` wheel is packaged with `pip wheel --no-deps` through
JAX's setuptools backend. This avoids the Bazel JAX target's dependency on
a prebuilt `jaxlib` wheel, which is unavailable for Windows ARM64.
`WHEEL_VERSION_SUFFIX=.dev0+selfbuilt` matches the native `jaxlib` wheel,
which is built separately with `build/build.py build --wheels=jaxlib`.

The workflow runs on pushes to `feat/windows-build` and on pull requests
that change the workflow, its integration patch or the JNI helper.
Once the workflow is on the repository's default branch, it can also be
started from **Actions → Build JAX Windows ARM64 wheels → Run workflow**.

After compiling, the job installs both local wheels and tests JIT matrix
multiplication and differentiation on the CPU. Tests run outside the source
checkout so they import the installed package. Only after these tests pass
does the job upload the `jax-windows-arm64-py3.14` artifact, containing the
wheels, integration patch and source metadata. Artifacts are kept for 14
days.

To install the downloaded wheels on Windows ARM64 with Python 3.14, run
this in PowerShell from the extracted artifact directory:

```powershell
$wheels = (Get-ChildItem *.whl).FullName
python -m pip install --only-binary=:all: $wheels
```

When updating the source commit, check whether the upstream fixes are still
needed and revalidate the integration patch. Remove the JNI helper and
library-path override once the pinned Bazel includes the string-length
fix. Recheck configuration trimming when changing Bazel or JAX's
`.bazelrc`, and keep the JAX version suffix aligned with the native build.
A full build can take several hours on the hosted runner; the job allows
up to six hours.
