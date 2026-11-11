# JAX wheels for Windows ARM64

The `Build JAX Windows ARM64 wheels` workflow builds CPU `jax` and `jaxlib`
wheels on `windows-11-arm`, using native ARM64 Python 3.14. `windows-latest`
is an x64 runner and cannot exercise this native ARM64 build.

The source is pinned to `17c608ec2206772f573ae986fe8da097a34b0c39` from
[jax-ml/jax#40345](https://github.com/jax-ml/jax/pull/40345). This includes the
PR's Windows ARM64 toolchain, Python wheel tags and XLA patches.

Two build fixes are applied by `.github/patches/jax-windows-arm64.patch`:

- Attach the PR's six XLA patches to its `archive_override`; the PR adds
  these files but does not apply them.
- Correct the Python toolchain patch's hunk and preserve the blank line in
  `rules_python` 2.2.0 so it applies to the pinned dependency.

Python 3.14 is used because the PR's 3.12/3.13 dependency locks omit NumPy
for Windows ARM64. Bazel is downloaded with the PR's SHA256 verification;
the runner supplies LLVM, Visual Studio's ARM64 C++ tools and MSYS2.

The workflow runs on pushes to `feat/windows-build` and on pull requests
that change the workflow or its integration patch.
Once the workflow is on the repository's default branch, it can also be
started from **Actions → Build JAX Windows ARM64 wheels → Run workflow**.

After compiling, the job installs both local wheels and tests JIT matrix
multiplication and differentiation on the CPU. Tests run outside the source
checkout so they import the installed package. The
`jax-windows-arm64-py3.14` artifact contains the wheels, the integration
patch and source metadata. Artifacts are kept for 14 days.

To install the downloaded wheels on Windows ARM64 with Python 3.14, run
this in PowerShell from the extracted artifact directory:

```powershell
$wheels = (Get-ChildItem *.whl).FullName
python -m pip install --only-binary=:all: $wheels
```

When updating the source commit, check whether the upstream fixes are still
needed and revalidate the integration patch. A full build can take several
hours on the hosted runner; the job allows up to six hours.
