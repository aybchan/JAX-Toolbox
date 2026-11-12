# Windows ARM64 CUDA and NCCL Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for root integration and delegate independent native-build, XLA, and packaging work. Track verified results below.

**Goal:** Build and verify Windows ARM64 JAX CUDA wheels with real NCCL integration through the authorized CI iteration loop.

**Architecture:** Build a pinned upstream NCCL SDK with native MSVC/NVCC, import that SDK through Bazel's Windows CUDA dependency rules, and enable the previously excluded XLA collective implementation. Bundle the NCCL DLL in the CUDA kernel wheel and verify that the final PJRT binary imports its real collective APIs.

**Tech Stack:** Windows 11 ARM64, CPython 3.14, CUDA 13.4.3, cuDNN 9.28.1, NCCL 2.32.6, MSVC/NVCC, Clang, Bazel 8.7, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-12-windows-arm-cuda-nccl.md`

## Global Constraints

- Use the source pins, SDK layout, GPU targets, and completion evidence in the spec.
- Push only `aybchan/feat/windows-build-cuda-nccl`; preserve the original shared checkout.
- Never claim GPU collective execution from a successful build, CPU test, or version query.
- Retain native patch section boundaries and validate both Git and Bazel patch application.
- No fabricated CUDA/NCCL implementations or silent disabling of ordinary collectives.

## Review Focus

- An x64 or placeholder NCCL library must fail architecture/export validation.
- A stale SDK or mismatched version must fail provenance and runtime version checks.
- Either CUDA package import order must register the bundled DLL directory without circular imports.
- Removing one disable flag must not leave another Windows guard excluding NCCL.
- Unsupported GIN features must not silently replace communication with no-ops.

### Task 1: Build a native NCCL SDK

**Files:** Create `.github/workflows/scripts/build-windows-nccl.ps1`; create `.github/patches/nccl-windows-arm64.patch` only if source fixes are required.

**Interfaces:** Consume the pinned NCCL checkout and `C:/cuda`; produce the SDK layout in the spec. Script parameters: `SourceRoot`, `BuildRoot`, `CudaPath`, `InstallPrefix`, and `Jobs`.

- [x] Inspect upstream Windows CMake paths and reproduce concrete source problems with focused compiler probes where possible.
- [x] Configure native ARM64 MSVC/NVCC, explicitly select the matching GPU architectures, build the shared NCCL target, and stage its actual headers, DLL, import library, license, and version manifest.
- [x] Fail on unsuccessful native commands, missing outputs, source/version mismatch, or incorrect target architecture.
- [x] Validate the script and any patch locally where possible; obtain the decisive compiler result from native CI.

### Task 2: Enable NCCL in the Windows Bazel/XLA path

**Files:** Modify `.github/patches/rules-ml-toolchain-windows-cuda.patch` and `.github/patches/xla-windows-arm64-cuda.patch`.

**Interfaces:** Consume `LOCAL_NCCL_PATH=C:/nccl`; expose real `@cuda_nccl//:nccl` and `:headers` targets with Windows import libraries and required public definitions. Root supplies `--define=no_nccl_support=false` and the matching NCCL version.

- [x] Replace incompatible NCCL targets and Linux-only repository assumptions with native Windows SDK imports.
- [x] Enable XLA's Windows `if_nccl` branch and add native NCCL linkage where Unix stubs were assumed.
- [x] Audit newly exposed GIN/device code and use real existing fallback paths or explicit unsupported errors for unsupported features.
- [x] Apply the full patch chain with Git and Bazel's native patcher; build targeted XLA collective targets in CI before the full wheels.

### Task 3: Package and verify the NCCL runtime

**Files:** Modify `.github/patches/jax-windows-arm64-cuda.patch`; create `.github/workflows/scripts/verify-windows-nccl.py` and focused tests.

**Interfaces:** Consume the staged SDK and final wheel directory. Bundle `jax_cuda13_plugin/nccl.dll` and `LICENSE.nccl.txt`. Register its package directory through the shared Windows DLL helper without executing the package merely to locate it.

- [x] Add meaningful regression checks for both import orders, missing SDK files, and wrong-architecture/missing-export payloads.
- [x] Preserve non-Windows behavior and retain DLL-directory handles for the lifetime of imported extensions.
- [ ] Verify the real SDK and wheel DLL architecture, expected version and required exports; query `ncclGetVersion` natively when loadable.
- [ ] Require the final PJRT PE import table to contain real NCCL collective APIs; keep the existing four-wheel and source-provenance checks.

### Task 4: Integrate the CI loop

**Files:** Modify `.github/workflows/jax-windows-cuda-wheels.yaml`; update this plan with each completed iteration.

**Interfaces:** Build/stage NCCL before Bazel configuration; pass its version/path and enable define to JAX. Include native NCCL logs and source identity in diagnostics and artifacts.

- [x] Add pinned NCCL checkout/build/SDK verification steps and relevant workflow path triggers.
- [x] Remove `--disable_nccl`, explicitly override the Windows no-NCCL define, and extend the existing preflight with real XLA NCCL targets.
- [x] Preserve the existing 8 GiB Bazel cache; add a separate exact-source/toolchain SDK cache if native build time warrants it.
- [x] Include bundled NCCL provenance and clear GPU-runtime limitations in the artifact.
- [ ] Run focused local checks, inspect the complete diff, commit owned paths, and push the same branch.
- [ ] Monitor the exact-head CI job. On failure, let diagnostics/cache finish, classify the log, implement a justified fix, and repeat.
- [ ] Download the successful real artifact and verify hashes, native architecture, NCCL imports/exports, metadata, and provenance before completing the goal.

## Verified baseline

- CUDA-only run `43209855712` passed at `44c101c7`.
- Isolated checkout starts at that exact commit; remote branch confirmed unchanged.
- Existing installer tests: 5 passed. Existing cache tests: 4 passed.
- GPU execution remains unvalidated; the previous CPU smoke test logged a CUDA PJRT loader error on the GPU-less runner.

## Iterations

NCCL iteration 1: implementation ready for the first native CI build.

- All 25 workflow-script tests pass, including negative PE/COFF and provenance checks, both package import orders, declared runtime copying, and missing-license rejection.
- Native PowerShell parsing and build-helper checks pass. Workflow YAML, embedded Python, and all PowerShell steps parse.
- Complete JAX, rules_ml_toolchain, and XLA patches apply cleanly. Native Bazel patching and repository analysis pass; ten selector cases and five runtime dependency cases preserve enable/disable behavior.
- Windows uses the existing ordinary ragged all-to-all fallback; only the unsupported experimental GIN device kernel is excluded.
- Native NCCL compilation, final PJRT linkage, installed DLL loading, and real artifacts remain pending CI.

### Iteration 1 — 15d92ad3

Run `43240511726` failed during native SDK setup. The ARM64 MSVC probe passed, but PowerShell returned both CMake installations; the script treated the resulting array as one executable path. No NCCL compilation or wheel build ran. Select one executable in PATH order for each native tool before retrying.

### Iteration 2 — 96f7ec57

Run `43241667596` passed the native NCCL SDK checks and CUDA/NCCL preflight. Native PJRT linking completed, then its wheel builder failed to import generated `jaxlib.tools.platform_tags`. The Bazel cache and failure diagnostics were saved successfully.

The generated module is already declared in `build_utils.srcs`. A local Bazel reproduction using the actual builder and pinned rules_python passes both direct execution and invocation as an execution-configuration tool. Windows cached-file materialization remains a hypothesis: the next failure diagnostics capture the generated runfile's link type, target, existence, and matching manifest entry, alongside the working `build_utils.py` input. Do not replace the import or remove caching without confirming the cause.

Run the actual PJRT wheel builder's `--help` through an execution-configuration smoke action before GPU preflight. Keep this small action out of the disk/remote cache so a previous successful invocation cannot conceal broken runfiles on a fresh runner; its dependencies retain normal caching. This check validates tool startup only, and the complete wheel builds and validation remain required.

### Branch split requested on November 12

Continue NCCL development only on `aybchan/feat/windows-build-cuda-nccl`, preserving commits `15d92ad3` and `96f7ec57`. Restore `aybchan/feat/windows-build-cuda` to the pre-NCCL commit `44c101c77e5f9f5999c41efb0b25de983d818e15`. This supersedes the earlier hillclimb branch destination; the CUDA-only implementation plan remains historical.

### Iteration 3 — 24fec4e7

The new-branch run `43244507192` built NCCL on native Windows ARM64, passed SDK import-library/export/provenance checks and the DLL loading/version query, and saved the verified SDK cache. CUDA/NCCL collective preflight compilation passed at 17:36 UTC; the full wheel build is running.

A production wheel-input audit found that the JAX root module had not imported the `cuda_nccl` repository used by `jax_plugin_sources.static_srcs`. A native Bazel fixture reproduces the missing apparent-repository mapping and verifies that importing the existing NCCL extension exposes both runtime files. The next iteration imports that shared extension in JAX and includes `jax_plugin_sources` in the existing preflight, so packaging-input resolution is checked before the full wheel build.

Queue subsequent runs on `feat/windows-build-cuda-nccl` behind the active run by disabling cancellation for that branch while keeping the same concurrency group. This lets prepared fixes be pushed without interrupting compilation and skipping the cache save. Pull request refs and other branch refs retain their existing cancellation behavior. GitHub keeps the latest pending run in the group; the active run continues.
