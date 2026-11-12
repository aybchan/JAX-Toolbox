# Windows ARM64 CUDA wheel implementation plan

**Goal:** Build native Windows ARM64 JAX CUDA plugin and PJRT wheels in GitHub Actions on `aybchan/feat/windows-build-cuda`.

**Architecture:** Extend the existing pinned Windows ARM CPU build. Install checksum-verified native NVIDIA redistributions, adapt the pinned JAX/XLA/toolchain dependencies through reproducible patches, and use CI failures to drive each subsequent change.

**Tech stack:** GitHub Actions Windows ARM64, Python 3.14, Bazel 8.7.0, Clang/MSVC ARM64, CUDA 13.4.3, cuDNN 9.28.1.

**Spec:** The user explicitly requested an autonomous evaluate/plan/implement/push/monitor loop until Windows ARM CUDA wheels build successfully.

## Constraints

- Work in `/Users/alechan/code/external/JAX-Toolbox` on `feat/windows-build-cuda`; push only to the `aybchan` fork.
- Preserve all unrelated pre-existing local changes and untracked files.
- Read tokens only from macOS Keychain. Never log or persist tokens.
- Preserve the successful CPU build setup and its pinned source revisions.
- Do not substitute CPU-only wheels for the requested CUDA artifacts.
- A hosted runner without an NVIDIA GPU can validate build, packaging, PE architecture, and CPU functionality; GPU execution requires suitable hardware and must be reported separately.

## Tasks

- [x] Inspect previous CI logs and current workflow. Latest CPU run is compiling; prior missing prebuilt-jaxlib failure is already fixed in the base commit.
- [x] Create the requested branch from `4a8041fde020af5cf2cefa00b3bec2ca84ab870e`.
- [ ] Add a pinned Windows ARM64 CUDA/cuDNN installer with checksum and archive-layout tests.
- [ ] Enable CUDA plugin/PJRT targets, retain necessary named Bazel configs, and trigger pushes on the new branch.
- [ ] Adapt dependency selection and native toolchain setup based on pinned-source evidence and CI diagnostics.
- [ ] Adapt Windows linking and packaging, retaining a consistent plugin filename and exported PJRT API.
- [ ] Validate locally, commit only task files using existing commit style, and push.
- [ ] Monitor the relevant CI run, retrieve failing logs, and repeat targeted fixes.
- [ ] Verify CI success and inspect actual Windows ARM64 CUDA wheel artifacts before completing.

## Review focus

- Linux or x64 libraries must never be silently accepted as Windows ARM64 dependencies.
- The workflow must retain transitively referenced CUDA configurations after trimming its Bazel configuration.
- Wheel checks must require both CUDA packages and ARM64 PE native binaries.
- Install/load checks must distinguish absence of GPU hardware from compilation and packaging failures.
- CUDA patches must apply cleanly to the exact pinned sources and fail visibly on drift.

## Iteration ledger

- Initial investigation: CPU workflow strips every CUDA config and requests only jaxlib. Pinned rules_ml_toolchain selects Linux CUDA redistributions and `.so` libraries. JAX GPU target has ELF linker options. Native Windows ARM64 CUDA/cuDNN redistributions exist.
- Ruling: Execute the user's supplied loop directly; the requested branch, destination, and success criterion already authorize implementation and pushes.

- Iteration 1 (`fd78236b`, run 43123733711): native CUDA/cuDNN install and source patching succeeded. Bazel rejected the multi-file toolchain patch because plain unified-diff file headers do not flush its pending hunks. Reproduced with repository_ctx.patch against the checksum-verified pinned archive; adding diff --git boundaries passes the same local Bazel check.

- Iteration 2 (`da2dd15d`, run 43124770129): patches and native dependency setup passed; CUDA compiler lookup failed because where.exe returned no result for the existing absolute Clang path. Added direct Windows path validation and native executable lookup (four new regression cases pass). The next iteration also routes Windows XLA CUDA aliases to real import libraries, avoiding its unsupported Windows assembly-stub generator, and builds the GPU targets before jaxlib.

- Source review during iteration 3 (`27dacd79`, run 43125867480): identified an unused POSIX sys/resource.h include in the cuDNN wrapper and an explicit Windows Unimplemented return in PJRT plugin loading. The candidate loader passes missing-library/export, registration, and duplicate-registration checks; unchanged POSIX loading passes. Combined XLA patch passes native Bazel patch application. Superseding the ongoing run with these source-proven fixes; future pushes share a branch concurrency group.

- Iteration 4 (`e8077db4`, run 43126946847): source patches and dependency setup pass. Bazel exits during early target analysis with 0x80000001 (STATUS_GUARD_PAGE_VIOLATION), without a normal analysis error or printed crash report. Add JVM native-library tracing, a fixed fatal-error output path, Windows Application Error/WER collection, and failure artifacts before changing the runtime or compiler.

- Follow-up source audit: CUDA diagnostics checked PLATFORM_WINDOWS before the header defining it. A preprocessing fixture using the pinned Abseil and XLA headers reproduces POSIX header selection with _WIN32 enabled; including tsl/platform/platform.h before the guards fixes it. Added the explicit Bazel dependency; combined XLA patch passes Git and native Bazel patch parsing.
