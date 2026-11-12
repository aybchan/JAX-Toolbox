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
- [x] Add a pinned Windows ARM64 CUDA/cuDNN installer with checksum and archive-layout tests.
- [x] Enable CUDA plugin/PJRT targets, retain necessary named Bazel configs, and trigger pushes on the new branch.
- [x] Adapt dependency selection and native toolchain setup based on pinned-source evidence and CI diagnostics.
- [x] Adapt Windows linking and packaging, retaining a consistent plugin filename and exported PJRT API.
- [x] Validate locally, commit only task files using existing commit style, and push.
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

- Iteration 3 completed with a concrete compile failure after 12,778 actions: clang-cl rejects separate -x and cuda response-file lines. Reproduced both CI diagnostics locally; Windows now emits the joined -xcuda argument, which produces an ARM64 COFF object. Linux flags are unchanged. Twelve focused compiler tests and native Bazel patch application pass. Preserve failed compiler response files in CI diagnostics.

- Shorten compiler iterations: configure once, build the exact bf16 CUDA sort-kernel target first, then reuse its outputs for the complete wheel build. Iteration 5 diagnostics confirmed the patched JNI DLL loaded; analysis reached 29,960 configured targets without reproducing the native crash before supersession.

- Iteration 6 (`b39df1c7`, run 43129034881): the focused CUDA kernel passes header parsing and device assembly, then fails fatbinary packaging. Reproduced the cause: clang-cl /Fo assigns the same output file to every GPU architecture, so later ptxas outputs overwrite earlier cubins. Patch rules_cc 0.2.20 to use forwarded -o for clang-cl C/C++ actions, retaining /Fo for MSVC and assembly. Three compiler regression checks pass, including distinct multiarch device outputs and a real ARM64 CPU object. All requested GPU architectures remain enabled.

- Iteration 7 (`d330c23a`, run 43130537316): CUDA compiler completes the bf16 kernel, including all device images and ARM64 host compilation; Bazel rejects absolute /showIncludes entries under C:/cuda/include. Declare LOCAL_CUDA_PATH/include in the Windows clang-cl builtin include roots. Native Starlark checks pass for unset/empty values, normal paths, Windows separators, spaces, and escaping. Include validation remains enabled and cuDNN/INCLUDE settings are unchanged.

- Iteration 8 (`bbfccabf`, run 43131830675): the real multiarchitecture bf16 CUDA kernel preflight succeeds. Full-wheel Bazel then crashes during analysis with native access violation 0xC0000005; patched JNI is confirmed loaded, and neither HotSpot nor Windows events produced a useful trace. Controlled runtime experiment: run both Bazel invocations on the installed native ARM64 JDK 21 (Bazel 8.7 targets Java 21) instead of its embedded newer JVM. This is not yet a confirmed crash root cause.

- Iteration 9 (`8dc71024`, run 43133104448): JDK 21 completes kernel preflight and full target analysis. Compilation reaches the Householder CUDA kernel, then Clang rejects Abseil __declspec(thread) during device parsing. Use the standard thread_local spelling only in the Windows Clang CUDA device pass; host declarations and ARM64 COFF objects remain identical, and real device TLS remains rejected. Append to the existing pinned Abseil override. Also add a 4 GiB saved Bazel disk cache keyed by runner/toolchain identity, retained after failures and skipped after cancellation; four filesystem tests pass.

- Shared checkout update: commit d97ca720 already fixes the Mosaic NVVM builder mismatch found in CPU baseline run 43116751667 after 7,362 seconds. It is included in the current CUDA branch; independently verified optional predicate behavior and ARM64 syntax. Preserve that shared commit and its CPU workflow/documentation changes.

- Iteration 10 (`e842538a`, run 43136423589): CUDA preflight and full target analysis pass; compilation reaches linalg device assembly and ptxas reports an unresolved static copysign(float, float). The pinned Windows CUDA headers declare but omit this device overload. Use libcudacxx copysign for CUDA tridiagonal solves, retaining std::copysign for CPU/ROCm. Expand the early compile check to all four JAX CUDA source units. The first disk cache saved successfully (34,860,079 bytes).

- Installed-wheel source review: register Windows CUDA DLL directories before eager kernel extension imports, using one retained-handle helper shared with PJRT discovery. Make the PJRT loader search these directories through the documented Windows search flags and an absolute Unicode path. Honor CUDA_PATH when CUDA_ROOT is absent on Windows so XLA can locate libdevice. Import-order, path-selection, and mocked WinAPI loader regressions pass; actual GPU execution still requires NVIDIA hardware.

- Iteration 11 (`5955bd53`, run 43138761245): all five kernel preflight targets pass, including the repaired linalg CUDA kernel. Full build reaches native extension linking; sparse and PRNG both fail on cuGetErrorName from cuda_gpu_kernel_helpers. Their link response files include cudart.lib but omit cuda.lib. Add direct driver/runtime dependencies to the shared helper and link both affected extensions in preflight. The cache restored successfully and saved 107,231,139 bytes after failure.
