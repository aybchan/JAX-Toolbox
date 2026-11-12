# Windows ARM64 CUDA and NCCL wheels

Extend the successful CUDA build at `44c101c77e5f9f5999c41efb0b25de983d818e15` with genuine NCCL support. Iterate on GitHub Actions failures until the wheels build, install, and pass package validation. Push only to `aybchan/feat/windows-build-cuda`.

## Build contract

- Preserve JAX source `17c608ec2206772f573ae986fe8da097a34b0c39`, Python 3.14, CUDA 13.4.3, cuDNN 9.28.1, and the existing Windows ARM64 fixes.
- Build NCCL source `ad2d192c1f279aefcaf340f67e07880b65a1116b` (2.32.6) with native Windows ARM64 tools. Pin any subsequent source change explicitly.
- Stage the NCCL SDK at `C:/nccl`: `include/`, `lib/nccl.lib`, `bin/nccl.dll`, `LICENSE.txt`, `LICENSE.nccl.txt`, `build-info.json`, and `windows_cuda_versions.json` containing `{"cuda_nccl":"2.32.6"}`.
- Match the existing JAX GPU targets: sm_75, sm_80, sm_90, sm_100, sm_110, and sm_120 plus compute_120 PTX.
- Enable ordinary XLA NCCL collectives using real NCCL headers and libraries. Do not replace communication with no-op implementations or report success with NCCL disabled.
- Keep NVSHMEM disabled. Unsupported Windows GIN/DOCA paths may use a genuine existing fallback or report unsupported functionality explicitly.
- Bundle one NCCL runtime DLL and its license in `jax_cuda13_plugin`. Register that directory before loading CUDA extensions or PJRT, retaining Windows DLL directory handles.
- Preserve the existing CUDA artifact checks, CPU execution checks, diagnostics, and bounded Bazel cache.

## Completion evidence

The successful CI commit must produce the four real JAX wheels, install them on native Windows ARM64 Python, and pass CPU smoke tests. Inspect all native payloads as ARM64 PE files. Verify the bundled NCCL DLL, its actual version, required exports, and PJRT's real NCCL imports. Verify source and patch provenance against that commit.

Run a native NCCL version query without a GPU when its actual DLL dependencies permit it. Do not ignore a loader failure or substitute a fabricated driver library. A loader limitation must be diagnosed and reported accurately.

The hosted runner has no NVIDIA GPU. This task establishes the NCCL-enabled wheel build and packaging; correctness and performance of GPU collectives require a separate hardware test. Preserve that limitation in artifacts and the final report.

## Workspace

Another chat owns the original repository checkout on `feat/remove-fixed-prealloc`. Work in the isolated checkout at `/Users/alechan/Documents/ChatGPT/Nov 10/JAX-Toolbox-nccl`; leave the original checkout unchanged. Use macOS Keychain exclusively for API tokens and the existing SSH remote for authorized pushes.
