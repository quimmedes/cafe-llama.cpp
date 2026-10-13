# Bundled XDNA2 GEMM dependencies

This directory contains the dependency subset used by `ggml-ryzenai`. Builds and inference do not require a FastFlowLM checkout, installation, executable, or model format. System XRT development files and runtime, the XDNA userspace plugin, and the installed amdxdna driver are still required.

The files were copied unchanged from [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM/tree/6c9ed874908e4b793014a87bce4b43af644ef0db), commit `6c9ed874908e4b793014a87bce4b43af644ef0db`:

| Files | Purpose |
| --- | --- |
| `include/modules/gemm.hpp` and its transitive headers | GEMM API, instruction generation, XRT buffers and context management |
| `lib/libgemm.so` | Precompiled Linux x86_64 GEMM sequence generator |
| `lib/libaiebu.a` | Matching precompiled transaction-to-ELF assembler |
| `xclbins/mm.xclbin` | XDNA2 BF16 GEMM device program, from `src/xclbins/Qwen3-0.6B-NPU2/mm.xclbin` |

`PROVENANCE.json` records each upstream file path and SHA-256 hash, plus hashes of the license files. There are 67 headers and approximately 7.5 MB of bundled files. FastFlowLM's executable, model runners, model data, HRX runtime, other device kernels, and kernel-driver implementation are excluded.

CMake uses the bundled headers, copies the shared GEMM library and device program into the build tree, and links the aiebu archive statically. Installation includes the shared GEMM library, device program, this document, provenance and all license files. Keep the library, archive, headers and kernel as a compatible set when updating the bundle. Updates require rerunning native NPU tests and CPU/NPU model comparisons described in [the backend documentation](../../../../docs/backend/RYZENAI.md).

## Notices

These files retain their original licenses; the project's MIT license does not relicense them. Original notices inside the headers are preserved.

- FastFlowLM utility sources: [MIT](licenses/FastFlowLM-MIT.txt), copyright 2026 Advanced Micro Devices, Inc.
- Precompiled artifacts from FastFlowLM: original [terms for proprietary binaries](licenses/FastFlowLM-TERMS.md), copied from the same commit. The upstream GEMM/kernel source is not included in this bundle; these artifacts are not built from source by cafe-llama.cpp.
- `include/aiebu/aiebu.h`: MIT, copyright 2024-2025 Advanced Micro Devices, Inc. See [third-party MIT notices](licenses/ThirdParty-MIT.txt).
- `include/nlohmann/**`: MIT, copyright 2013-2025 Niels Lohmann and, for its Hedley headers, 2016-2021 Evan Nemerson. See [third-party MIT notices](licenses/ThirdParty-MIT.txt).
- `include/biovault_bfloat16.h`: [Apache-2.0](licenses/Apache-2.0.txt), copyright 2020 LKEB, Leiden University Medical Center; adapted from oneDNN. Its attribution and original source URL remain in the header.
- `include/npu_utils/amdxdna_accel.h`: GPL-2.0 WITH Linux-syscall-note, copyright 2022-2024 Advanced Micro Devices, Inc. This is a userspace UAPI header. See [GPL-2.0](licenses/GPL-2.txt) and the [Linux syscall exception](licenses/Linux-syscall-note.txt).
