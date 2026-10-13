# Ryzen AI XDNA2 NPU (experimental)

`GGML_RYZENAI=ON` adds the `RyzenAI` device to GGML. It runs supported weight matrix multiplies on an AMD XDNA2 NPU using a bundled BF16 GEMM kernel from FastFlowLM and system XRT. GGUF weights are read directly. Builds and inference do not require an external FastFlowLM checkout, installation, executable, or model conversion. The standard GGML scheduler handles CPU fallback and transfers between devices.

This is partial offload. Attention, normalization, convolution, activation functions, small projections, dynamically computed weight matrices, and operations with explicit F32 precision requirements remain on CPU. The backend accepts two-dimensional weight matrices with contiguous rows, K >= 512 and N >= 128, and F32/F16/BF16 activations, including activation batch planes. GGML quantized weight types with a `to_float` converter are unpacked on the host. Padded input, output and weight buffers are each limited to 512 MiB. Larger operations remain on CPU.

BF16 arithmetic can change embeddings and generated answers. NPU use does not guarantee a speedup, especially for single-token decoding: the kernel pads token rows to multiples of 256. Check correctness and performance with your own inputs before relying on this backend.

## Requirements

- Linux x86_64, XDNA2 PCI device `1022:17f0`, revision `10` (Strix Point, including Ryzen AI 9 HX370) or `11` (Strix Halo). Other NPU generations are not accepted by this kernel integration.
- Working `amdxdna` driver and firmware, with access to `/dev/accel/accel0`. The driver manages buffers, hardware contexts and command submission; it does not contain the GEMM kernel.
- XRT development headers and runtime, including the XDNA userspace plugin. Check `xrt-smi examine`. Sufficient memlock allowance is needed for DMA buffers.
- XRT development packages (`libxrt-dev`, `uuid-dev`, `libdrm-dev`, or equivalent local prefixes).
- The bundled headers, `libgemm.so`, `libaiebu.a`, and XDNA2 `mm.xclbin` in [`ggml/src/ggml-ryzenai/vendor`](../../ggml/src/ggml-ryzenai/vendor/README.md). They are included in this repository with original notices, binary terms, and per-file SHA-256 provenance. No downloads are made for these assets during configuration or build.

The local integration was tested with FastFlowLM commit `6c9ed874908e4b793014a87bce4b43af644ef0db` and amdxdna-dkms checkout `bc8437aa9eb3dbf71cfe16de99e6a6aa6e9e7960`, on HX370 with XRT 2.21.75, kernel 7.0.0-38-generic and firmware 1.1.2.64. No driver changes are made by this build.

## Build

From the cafe-llama.cpp checkout:

```bash
cmake --preset ryzenai
cmake --build build-ryzenai --target llama-app llama-cli llama-server test-ryzenai test-backend-ops -j8
build-ryzenai/bin/llama-cli --list-devices
```

The `ryzenai` preset enables both `GGML_RYZENAI` and `GGML_VULKAN`; it requires the Vulkan development files and `glslc`. For NPU plus CPU only:

```bash
cmake -S . -B build-ryzenai-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_RYZENAI=ON
cmake --build build-ryzenai-cpu -j8
```

`GGML_RYZENAI` is off by default. Disabled builds do not require XRT or the bundled assets. Set `GGML_RYZENAI_XRT_ROOT` for a local XRT prefix. The default kernel is bundled; set the CMake cache variable `GGML_RYZENAI_XCLBIN` to select a matching kernel at build time, or the environment variable of the same name to override it at runtime. Kernel/library/header versions must match; unrelated xclbins are not interchangeable. Old auto-detected FastFlowLM paths are replaced with the bundled defaults when reconfiguring an existing build.

## Projector on NPU, language model on GPU

Use the device names printed by `--list-devices`. On the test machine `Vulkan0` is Intel Arc B580 and `Vulkan1` is Radeon 890M; device numbering may differ on another machine. Replace `Vulkan0` with `Vulkan1` to use the HX370's integrated GPU.

```bash
build-ryzenai/bin/llama-cli \
    -m samplemodel/Qwen3.5-4B-Q4_K_M.gguf \
    --mmproj samplemodel/mmproj-BF16.gguf \
    --device Vulkan0 --mmproj-device RyzenAI -ngl 99 \
    --image /path/to/image.jpg -p 'Briefly describe this image.' \
    --image-max-tokens 256 -c 1024 -b 256 -ub 256 \
    --reasoning off --single-turn -n 128
```

`--device` selects the language model devices, while `--mmproj-device` independently selects the projector device. Use `--device none -ngl 0 --mmproj-device RyzenAI` for CPU language inference with NPU projector offload. Use `--mmproj-device none` for a CPU projector baseline. The same options work with `llama-server` and the unified `llama server` binary.

## Split language model layers between NPU and GPU

```bash
build-ryzenai/bin/llama-cli \
    -m samplemodel/Qwen3.5-4B-Q4_K_M.gguf \
    --device RyzenAI,Vulkan0 --split-mode layer \
    --tensor-split 2,31 -ngl 99 \
    -c 512 -b 128 -ub 128 --reasoning off --single-turn \
    -p 'Reply with a short greeting.' -n 32
```

The split values are relative proportions in device order, not layer counts. For this 32-layer model the placement includes the output layer, so `2,31` assigns the first two repeating layers to RyzenAI and the remaining layers to Vulkan0. Within NPU-assigned layers only supported matrix multiplies run on NPU; other operators use CPU. `-ngl` controls the total number of offloaded placements, including the output layer. Use `--device RyzenAI -ngl 4` to select the last three repeating layers and the output placement; the large output projection exceeds the NPU buffer limit and remains on CPU.

Only `--split-mode layer` and single-device `--split-mode none` are supported by this integration. Row/tensor splitting, GPU/NPU concurrent execution of a single layer, and full-layer NPU execution are not implemented. Layer execution follows model dependencies. Projector selection can be combined with layer splitting.

## Memory and execution

Immutable weight buffers cache packed BF16 matrices. This needs about two bytes per padded weight element in addition to the original GGUF storage, GPU allocations and activations. A Q4_K_M model can therefore need substantially more RAM after offload. Start with a small NPU layer share on a 16 GiB system. Cache entries belong to the model buffer and are released with it; writes through GGML tensor APIs invalidate them. Mutable inputs are packed on each run.

Language-model and projector backends share one XRT device/context and serialize their NPU work. Errors in NPU initialization or execution are reported; an explicitly requested missing `RyzenAI` device is rejected. CPU fallback is for operations declared unsupported before execution, not a way to conceal a failed NPU dispatch.

The build copies the bundled `libgemm.so` and kernel into `build-ryzenai/bin`. Installation copies the GEMM library alongside the GGML backend and the kernel to `share/llama.cpp/ryzenai/mm.xclbin`, together with provenance and all license files. Keep the installed directory tree together. XRT, the XDNA userspace plugin and the kernel driver remain system dependencies. The vendored sources retain their original licenses; the precompiled artifacts retain FastFlowLM's [proprietary binary terms](../../ggml/src/ggml-ryzenai/vendor/licenses/FastFlowLM-TERMS.md) and are not rebuilt from source by this project.

The preset's install RPATH supports a relocated `bin`/`lib`/`share` tree. A staged install under `/tmp/cafe-ryzenai-bundled-install`, executed from `/tmp`, passed a combined test with the projector on NPU and language-model layers split between NPU and Radeon 890M. FastFlowLM's source and installation directories and the cafe-llama.cpp source checkout were hidden in a temporary mount namespace; the sample GGUF files were mounted separately. It executed 98 projector NPU matmuls and 240 language-model NPU matmuls. A separate `GGML_RYZENAI=OFF` build passed and did not link XRT or the GEMM library.

## Validation

```bash
ctest --test-dir build-ryzenai -R '^test-ryzenai$' --output-on-failure
build-ryzenai/bin/test-backend-ops test -b RyzenAI -o MUL_MAT \
    -p 'm=160,n=(1|257),k=768'
```

The hardware integration test checks actual NPU dispatch counts, NPU/CPU scheduler assignment, repeated execution, growing/shrinking token counts, cache invalidation, two backend instances sharing a runtime, and rejection of overflowing dimensions/strides and explicit F32 precision requirements. It skips when the hardware/runtime is absent. The 12 matrix tests cover F32/F16/BF16/Q4_K/Q6_K/Q8_0 against the CPU reference, including padding and broadcast activation planes. Both sets passed on the HX370.

A fresh NPU/CPU build (`build-ryzenai-bundled`) passed without any external FastFlowLM CMake paths. The bundled backend was then recompiled and the hardware test rerun with `/home/banpot/FastFlowLM` and `/opt/fastflowlm` hidden in a temporary mount namespace. Both backend instances executed nine NPU matmuls, with zero error for the test's exactly representable inputs. All 72 upstream files and six license files matched their recorded SHA-256 hashes; the installed library, kernel and license files also matched the bundle.

Set `GGML_RYZENAI_TRACE=1` and add `-v` to the CLI to log each actual NPU multiply. On the sample model, the two-layer split executed 144 NPU matmuls including graph reservation and inference, and generated `Hello! How can I help you today?` with either Intel Arc B580 or Radeon 890M as the GPU. The projector executed 98 NPU matmuls for `FastFlowLM/src/tb_files/pcb.jpg` at a 256-token image limit, producing 252 image embeddings of width 2560.

For the same image and settings, `MTMD_DEBUG_EMBEDDINGS=/path/to/output.bin` captures projector embeddings. The format is two int32 values (token count, embedding width) followed by F32 data. A CPU/NPU comparison of all 645120 values had cosine similarity 0.970007, relative RMSE 0.243176, RMSE 0.0374791 and maximum absolute difference 3.06068. All outputs were finite. This confirms substantial numerical drift; these smoke tests do not establish model accuracy. Prefer CPU/GPU projector execution when this drift is unacceptable. No speedup claim is made from these runs.

Powered by [FastFlowLM](https://github.com/ROCm/FastFlowLM). See its [Linux setup guide](https://github.com/ROCm/FastFlowLM/blob/main/docs/linux-getting-started.md) for runtime installation.
