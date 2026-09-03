# llama.cpp

![ilustration](ilustration.png)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## MoE Offloading & Memory Optimization

In Mixture of Experts (MoE) models (such as **Qwen 3.8 Flash Next**, **DeepSeek-V2/V3**, **Mixtral**, etc.), expert weights represent the majority of parameters and VRAM. `cafe-llama.cpp` provides flags to offload MoE weights to **pinned host RAM (`CUDA_Host`)** or **CPU RAM** while keeping attention, KV cache, and routers on the GPU:

| Flag                  | Long Flag                | Description |
|-----------------------|--------------------------|---|
| `--pipeline-parallel` | `--no-pipeline-parallel` | Enable the offloading acceleration pipeline, required for `-hmoe`, `-nhmoe`, `-hmoed`. |
| `-hmoe`               | `--host-moe`             | Keep **all MoE expert weights** in pinned host memory (`CUDA_Host`). Enables zero-copy async DMA over PCIe. |
| `-nhmoe N`            | `--n-host-moe N`         | Keep MoE weights of the **first N layers** in pinned host memory. |
| `-cmoe`               | `--cpu-moe`              | Keep **all MoE expert weights** in CPU system RAM. |
| `-ncmoe N`            | `--n-cpu-moe N`          | Keep MoE weights of the **first N layers** in CPU system RAM. |
| `-hmoed`              | `--host-moe-draft`       | Keep draft model MoE weights in pinned host memory (for speculative decoding). |
| `-nhmoed N`           | `--n-host-moe-draft N`   | Keep draft model MoE weights of the **first N layers** in pinned host memory (for speculative decoding). |
| `-cmoed`              | `--cpu-moe-draft`        | Keep draft model MoE weights in CPU system RAM (for speculative decoding). |
| `-ncmoed N`           | `--n-cpu-moe-draft N`    | Keep draft model MoE weights of the **first N layers** in CPU system RAM (for speculative decoding). |

### Qwen 3.8 Flash Next / Qwen4 Internal N-Gram (PLE) Optimization

Qwen 3.8 Flash Next includes an internal Prompt Lookup Expert (PLE) N-gram hash embedding table (`per_layer_token_embd`, ~51B parameters). `cafe-llama.cpp` provides dedicated flags to manage its memory footprint:

| Flag          | Long Flag                               | Description |
|---------------|-----------------------------------------|---|
| `--ngram-ssd` | `--offload-ngram-ssd`, `--no-ngram-ssd` | Exclusively offload the internal N-gram embedding table to SSD on-demand via memory mapping (`mmap`), leaving active model layers in RAM/VRAM. |
| `--no-ngram`  | `--disable-ngram`, `--no-load-ngram`    | Force completely disable the internal N-gram embedding table and PLE layers, skipping all PLE tensors (0 bytes allocated in RAM/VRAM). |
| `--ngram`     | `--load-ngram`                          | Normal mode: load internal N-gram table and PLE layers into memory (default). |

### Qwen 3.8 Flash Next & MTP Speculative Decoding

MTP (Multi-Token Prediction) draft models in GGUF format are available at:
👉 **[Hugging Face: quimmedes/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/quimmedes/Qwen3.8-Flash-Next-MTP-GGUF)**

Available quantizations:
- `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (~2.65 GB) - Recommended for balanced memory and speed
- `mtp-Qwen3.8-Flash-Next-Q6_K.gguf` (~3.24 GB)
- `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (~3.94 GB)
- `mtp-Qwen3.8-Flash-Next-BF16.gguf` (~7.40 GB) - Full precision

**Recommended Server Command for Qwen 3.8 Flash Next:**
```sh

llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
-ctk q8_0 -ctv q8_0 -kvu \
 -fa on -ngl 99 -nhmoe 36 -c 64000 --pipeline-parallel -np 1 --no-ngram 

Disable Ngram if you don't have enough RAM/VRAM
llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
-ctk q8_0 -ctv q8_0 -kvu \
 -fa on -ngl 99 -nhmoe 36 -c 64000 \
--no-ngram --pipeline-parallel -np 1 --no-ngram 



MTP with offload
llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl 99 \
  -nhmoe 36 \
  -fa on \
  -ctk q8_0 -ctv q8_0 -kvu \
  -c 64000 --pipeline-parallel --no-ngram 
  
  
  //Faster above 30% context load
  
  llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -md mtp.gguf -ctk q8_0 -ctv q8_0 -kvu -fa on -c 64000 \
  -np 1 -t 8 -b 1024 -ub 128 --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 2 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64 -ctkd q4_0 -ctvd q4_0 \
  --pipeline-parallel -nhmoe 34 -ngl 99 -ngld 99 --no-ngram 
 
  
  
```

## Building from Source

### 1. NVIDIA CUDA (Windows / Linux)
```sh
# CMake configure with CUDA backend
cmake -B build -DGGML_CUDA=ON

# Build Release
cmake --build build --config Release -j 2 
```

### 2. Vulkan (Cross-Platform AMD / Intel / NVIDIA)
```sh
# Requires Vulkan SDK installed
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j 2
```

### 3. AMD ROCm / HIP (Linux / Windows)
```sh
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1100;gfx1030"
cmake --build build --config Release -j 2
```

### 4. Apple Metal (macOS)
```sh
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j 2
```

### 5. CPU Only (AVX2 / AVX-512)
```sh
cmake -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build --config Release -j 2
```


-j N : The number of CPU threads the processor will use to compile, 
a safe number is the amount of physical cores of the processor.

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity


The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
