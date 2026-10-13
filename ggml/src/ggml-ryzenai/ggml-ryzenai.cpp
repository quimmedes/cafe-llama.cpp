#include "ggml-ryzenai.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "modules/gemm.hpp"
#include "npu_utils/npu_utils_xrt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <sys/sysinfo.h>

static constexpr int64_t tile_m = 256;
static constexpr int64_t tile_k = 512;
static constexpr int64_t tile_n = 128;
static constexpr int64_t max_elements = (512 * 1024 * 1024) / sizeof(bf16);

static std::string ryzenai_xclbin_path() {
    const char * path = std::getenv("GGML_RYZENAI_XCLBIN");
    if (path && *path) {
        return path;
    }
    std::error_code ec;
    auto executable = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        auto installed = executable.parent_path() / GGML_RYZENAI_INSTALLED_XCLBIN;
        if (std::filesystem::is_regular_file(installed, ec)) {
            return installed.string();
        }
    }
    return GGML_RYZENAI_DEFAULT_XCLBIN;
}

static bool ryzenai_has_xdna2() {
    std::error_code ec;
    std::filesystem::directory_iterator entries("/sys/bus/pci/drivers/amdxdna", ec);
    if (ec) {
        return false;
    }
    for (const auto & entry : entries) {
        std::string vendor, device, revision;
        std::ifstream(entry.path() / "vendor") >> vendor;
        std::ifstream(entry.path() / "device") >> device;
        std::ifstream(entry.path() / "revision") >> revision;
        if (vendor == "0x1022" && device == "0x17f0" && (revision == "0x10" || revision == "0x11")) {
            return true;
        }
    }
    return false;
}

static int64_t ryzenai_pad(int64_t n, int64_t tile) {
    return ((n + tile - 1) / tile) * tile;
}

static size_t ryzenai_weight_index(int64_t k, int64_t n, int64_t padded_k) {
    // FastFlowLM layout: [N/128, K/512, N_tile/8, K_inner, K_tile/8, N_inner].
    return (n / tile_n) * (padded_k / tile_k) * 65536 + (k / tile_k) * 65536 +
           ((n % tile_n) / 8) * 4096 + (k % 8) * 512 + ((k % tile_k) / 8) * 8 + n % 8;
}

static void ryzenai_read_row(const ggml_tensor * tensor, const void * src, std::vector<float> & row) {
    row.resize(tensor->ne[0]);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(row.data(), src, row.size() * sizeof(float));
    } else {
        ggml_get_type_traits(tensor->type)->to_float(src, row.data(), tensor->ne[0]);
    }
}

struct ryzenai_weight {
    buffer<bf16> bo;
    int64_t k;
    int64_t n;
};

struct ryzenai_runtime {
    xrt::device device;
    npu_xclbin_manager manager;
    npu_app_manager * app_manager;
    LM_Config config;
    Gemm gemm;
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::unique_ptr<npu_app>> apps;
    buffer<bf16> input;
    buffer<bf16> output;
    size_t input_capacity = 0;
    size_t output_capacity = 0;
    std::mutex mutex;

    ryzenai_runtime() : device(0), manager(device_npu2, &device),
        app_manager(manager.register_xclbin(ryzenai_xclbin_path())), gemm(config) {
        if (gemm.get_m() != 64 || gemm.get_k() != tile_k || gemm.get_n() != tile_n) {
            throw std::runtime_error("FastFlowLM GEMM tile ABI mismatch");
        }
    }

    ryzenai_weight pack_weight(const ggml_tensor * tensor) {
        const int64_t k = ryzenai_pad(tensor->ne[0], tile_k);
        const int64_t n = ryzenai_pad(tensor->ne[1], tile_n);
        auto bo = app_manager->create_bo_buffer<bf16>(k * n);
        std::fill(bo.data(), bo.data() + k * n, bf16(0.0f));
        std::vector<float> row;
        for (int64_t j = 0; j < tensor->ne[1]; ++j) {
            ryzenai_read_row(tensor, static_cast<const char *>(tensor->data) + j * tensor->nb[1], row);
            for (int64_t i = 0; i < tensor->ne[0]; ++i) {
                bo.data()[ryzenai_weight_index(i, j, k)] = bf16(row[i]);
            }
        }
        bo.sync_to_device();
        return { std::move(bo), k, n };
    }

    void compute(ggml_tensor * node, ryzenai_weight & w) {
        const auto * x = node->src[1];
        const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
        const int64_t m = ryzenai_pad(rows, tile_m);
        const size_t input_size = m * w.k;
        const size_t output_size = m * w.n;
        if (input_capacity < input_size) {
            input = app_manager->create_bo_buffer<bf16>(input_size);
            input_capacity = input_size;
        }
        if (output_capacity < output_size) {
            output = app_manager->create_bo_buffer<bf16>(output_size);
            output_capacity = output_size;
        }
        std::fill(input.data(), input.data() + input_size, bf16(0.0f));
        std::vector<float> row;
        int64_t flat = 0;
        for (int64_t b3 = 0; b3 < x->ne[3]; ++b3) {
            for (int64_t b2 = 0; b2 < x->ne[2]; ++b2) {
                for (int64_t r = 0; r < x->ne[1]; ++r, ++flat) {
                    const auto * src = static_cast<const char *>(x->data) + b3 * x->nb[3] + b2 * x->nb[2] + r * x->nb[1];
                    ryzenai_read_row(x, src, row);
                    for (int64_t k = 0; k < x->ne[0]; ++k) {
                        input.data()[flat * w.k + k] = bf16(row[k]);
                    }
                }
            }
        }
        auto key = std::make_tuple(m, w.k, w.n);
        if (apps.find(key) == apps.end() && apps.size() >= 32) {
            apps.erase(apps.begin());
        }
        auto & app = apps[key];
        if (!app) {
            app = std::make_unique<npu_app>(app_manager->create_app());
            gemm.generate_seq(app->seq(), m, w.k, w.n, 0, false, Gemm::NO_Activation, 0);
            app->seq()->cmds2seq();
            app->update_ctrl_seq();
        }
        input.sync_to_device();
        // Patch indices: output, input, packed weights.
        auto run = app->create_run(output, input, w.bo);
        run.start();
        auto state = run.wait(std::chrono::milliseconds(30000));
        if (state == ERT_CMD_STATE_TIMEOUT) {
            run.abort();
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error("NPU GEMM failed, ERT state=" + std::to_string(int(state)));
        }
        output.sync_from_device();
        auto * dst = static_cast<float *>(node->data);
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t n = 0; n < node->ne[0]; ++n) {
                dst[r * node->ne[0] + n] = float(output.data()[r * w.n + n]);
            }
        }
    }
};

static std::shared_ptr<ryzenai_runtime> ryzenai_get_runtime() {
    static std::mutex mutex;
    static std::weak_ptr<ryzenai_runtime> instance;
    std::lock_guard<std::mutex> lock(mutex);
    auto runtime = instance.lock();
    if (!runtime) {
        if (!ryzenai_has_xdna2()) {
            throw std::runtime_error("no supported XDNA2 device bound to amdxdna");
        }
        xrt::xclbin kernel(ryzenai_xclbin_path());
        auto kernels = kernel.get_kernels();
        if (std::none_of(kernels.begin(), kernels.end(), [](const auto & k) { return k.get_name().rfind("MLIR_AIE", 0) == 0; })) {
            throw std::runtime_error("GEMM xclbin has no MLIR_AIE kernel");
        }
        runtime = std::make_shared<ryzenai_runtime>();
        instance = runtime;
    }
    return runtime;
}

struct ryzenai_buffer_context {
    ggml_backend_buffer_t host;
    std::shared_ptr<ryzenai_runtime> runtime;
    using weight_key = std::tuple<uintptr_t, ggml_type, int64_t, int64_t, size_t>;
    std::map<weight_key, ryzenai_weight> weights;

    explicit ryzenai_buffer_context(ggml_backend_buffer_t host) : host(host) {}

    ~ryzenai_buffer_context() {
        weights.clear();
        ggml_backend_buffer_free(host);
    }
};

static ryzenai_buffer_context * ryzenai_buffer_context_get(ggml_backend_buffer_t buffer) {
    return static_cast<ryzenai_buffer_context *>(buffer->context);
}

static void ryzenai_buffer_free(ggml_backend_buffer_t buffer) {
    delete ryzenai_buffer_context_get(buffer);
}

static void * ryzenai_buffer_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_get_base(ryzenai_buffer_context_get(buffer)->host);
}

static void ryzenai_buffer_reset(ggml_backend_buffer_t buffer) {
    ryzenai_buffer_context_get(buffer)->weights.clear();
}

static void ryzenai_buffer_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ryzenai_buffer_reset(buffer);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void ryzenai_buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ryzenai_buffer_reset(buffer);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void ryzenai_buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static bool ryzenai_buffer_copy(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }
    ryzenai_buffer_set(buffer, dst, src->data, 0, ggml_nbytes(src));
    return true;
}

static void ryzenai_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ryzenai_buffer_reset(buffer);
    ggml_backend_buffer_clear(ryzenai_buffer_context_get(buffer)->host, value);
}

static const ggml_backend_buffer_i ryzenai_buffer_i = {
    /* .free_buffer   = */ ryzenai_buffer_free,
    /* .get_base      = */ ryzenai_buffer_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ryzenai_buffer_memset,
    /* .set_tensor    = */ ryzenai_buffer_set,
    /* .get_tensor    = */ ryzenai_buffer_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ryzenai_buffer_copy,
    /* .clear         = */ ryzenai_buffer_clear,
    /* .reset         = */ ryzenai_buffer_reset,
};

static const char * ryzenai_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "RyzenAI";
}

static ggml_backend_buffer_t ryzenai_wrap_buffer(ggml_backend_buffer_type_t buft, ggml_backend_buffer_t host) {
    if (!host) {
        return nullptr;
    }
    try {
        auto ctx = std::make_unique<ryzenai_buffer_context>(host);
        auto result = ggml_backend_buffer_init(buft, ryzenai_buffer_i, ctx.get(), ggml_backend_buffer_get_size(host));
        ctx.release();
        return result;
    } catch (const std::bad_alloc &) {
        ggml_backend_buffer_free(host);
        return nullptr;
    }
}

static ggml_backend_buffer_t ryzenai_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return ryzenai_wrap_buffer(buft, ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size));
}

static size_t ryzenai_buft_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

static bool ryzenai_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static ggml_backend_buffer_type_t ryzenai_device_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ ryzenai_buft_name,
            /* .alloc_buffer     = */ ryzenai_buft_alloc,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ ryzenai_buft_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ ryzenai_buft_is_host,
        },
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
    return &buft;
}

static bool ryzenai_valid_strides(const ggml_tensor * tensor) {
    const size_t limit = max_elements * sizeof(float);
    size_t span = ggml_row_size(tensor->type, tensor->ne[0]);
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] > 1) {
            if (!tensor->nb[d] || tensor->nb[d] > (limit - span) / size_t(tensor->ne[d] - 1)) {
                return false;
            }
            span += size_t(tensor->ne[d] - 1) * tensor->nb[d];
        }
    }
    return true;
}

static bool ryzenai_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            break;
        default:
            return false;
    }
    const auto * w = op->src[0];
    const auto * x = op->src[1];
    if (!w || !x || op->type != GGML_TYPE_F32 || w->type < 0 || w->type >= GGML_TYPE_COUNT ||
        (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) ||
        (w->type != GGML_TYPE_F32 && !ggml_get_type_traits(w->type)->to_float) ||
        ggml_get_op_params_i32(op, 0) != GGML_PREC_DEFAULT || ggml_get_op_params_i32(op, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(op, 3) != GGML_PREC_DEFAULT ||
        w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] < tile_k || w->ne[1] < tile_n ||
        w->ne[0] > max_elements - tile_k || w->ne[1] > max_elements - tile_n || x->ne[0] != w->ne[0]) {
        return false;
    }
    int64_t rows = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (x->ne[d] <= 0 || x->ne[d] > (max_elements - tile_m) / rows) {
            return false;
        }
        rows *= x->ne[d];
    }
    const int64_t k = ryzenai_pad(w->ne[0], tile_k);
    const int64_t n = ryzenai_pad(w->ne[1], tile_n);
    const int64_t m = ryzenai_pad(rows, tile_m);
    if (k > max_elements / n || m > max_elements / k || m > max_elements / n ||
        op->ne[0] != w->ne[1] || op->ne[1] != x->ne[1] || op->ne[2] != x->ne[2] || op->ne[3] != x->ne[3]) {
        return false;
    }
    const ggml_tensor * base = w;
    while (base->view_src) {
        base = base->view_src;
    }
    return base->op == GGML_OP_NONE && ggml_is_contiguous_rows(w) && ggml_is_contiguous_rows(x) && ggml_is_contiguous(op) &&
           ryzenai_valid_strides(w) && ryzenai_valid_strides(x);
}

struct ryzenai_backend_context {
    std::shared_ptr<ryzenai_runtime> runtime;
    uint64_t n_matmuls = 0;
    double elapsed_ms = 0.0;
};

static const char * ryzenai_backend_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "RyzenAI";
}

static void ryzenai_backend_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ryzenai_backend_context *>(backend->context);
    if (ctx->n_matmuls > 0) {
        GGML_LOG_INFO("RyzenAI: %llu NPU matmuls, %.2f ms (packing, transfer and execution)\n", (unsigned long long) ctx->n_matmuls, ctx->elapsed_ms);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ryzenai_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<ryzenai_backend_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->runtime->mutex);
    try {
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * node = graph->nodes[i];
            if (ggml_is_empty(node) || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_is_view(node) || node->op == GGML_OP_NONE) {
                continue;
            }
            if (!ryzenai_device_supports_op(backend->device, node)) {
                return GGML_STATUS_FAILED;
            }
            const auto * w = node->src[0];
            if (!w->buffer || ggml_backend_buffer_get_type(w->buffer) != ryzenai_device_buffer_type(backend->device) ||
                !node->src[1]->buffer || !ggml_backend_buffer_is_host(node->src[1]->buffer) ||
                !node->buffer || !ggml_backend_buffer_is_host(node->buffer)) {
                return GGML_STATUS_FAILED;
            }
            const auto start = std::chrono::steady_clock::now();
            if (ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                auto * buf = ryzenai_buffer_context_get(w->buffer);
                ryzenai_buffer_context::weight_key key { reinterpret_cast<uintptr_t>(w->data), w->type, w->ne[0], w->ne[1], w->nb[1] };
                auto it = buf->weights.find(key);
                if (it == buf->weights.end()) {
                    buf->runtime = ctx->runtime;
                    it = buf->weights.emplace(key, ctx->runtime->pack_weight(w)).first;
                }
                ctx->runtime->compute(node, it->second);
            } else {
                auto packed = ctx->runtime->pack_weight(w);
                ctx->runtime->compute(node, packed);
            }
            ++ctx->n_matmuls;
            ctx->elapsed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const char * trace = std::getenv("GGML_RYZENAI_TRACE");
            if (trace && std::strcmp(trace, "1") == 0) {
                GGML_LOG_INFO("RyzenAI: NPU MUL_MAT %s K=%lld N=%lld M=%lld\n", w->name, (long long) w->ne[0], (long long) w->ne[1], (long long) (node->src[1]->ne[1] * node->src[1]->ne[2] * node->src[1]->ne[3]));
            }
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("RyzenAI: %s\n", e.what());
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("RyzenAI: %s\n", e.what());
        return GGML_STATUS_FAILED;
    }
}

static ggml_backend_t ryzenai_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    try {
        auto ctx = std::make_unique<ryzenai_backend_context>();
        ctx->runtime = ryzenai_get_runtime();
        static ggml_guid guid = { 0x87, 0x63, 0x91, 0x2a, 0xcd, 0x1d, 0x49, 0xa8, 0x8c, 0xa1, 0x13, 0x34, 0x18, 0xb4, 0x5c, 0x20 };
        ggml_backend_i iface = {};
        iface.get_name = ryzenai_backend_name;
        iface.free = ryzenai_backend_free;
        iface.graph_compute = ryzenai_backend_graph_compute;
        auto * backend = new ggml_backend { &guid, iface, dev, ctx.get() };
        ctx.release();
        GGML_LOG_INFO("RyzenAI: XDNA2 BF16 GEMM via FastFlowLM/XRT, unsupported operations use CPU\n");
        return backend;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("RyzenAI: initialization failed: %s; check xrt-smi examine, /dev/accel permissions, memlock and matching GEMM assets\n", e.what());
        return nullptr;
    }
}

static const char * ryzenai_device_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "RyzenAI";
}

static const char * ryzenai_device_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "AMD XDNA2 NPU (FastFlowLM BF16 GEMM, experimental)";
}

static void ryzenai_device_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    struct sysinfo info = {};
    *free = *total = 0;
    if (sysinfo(&info) == 0) {
        *free = size_t(info.freeram) * info.mem_unit;
        *total = size_t(info.totalram) * info.mem_unit;
    }
}

static enum ggml_backend_dev_type ryzenai_device_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // Use the layer offload device path; unsupported operators stay on CPU.
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ryzenai_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = {};
    props->name = ryzenai_device_name(dev);
    props->description = ryzenai_device_description(dev);
    props->type = ryzenai_device_type(dev);
    ryzenai_device_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.buffer_from_host_ptr = true;
    props->caps.mmap_support = true;
}

static ggml_backend_buffer_t ryzenai_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(max_tensor_size);
    return ryzenai_wrap_buffer(ryzenai_device_buffer_type(dev), ggml_backend_cpu_buffer_from_ptr(ptr, size));
}

static bool ryzenai_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ryzenai_device_buffer_type(dev);
}

static const char * ryzenai_reg_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "RyzenAI";
}

static uint64_t ryzenai_backend_matmul_count(ggml_backend_t backend) {
    return static_cast<ryzenai_backend_context *>(backend->context)->n_matmuls;
}

static void * ryzenai_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_ryzenai_get_n_matmuls") == 0) {
        return reinterpret_cast<void *>(ryzenai_backend_matmul_count);
    }
    return nullptr;
}

static size_t ryzenai_reg_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    try {
        if (!ryzenai_has_xdna2() || !std::filesystem::is_regular_file(ryzenai_xclbin_path())) {
            return 0;
        }
        xrt::device device(0);
        return 1;
    } catch (const std::exception & e) {
        GGML_LOG_DEBUG("RyzenAI: device unavailable: %s\n", e.what());
        return 0;
    }
}

static ggml_backend_dev_t ryzenai_reg_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = [reg] {
        ggml_backend_device result = {};
        result.reg = reg;
        result.iface.get_name = ryzenai_device_name;
        result.iface.get_description = ryzenai_device_description;
        result.iface.get_memory = ryzenai_device_memory;
        result.iface.get_type = ryzenai_device_type;
        result.iface.get_props = ryzenai_device_props;
        result.iface.init_backend = ryzenai_device_init;
        result.iface.get_buffer_type = ryzenai_device_buffer_type;
        result.iface.buffer_from_host_ptr = ryzenai_device_buffer_from_ptr;
        result.iface.supports_op = ryzenai_device_supports_op;
        result.iface.supports_buft = ryzenai_device_supports_buft;
        return result;
    }();
    return &device;
}

ggml_backend_reg_t ggml_backend_ryzenai_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface = */ {
            /* .get_name         = */ ryzenai_reg_name,
            /* .get_device_count = */ ryzenai_reg_device_count,
            /* .get_device       = */ ryzenai_reg_device,
            /* .get_proc_address = */ ryzenai_reg_get_proc_address,
        },
        /* .context = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ryzenai_reg)
