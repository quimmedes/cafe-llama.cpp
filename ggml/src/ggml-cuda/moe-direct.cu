#include "moe-direct.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "ggml-cpu.h"

#include <algorithm>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <atomic>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <chrono>
#include <deque>
#include <thread>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// experts per context over all registered tensors
static constexpr int64_t md_table_cap   = 1 << 21;
static constexpr size_t  md_slot_align  = 256;

struct md_config {
    bool    enabled;
    int64_t reserve_mb;    // < 0: derived from the device size
    int64_t budget_mb;     // 0: no cap
    int     mmv_max;
    int64_t fill_mb;       // bytes copied host -> VRAM per graph
    int     evict_max;     // replacements per graph
    int     stats;         // log period in graphs, 0 = only at teardown
    int     decay;         // graphs per halving of the expert scores
    bool    profile;       // keep the expert heat on disk between runs
    float   cpu_frac;      // share of the VRAM misses that the CPU computes
    int     cpu_threads;
};

static const md_config & md_cfg() {
    static const md_config cfg = [] {
        auto env_i = [](const char * name, int64_t def) {
            const char * v = getenv(name);
            return v && *v ? (int64_t) strtoll(v, nullptr, 10) : def;
        };
        md_config c;
        c.enabled    = env_i("GGML_CUDA_MOE_DIRECT", 1) != 0;
        c.reserve_mb = env_i("GGML_CUDA_MOE_DIRECT_RESERVE_MB", -1);
        c.budget_mb  = env_i("GGML_CUDA_MOE_DIRECT_BUDGET_MB", 0);
        c.mmv_max    = (int) std::max<int64_t>(1, env_i("GGML_CUDA_MOE_DIRECT_MMV_MAX", 64));
        c.fill_mb    = std::max<int64_t>(1, env_i("GGML_CUDA_MOE_DIRECT_FILL_MB", 256));
        c.evict_max  = (int) std::max<int64_t>(0, env_i("GGML_CUDA_MOE_DIRECT_EVICT_MAX", 32));
        c.stats      = (int) std::max<int64_t>(0, env_i("GGML_CUDA_MOE_DIRECT_STATS", 0));
        c.decay      = (int) std::max<int64_t>(1, env_i("GGML_CUDA_MOE_DIRECT_DECAY", 128));
        c.profile    = env_i("GGML_CUDA_MOE_DIRECT_PROFILE", 1) != 0;
        {
            const char * v = getenv("GGML_CUDA_MOE_DIRECT_CPU_FRAC");
            c.cpu_frac = v && *v ? std::min(1.0f, std::max(0.0f, (float) atof(v))) : 0.5f;
        }
        // physical cores minus two (assumes SMT): the misses are bound by RAM bandwidth, not by cores
        c.cpu_threads = (int) std::max<int64_t>(1, env_i("GGML_CUDA_MOE_DIRECT_CPU_THREADS", std::max<int>(2, (int) std::thread::hardware_concurrency()/2 - 2)));
        return c;
    }();
    return cfg;
}

bool ggml_cuda_moe_direct_enabled() {
    return md_cfg().enabled;
}

int ggml_cuda_moe_direct_mmv_max_tokens() {
    return md_cfg().mmv_max;
}

bool ggml_cuda_moe_direct_type_supported(const ggml_tensor * src0) {
    return ggml_cuda_moe_direct_mmv_supported(src0->type) && src0->ne[3] == 1 && src0->nb[0] == ggml_type_size(src0->type) &&
        src0->nb[2] >= src0->ne[1]*src0->nb[1];
}

// VRAM used by all contexts, per device, for the budget cap
static std::atomic<size_t> g_md_bytes[GGML_CUDA_MAX_DEVICES];

struct md_entry {
    const char * host;
    size_t       nb02;
    size_t       expert_bytes;
    int64_t      n_expert;
    int64_t      off;
    int          cls;
    uint64_t     key;
};

struct md_class {
    size_t               slot_bytes;
    std::vector<char *>  chunks;
    std::vector<char *>  slot_ptr;
    std::vector<int64_t> owner;
    std::vector<int32_t> free_slots;
};

struct md_fill_item {
    int64_t g;
    int32_t slot;
};

struct md_fill {
    cudaEvent_t               ev;
    std::vector<md_fill_item> items;
};

struct md_cpu;

struct ggml_cuda_moe_direct_ctx {
    int device;

    cudaStream_t fill_stream = nullptr;
    cudaEvent_t  snap_ev     = nullptr;
    cudaEvent_t  evict_ev    = nullptr;

    uint64_t * table_dev   = nullptr;
    uint64_t * table_host  = nullptr;
    uint32_t * counts_dev  = nullptr;
    uint32_t * counts_snap = nullptr;

    bool    snap_pending = false;
    int64_t snap_n       = 0;
    int64_t n_total      = 0;

    std::vector<uint32_t> counts_last;
    std::vector<float>    score;
    std::vector<int32_t>  resident;
    std::vector<int32_t>  g_entry;
    std::vector<uint8_t>  in_flight;

    std::vector<md_entry>                   entries;
    std::unordered_map<const void *, int>   by_host;
    std::vector<md_class>                   classes;
    std::deque<md_fill>                     fills;
    std::vector<cudaEvent_t>                free_events;

    int64_t dirty_lo = INT64_MAX;
    int64_t dirty_hi = -1;

    size_t   bytes         = 0;
    size_t   reserve_bytes = 0;
    uint64_t grow_retry_at = 0;

    uint64_t steps        = 0;
    uint64_t decay_count  = 0;
    uint64_t n_filled     = 0;
    uint64_t n_evicted    = 0;
    uint64_t uses_hit     = 0;
    uint64_t uses_total   = 0;
    uint64_t last_hit     = 0;
    uint64_t last_total   = 0;
    int      n_profile_loaded = 0;
    md_cpu * cpu = nullptr;
    bool     cpu_tried = false;
    bool     profile_logged   = false;
};

static void md_mark_dirty(ggml_cuda_moe_direct_ctx & md, int64_t g) {
    md.dirty_lo = std::min(md.dirty_lo, g);
    md.dirty_hi = std::max(md.dirty_hi, g);
}

static uint64_t md_host_ptr(const ggml_cuda_moe_direct_ctx & md, int64_t g) {
    const md_entry & en = md.entries[md.g_entry[g]];
    return (uint64_t) (uintptr_t) (en.host + (g - en.off)*en.nb02);
}

static bool md_init(ggml_cuda_moe_direct_ctx & md, cudaStream_t stream) {
    int least = 0;
    int greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&md.fill_stream, cudaStreamNonBlocking, least));
    CUDA_CHECK(cudaEventCreateWithFlags(&md.snap_ev,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&md.evict_ev, cudaEventDisableTiming));

    CUDA_CHECK(cudaMalloc((void **) &md.table_dev,  md_table_cap*sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void **) &md.counts_dev, md_table_cap*sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost((void **) &md.table_host,  md_table_cap*sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost((void **) &md.counts_snap, md_table_cap*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(md.counts_dev, 0, md_table_cap*sizeof(uint32_t), stream));

    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (md_cfg().reserve_mb >= 0) {
        md.reserve_bytes = (size_t) md_cfg().reserve_mb << 20;
    } else {
        // same rule as the CPU-side expert cache: 6% of VRAM, 1-3 GiB, at most a quarter
        size_t r = (size_t) (total_mem*0.06) & ~((size_t(128) << 20) - 1);
        r = std::min(std::max(r, size_t(1024) << 20), size_t(3072) << 20);
        md.reserve_bytes = std::min(r, total_mem/4);
    }
    GGML_LOG_INFO("%s: device %d: MoE experts in pinned host memory run on the GPU, VRAM expert cache keeps %zu MiB free\n",
        __func__, md.device, md.reserve_bytes >> 20);
    return true;
}

static int md_class_for(ggml_cuda_moe_direct_ctx & md, size_t expert_bytes) {
    const size_t slot_bytes = GGML_PAD(expert_bytes, md_slot_align);
    for (size_t i = 0; i < md.classes.size(); ++i) {
        if (md.classes[i].slot_bytes == slot_bytes) {
            return (int) i;
        }
    }
    md.classes.push_back({});
    md.classes.back().slot_bytes = slot_bytes;
    return (int) md.classes.size() - 1;
}

// The expert heat of each tensor is kept on disk between runs, so a new process fills the VRAM cache
// with the experts that were hot last time instead of starting cold.
static std::string md_profile_dir() {
    if (!md_cfg().profile) {
        return "";
    }
    const char * v = getenv("GGML_CUDA_MOE_DIRECT_PROFILE_DIR");
    if (v && *v) {
        return v;
    }
    if ((v = getenv("LLAMA_CACHE")) && *v) {
        return std::string(v) + "/moe-direct";
    }
    if ((v = getenv("XDG_CACHE_HOME")) && *v) {
        return std::string(v) + "/llama.cpp/moe-direct";
    }
    if ((v = getenv("HOME")) && *v) {
        return std::string(v) + "/.cache/llama.cpp/moe-direct";
    }
    if ((v = getenv("LOCALAPPDATA")) && *v) {
        return std::string(v) + "/llama.cpp/moe-direct";
    }
    return "";
}

// name, shape, type and the first bytes of the first and last expert: stable per model file, distinct across models
static uint64_t md_profile_key(const ggml_tensor * src0) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void * p, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ull;
        }
    };
    mix(src0->name, strlen(src0->name));
    mix(src0->ne, sizeof(src0->ne));
    mix(&src0->type, sizeof(src0->type));
    const size_t n = std::min<size_t>(256, src0->ne[1]*src0->nb[1]);
    mix(src0->data, n);
    mix((const char *) src0->data + (src0->ne[2] - 1)*src0->nb[2], n);
    return h;
}

static std::string md_profile_path(const std::string & dir, uint64_t key) {
    char name[32];
    snprintf(name, sizeof(name), "/%016" PRIx64 ".bin", key);
    return dir + name;
}

static void md_profile_load(ggml_cuda_moe_direct_ctx & md, const md_entry & en) {
    const std::string dir = md_profile_dir();
    if (dir.empty()) {
        return;
    }
    FILE * f = fopen(md_profile_path(dir, en.key).c_str(), "rb");
    if (!f) {
        return;
    }
    std::vector<float> s(en.n_expert);
    int64_t n = 0;
    if (fread(&n, sizeof(n), 1, f) == 1 && n == en.n_expert && fread(s.data(), sizeof(float), n, f) == (size_t) n) {
        for (int64_t e = 0; e < n; ++e) {
            if (std::isfinite(s[e]) && s[e] > 0.0f) {
                md.score[en.off + e] = s[e];
            }
        }
        md.n_profile_loaded++;
    }
    fclose(f);
}

static void md_profile_save(const ggml_cuda_moe_direct_ctx & md) {
    const std::string dir = md_profile_dir();
    if (dir.empty() || md.uses_total == 0) {
        return;
    }
    std::string cur;
    for (size_t i = 0; i < dir.size(); ++i) {
        cur += dir[i];
        if (dir[i] == '/' || i + 1 == dir.size()) {
#ifdef _WIN32
            _mkdir(cur.c_str());
#else
            mkdir(cur.c_str(), 0755);
#endif
        }
    }
    for (const md_entry & en : md.entries) {
        const std::string path = md_profile_path(dir, en.key);
        const std::string tmp  = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) {
            continue;
        }
        const int64_t n = en.n_expert;
        const bool ok = fwrite(&n, sizeof(n), 1, f) == 1 && fwrite(md.score.data() + en.off, sizeof(float), n, f) == (size_t) n;
        fclose(f);
        if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
            remove(tmp.c_str());
        }
    }
}

static void md_register(ggml_cuda_moe_direct_ctx & md, const ggml_tensor * src0) {
    if (md.by_host.count(src0->data)) {
        return;
    }
    const int64_t n_expert = src0->ne[2];
    if (md.n_total + n_expert > md_table_cap) {
        md.by_host[src0->data] = -1;
        return;
    }

    md_entry en;
    en.host         = (const char *) src0->data;
    en.nb02         = src0->nb[2];
    en.expert_bytes = src0->ne[1]*src0->nb[1];
    en.n_expert     = n_expert;
    en.off          = md.n_total;
    en.cls          = md_class_for(md, en.expert_bytes);

    const int idx = (int) md.entries.size();
    md.entries.push_back(en);
    md.by_host[src0->data] = idx;

    md.n_total += n_expert;
    md.counts_last.resize(md.n_total, 0);
    md.score.resize(md.n_total, 0.0f);
    md.resident.resize(md.n_total, -1);
    md.g_entry.resize(md.n_total, idx);
    md.in_flight.resize(md.n_total, 0);

    for (int64_t e = 0; e < n_expert; ++e) {
        md.table_host[en.off + e] = md_host_ptr(md, en.off + e);
    }
    md_mark_dirty(md, en.off);
    md_mark_dirty(md, en.off + n_expert - 1);

    md.entries[idx].key = md_profile_key(src0);
    md_profile_load(md, md.entries[idx]);
}

static bool md_grow(ggml_cuda_moe_direct_ctx & md, int cls) {
    if (md.steps < md.grow_retry_at) {
        return false;
    }
    md_class & c = md.classes[cls];
    const int64_t n_slots = std::max<int64_t>(16, std::min<int64_t>((256 << 20) / c.slot_bytes, 512));
    const size_t chunk = n_slots*c.slot_bytes;

    const int64_t budget_mb = md_cfg().budget_mb;
    if (budget_mb > 0 && g_md_bytes[md.device].load() + chunk > ((size_t) budget_mb << 20)) {
        md.grow_retry_at = UINT64_MAX;
        return false;
    }
    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (free_mem < md.reserve_bytes + chunk) {
        md.grow_retry_at = md.steps + 256;
        return false;
    }
    char * ptr = nullptr;
    if (cudaMalloc((void **) &ptr, chunk) != cudaSuccess) {
        (void) cudaGetLastError();
        md.grow_retry_at = md.steps + 256;
        return false;
    }
    c.chunks.push_back(ptr);
    for (int64_t i = 0; i < n_slots; ++i) {
        c.free_slots.push_back((int32_t) c.slot_ptr.size());
        c.slot_ptr.push_back(ptr + i*c.slot_bytes);
        c.owner.push_back(-1);
    }
    md.bytes += chunk;
    g_md_bytes[md.device] += chunk;
    return true;
}

static void md_log_stats(ggml_cuda_moe_direct_ctx & md, const char * when) {
    int64_t n_res = 0;
    for (int32_t s : md.resident) {
        n_res += s >= 0;
    }
    const uint64_t d_total = md.uses_total - md.last_total;
    const uint64_t d_hit   = md.uses_hit   - md.last_hit;
    md.last_total = md.uses_total;
    md.last_hit   = md.uses_hit;
    GGML_LOG_INFO("moe-direct %s: device %d: %" PRId64 "/%" PRId64 " experts in VRAM (%zu MiB), hit rate %.1f%% (recent %.1f%%), filled %" PRIu64 ", replaced %" PRIu64 "\n",
        when, md.device, n_res, md.n_total, md.bytes >> 20,
        md.uses_total ? 100.0*md.uses_hit/md.uses_total : 0.0,
        d_total ? 100.0*d_hit/d_total : 0.0, md.n_filled, md.n_evicted);
}

void ggml_cuda_moe_direct_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph,
        bool (*is_direct)(const ggml_tensor * src0)) {
    ggml_cuda_moe_direct_ctx * mdp = ctx.moe_direct;
    cudaStream_t stream = ctx.stream();

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID || !is_direct(node->src[0]) || !ggml_cuda_moe_direct_type_supported(node->src[0])) {
            continue;
        }
        if (!mdp) {
            mdp = new ggml_cuda_moe_direct_ctx;
            mdp->device = ctx.device;
            md_init(*mdp, stream);
            ctx.moe_direct = mdp;
        }
        md_register(*mdp, node->src[0]);
    }
    if (!mdp || mdp->n_total == 0) {
        return;
    }
    ggml_cuda_moe_direct_ctx & md = *mdp;
    md.steps++;

    if (md.n_profile_loaded > 0 && !md.profile_logged) {
        md.profile_logged = true;
        GGML_LOG_INFO("%s: device %d: expert heat of %d tensors loaded from %s, the VRAM cache fills with them first\n",
            __func__, md.device, md.n_profile_loaded, md_profile_dir().c_str());
    }

    // use counts of the graphs that finished since the last snapshot
    bool fresh = false;
    if (md.snap_pending) {
        const cudaError_t q = cudaEventQuery(md.snap_ev);
        if (q == cudaSuccess) {
            md.snap_pending = false;
            fresh = true;
            for (int64_t g = 0; g < md.snap_n; ++g) {
                const uint32_t d = md.counts_snap[g] - md.counts_last[g];
                if (d == 0) {
                    continue;
                }
                md.counts_last[g] = md.counts_snap[g];
                md.score[g] += (float) d;
                md.uses_total += d;
                if (md.resident[g] >= 0) {
                    md.uses_hit += d;
                }
            }
            if (++md.decay_count >= (uint64_t) md_cfg().decay) {
                md.decay_count = 0;
                for (float & s : md.score) {
                    s *= 0.5f;
                }
            }
        } else if (q != cudaErrorNotReady) {
            CUDA_CHECK(q);
        }
    }

    // publish finished fills
    while (!md.fills.empty()) {
        const cudaError_t q = cudaEventQuery(md.fills.front().ev);
        if (q == cudaErrorNotReady) {
            break;
        }
        CUDA_CHECK(q);
        md_fill & f = md.fills.front();
        for (const md_fill_item & it : f.items) {
            md_class & c = md.classes[md.entries[md.g_entry[it.g]].cls];
            md.resident[it.g]   = it.slot;
            md.in_flight[it.g]  = 0;
            c.owner[it.slot]    = it.g;
            md.table_host[it.g] = (uint64_t) (uintptr_t) c.slot_ptr[it.slot];
            md_mark_dirty(md, it.g);
            md.n_filled++;
        }
        md.free_events.push_back(f.ev);
        md.fills.pop_front();
    }

    // while VRAM can still grow, fill every graph; once full, look for replacements less often
    bool can_fill = md.steps >= md.grow_retry_at;
    for (const md_class & c : md.classes) {
        can_fill = can_fill || !c.free_slots.empty();
    }

    // plan new fills, hottest missing experts first
    md_fill fill;
    bool evicted = false;
    std::vector<int64_t> cands;
    if (fresh && md.fills.size() < 2 && (can_fill || md.steps % 16 == 0)) {
        for (int64_t g = 0; g < md.n_total; ++g) {
            if (md.resident[g] < 0 && !md.in_flight[g] && md.score[g] >= 1.0f) {
                cands.push_back(g);
            }
        }
        const size_t k = std::min<size_t>(cands.size(), 2048);
        std::partial_sort(cands.begin(), cands.begin() + k, cands.end(), [&](int64_t a, int64_t b) { return md.score[a] > md.score[b]; });
        cands.resize(k);
    }
    if (!cands.empty()) {
        // per class: residents by ascending score, built on first need
        std::vector<std::vector<int32_t>> victims(md.classes.size());
        std::vector<size_t>               victim_pos(md.classes.size(), 0);
        std::vector<uint8_t>              victims_built(md.classes.size(), 0);

        size_t fill_bytes = 0;
        int n_evict = 0;
        // fill faster while VRAM is still free, e.g. right after start with a loaded expert profile
        const size_t fill_cap = ((size_t) md_cfg().fill_mb << 20) * (can_fill ? 8 : 1);

        for (int64_t g : cands) {
            const md_entry & en = md.entries[md.g_entry[g]];
            md_class & c = md.classes[en.cls];
            if (fill_bytes + en.expert_bytes > fill_cap) {
                break;
            }
            int32_t slot = -1;
            if (c.free_slots.empty()) {
                md_grow(md, en.cls);
            }
            if (!c.free_slots.empty()) {
                slot = c.free_slots.back();
                c.free_slots.pop_back();
            } else if (n_evict < md_cfg().evict_max) {
                if (!victims_built[en.cls]) {
                    victims_built[en.cls] = 1;
                    auto & v = victims[en.cls];
                    for (int32_t s = 0; s < (int32_t) c.owner.size(); ++s) {
                        if (c.owner[s] >= 0) {
                            v.push_back(s);
                        }
                    }
                    std::sort(v.begin(), v.end(), [&](int32_t a, int32_t b) { return md.score[c.owner[a]] < md.score[c.owner[b]]; });
                }
                auto & v = victims[en.cls];
                size_t & pos = victim_pos[en.cls];
                if (pos < v.size()) {
                    const int32_t s = v[pos];
                    const int64_t old = c.owner[s];
                    // hysteresis against ping-pong between two experts of similar heat
                    if (md.score[g] > 2.0f*md.score[old] + 4.0f) {
                        pos++;
                        md.resident[old]   = -1;
                        c.owner[s]         = -1;
                        md.table_host[old] = md_host_ptr(md, old);
                        md_mark_dirty(md, old);
                        md.n_evicted++;
                        n_evict++;
                        evicted = true;
                        slot = s;
                    }
                }
            }
            if (slot < 0) {
                continue;
            }
            md.in_flight[g] = 1;
            fill.items.push_back({ g, slot });
            fill_bytes += en.expert_bytes;
        }
    }

    if (md.dirty_hi >= 0) {
        CUDA_CHECK(cudaMemcpyAsync(md.table_dev + md.dirty_lo, md.table_host + md.dirty_lo,
            (md.dirty_hi - md.dirty_lo + 1)*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        md.dirty_lo = INT64_MAX;
        md.dirty_hi = -1;
    }

    if (!fill.items.empty()) {
        if (evicted) {
            // the evicted slots are rewritten only after every kernel that could still read them
            CUDA_CHECK(cudaEventRecord(md.evict_ev, stream));
            CUDA_CHECK(cudaStreamWaitEvent(md.fill_stream, md.evict_ev, 0));
        }
        for (const md_fill_item & it : fill.items) {
            const md_entry & en = md.entries[md.g_entry[it.g]];
            md_class & c = md.classes[en.cls];
            CUDA_CHECK(cudaMemcpyAsync(c.slot_ptr[it.slot], en.host + (it.g - en.off)*en.nb02, en.expert_bytes,
                cudaMemcpyHostToDevice, md.fill_stream));
        }
        if (md.free_events.empty()) {
            cudaEvent_t ev;
            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            md.free_events.push_back(ev);
        }
        fill.ev = md.free_events.back();
        md.free_events.pop_back();
        CUDA_CHECK(cudaEventRecord(fill.ev, md.fill_stream));
        md.fills.push_back(std::move(fill));
    }

    if (!md.snap_pending) {
        CUDA_CHECK(cudaMemcpyAsync(md.counts_snap, md.counts_dev, md.n_total*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(md.snap_ev, stream));
        md.snap_pending = true;
        md.snap_n = md.n_total;
    }

    if (md.steps % 1024 == 0) {
        md_profile_save(md);
    }

    if (md_cfg().stats > 0 && md.steps % md_cfg().stats == 0) {
        md_log_stats(md, "stats");
    }
}

static __global__ void md_mark_kernel(const int32_t * __restrict__ ids, uint32_t n_used, uint32_t n_routes, uint32_t ids_stride,
        uint8_t * __restrict__ used, uint32_t * __restrict__ counts) {
    const uint32_t r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n_routes) {
        return;
    }
    const uint32_t tok = r / n_used;
    const int32_t  e   = ids[tok*ids_stride + (r - tok*n_used)];
    used[e] = 1;
    atomicAdd(&counts[e], 1u);
}

template <typename T>
static __global__ void md_gather_kernel(const uint64_t * __restrict__ table, const uint8_t * __restrict__ used,
        char * __restrict__ dst, size_t nb02, size_t n_elems) {
    const int64_t e = blockIdx.y;
    if (!used[e]) {
        // the stock kernels read up to one padded row past a used expert, into this one: keep it finite
        T * out = (T *) (dst + e*nb02);
        const size_t n_head = min(n_elems, (size_t) (4096/sizeof(T)));
        for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n_head; i += (size_t) gridDim.x*blockDim.x) {
            out[i] = T{};
        }
        return;
    }
    const T * src = (const T *) table[e];
    T * out = (T *) (dst + e*nb02);
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n_elems; i += (size_t) gridDim.x*blockDim.x) {
        out[i] = src[i];
    }
}


// ---------------------------------------------------------------------------------------------
// CPU share of the misses. The GPU writes the routed activations of the routes it hands to the CPU
// into mapped host memory and bumps a sequence number; host workers poll it, compute those routes
// from the pinned weights and post the results. A later kernel waits for them and scatters them into dst.
// Everything stays inside the captured CUDA graph: no driver call and no stream sync between the halves.

static constexpr int     md_cpu_routes = 128;
static constexpr int64_t md_cpu_act    = 1 << 20; // floats
static constexpr int64_t md_cpu_out    = 1 << 21; // floats
static constexpr int     md_cpu_rows   = 32;      // rows per work item

struct md_mailbox {
    alignas(64) volatile uint32_t seq;  // written by the GPU, last
    alignas(64) volatile uint32_t done; // written by the CPU
    alignas(64) uint32_t n_cpu;
    int32_t  type;
    uint32_t ncols;
    uint32_t nrows;
    uint32_t has_gate;
    uint64_t host_up;
    uint64_t host_gate;
    uint64_t nb02;
    uint64_t nb01;
    int32_t  expert[md_cpu_routes];
    float    act[md_cpu_act];
    float    out[md_cpu_out];
};

// device side state, shared by the publish, matvec and merge kernels of one op
struct md_cpu_dev {
    uint32_t seq;
    uint32_t n_cpu;
    float    acc;
    int32_t  route[md_cpu_routes];
    uint8_t  skip[md_cpu_routes];
};

typedef const struct ggml_type_traits_cpu * (*md_traits_fn)(enum ggml_type);

struct md_cpu {
    md_mailbox * mb     = nullptr; // host view
    md_mailbox * mb_dev = nullptr; // device view of the same memory
    md_cpu_dev * dev    = nullptr;
    md_traits_fn traits = nullptr;

    std::vector<std::thread> threads;
    std::atomic<bool>     stop{false};
    std::atomic<uint32_t> job_seq{0};
    std::atomic<uint64_t> next{0};   // (seq << 32) | next work item
    std::atomic<uint32_t> n_done{0}; // finished work items of the current job

    // copy of the current job, filled by the leader before job_seq is published
    uint32_t seq = 0;
    uint32_t n_items = 0;
    uint32_t n_cpu = 0;
    ggml_type type;
    uint32_t ncols, nrows, has_gate;
    const char * host_up;
    const char * host_gate;
    size_t nb02, nb01;
    int32_t expert[md_cpu_routes];
};

static inline void md_cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

// spin while work is frequent, then poll slowly so an idle server does not burn a core
struct md_backoff {
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    void hit() { last = std::chrono::steady_clock::now(); }
    void wait() {
        if (std::chrono::steady_clock::now() - last < std::chrono::milliseconds(20)) {
            md_cpu_pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
};

static void md_cpu_worker(md_cpu * c, int id) {
    md_backoff bo;
    uint32_t seen = 0;
    std::vector<char> q;
    int32_t q_route = -1;
    uint32_t q_seq = 0;
    while (!c->stop.load(std::memory_order_relaxed)) {
        if (id == 0) {
            const uint32_t s = c->mb->seq;
            if (s == seen) {
                bo.wait();
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            md_mailbox * mb = c->mb;
            c->seq      = s;
            c->n_cpu    = mb->n_cpu;
            c->type     = (ggml_type) mb->type;
            c->ncols    = mb->ncols;
            c->nrows    = mb->nrows;
            c->has_gate = mb->has_gate;
            c->host_up  = (const char *) (uintptr_t) mb->host_up;
            c->host_gate= (const char *) (uintptr_t) mb->host_gate;
            c->nb02     = mb->nb02;
            c->nb01     = mb->nb01;
            memcpy(c->expert, mb->expert, c->n_cpu*sizeof(int32_t));
            c->n_items  = c->n_cpu * ((c->nrows + md_cpu_rows - 1) / md_cpu_rows);
            c->n_done.store(0, std::memory_order_relaxed);
            c->next.store((uint64_t) s << 32, std::memory_order_relaxed);
            c->job_seq.store(s, std::memory_order_release);
            if (c->n_items == 0) {
                c->mb->done = s;
            }
        } else {
            const uint32_t s = c->job_seq.load(std::memory_order_acquire);
            if (s == seen) {
                bo.wait();
                continue;
            }
        }
        seen = c->job_seq.load(std::memory_order_acquire);
        bo.hit();

        const struct ggml_type_traits_cpu * tr = c->traits(c->type);
        const struct ggml_type_traits_cpu * tq = c->traits(tr->vec_dot_type);
        const size_t q_size = ggml_row_size(tr->vec_dot_type, c->ncols);
        const uint32_t items_per_route = (c->nrows + md_cpu_rows - 1) / md_cpu_rows;
        if (q.size() < q_size) {
            q.resize(q_size);
        }
        for (;;) {
            uint64_t v = c->next.load(std::memory_order_relaxed);
            uint32_t item;
            for (;;) {
                if ((uint32_t) (v >> 32) != seen || (uint32_t) v >= c->n_items) {
                    item = UINT32_MAX;
                    break;
                }
                if (c->next.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) {
                    item = (uint32_t) v;
                    break;
                }
            }
            if (item == UINT32_MAX) {
                break;
            }
            const int32_t j = item / items_per_route;
            const uint32_t r0 = (item % items_per_route) * md_cpu_rows;
            const uint32_t r1 = std::min(c->nrows, r0 + md_cpu_rows);
            if (q_route != j || q_seq != seen) {
                tq->from_float(c->mb->act + (size_t) j*c->ncols, q.data(), c->ncols);
                q_route = j;
                q_seq = seen;
            }
            const char * wu = c->host_up + (size_t) c->expert[j]*c->nb02;
            const char * wg = c->has_gate ? c->host_gate + (size_t) c->expert[j]*c->nb02 : nullptr;
            float * out = c->mb->out + (size_t) j*c->nrows;
            for (uint32_t r = r0; r < r1; ++r) {
                float u = 0.0f;
                tr->vec_dot(c->ncols, &u, 0, wu + r*c->nb01, 0, q.data(), 0, 1);
                if (wg) {
                    float g = 0.0f;
                    tr->vec_dot(c->ncols, &g, 0, wg + r*c->nb01, 0, q.data(), 0, 1);
                    u *= g / (1.0f + expf(-g));
                }
                out[r] = u;
            }
            if (c->n_done.fetch_add(1, std::memory_order_acq_rel) + 1 == c->n_items) {
                std::atomic_thread_fence(std::memory_order_release);
                c->mb->done = seen;
            }
        }
    }
}

static md_cpu * md_cpu_create() {
    if (md_cfg().cpu_frac <= 0.0f) {
        return nullptr;
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return nullptr;
    }
    auto fn = (md_traits_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_get_type_traits_cpu");
    if (!fn) {
        return nullptr;
    }
    md_cpu * c = new md_cpu;
    c->traits = fn;
    CUDA_CHECK(cudaHostAlloc((void **) &c->mb, sizeof(md_mailbox), cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &c->mb_dev, c->mb, 0));
    c->mb->seq  = 0;
    c->mb->done = 0;
    CUDA_CHECK(cudaMalloc((void **) &c->dev, sizeof(md_cpu_dev)));
    CUDA_CHECK(cudaMemset(c->dev, 0, sizeof(md_cpu_dev)));
    for (int i = 0; i < md_cfg().cpu_threads; ++i) {
        c->threads.emplace_back(md_cpu_worker, c, i);
    }
    GGML_LOG_INFO("%s: %d CPU threads compute %.0f%% of the experts missing from VRAM\n", __func__,
        md_cfg().cpu_threads, 100.0f*md_cfg().cpu_frac);
    return c;
}

static void md_cpu_free(md_cpu * c) {
    if (!c) {
        return;
    }
    c->stop = true;
    for (auto & t : c->threads) {
        t.join();
    }
    CUDA_CHECK(cudaFree(c->dev));
    CUDA_CHECK(cudaFreeHost(c->mb));
    delete c;
}

// picks the CPU routes among the misses and publishes their activations
static __global__ void md_cpu_publish_kernel(
        md_mailbox * __restrict__ mb, md_cpu_dev * __restrict__ st,
        const uint64_t * __restrict__ table, const uint64_t * __restrict__ gate_table,
        uint64_t host_up, uint64_t host_gate, uint64_t nb02, uint64_t nb01,
        const int32_t * __restrict__ ids, uint32_t n_used, uint32_t n_routes, uint32_t ids_stride,
        const float * __restrict__ src1, uint32_t ne11, uint64_t s11, uint64_t s12,
        int32_t type, uint32_t ncols, uint32_t nrows, float frac) {
    __shared__ uint32_t n_cpu;
    __shared__ uint32_t seq;
    if (threadIdx.x == 0) {
        float acc = st->acc;
        uint32_t n = 0;
        for (uint32_t r = 0; r < n_routes; ++r) {
            const uint32_t tok = r / n_used;
            const int32_t  e   = ids[tok*ids_stride + (r - tok*n_used)];
            const bool miss = table[e] == host_up + (uint64_t) e*nb02 ||
                (gate_table && gate_table[e] == host_gate + (uint64_t) e*nb02);
            uint8_t sk = 0;
            if (miss) {
                acc += frac;
                if (acc >= 1.0f) {
                    acc -= 1.0f;
                    sk = 1;
                    st->route[n] = r;
                    mb->expert[n] = e;
                    n++;
                }
            }
            st->skip[r] = sk;
        }
        st->acc   = acc;
        st->n_cpu = n;
        n_cpu = n;
        seq = st->seq + (n > 0);
        st->seq = seq;
        if (n > 0) {
            mb->n_cpu     = n;
            mb->type      = type;
            mb->ncols     = ncols;
            mb->nrows     = nrows;
            mb->has_gate  = gate_table != nullptr;
            mb->host_up   = host_up;
            mb->host_gate = host_gate;
            mb->nb02      = nb02;
            mb->nb01      = nb01;
        }
    }
    __syncthreads();
    if (n_cpu == 0) {
        return;
    }
    for (uint32_t j = 0; j < n_cpu; ++j) {
        const uint32_t r   = st->route[j];
        const uint32_t tok = r / n_used;
        const uint32_t k   = r - tok*n_used;
        const float * x = src1 + (k % ne11)*s11 + tok*s12;
        for (uint32_t i = threadIdx.x; i < ncols; i += blockDim.x) {
            mb->act[(size_t) j*ncols + i] = x[i];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        mb->seq = seq;
    }
}

static __global__ void md_cpu_merge_kernel(const md_mailbox * __restrict__ mb, const md_cpu_dev * __restrict__ st,
        float * __restrict__ dst, uint32_t n_used, uint32_t nrows, uint32_t s_channel, uint32_t s_col) {
    const uint32_t j = blockIdx.y;
    if (j >= st->n_cpu) {
        return;
    }
    if (threadIdx.x == 0) {
        const uint32_t seq = st->seq;
        while ((int32_t) (mb->done - seq) < 0) {
#if defined(GGML_USE_HIP) || __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(256);
#endif
        }
    }
    __syncthreads();
    const uint32_t row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const uint32_t r   = st->route[j];
    const uint32_t tok = r / n_used;
    const uint32_t k   = r - tok*n_used;
    const volatile float * out = mb->out;
    dst[k*s_channel + tok*s_col + row] = out[(size_t) j*nrows + row];
}

bool ggml_cuda_moe_direct_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * gate_src0, ggml_tensor * out,
        void (*mul_mat_id)(ggml_backend_cuda_context & ctx, ggml_tensor * dst)) {
    ggml_cuda_moe_direct_ctx * mdp = ctx.moe_direct;
    if (!mdp) {
        return false;
    }
    ggml_cuda_moe_direct_ctx & md = *mdp;

    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const ggml_tensor * ids  = mm->src[2];

    auto it = md.by_host.find(src0->data);
    if (it == md.by_host.end() || it->second < 0 || src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    const md_entry & en = md.entries[it->second];

    const md_entry * gen = nullptr;
    if (gate_src0) {
        auto git = md.by_host.find(gate_src0->data);
        if (git == md.by_host.end() || git->second < 0 || gate_src0->type != src0->type || !ggml_are_same_stride(gate_src0, src0)) {
            return false;
        }
        gen = &md.entries[git->second];
    }

    cudaStream_t stream = ctx.stream();

    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];
    const int64_t n_routes = n_used*n_tok;
    const uint32_t ids_stride = ids->nb[1] / sizeof(int32_t);

    if (n_tok <= md_cfg().mmv_max) {
        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t ne12 = src1->ne[2];
        const int64_t ne13 = src1->ne[3];
        const int64_t ts1  = ggml_type_size(src1->type);
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12*ne11*ne10_padded*sizeof(block_q8_1)/QK8_1);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), src0->type, ne10,
            src1->nb[1]/ts1, src1->nb[2]/ts1, src1->nb[3]/ts1, ne10_padded, ne11, ne12, ne13, stream);

        const int64_t s11 = ne10_padded/QK8_1;

        ggml_cuda_moe_direct_mmv_args a;
        a.table              = md.table_dev + en.off;
        a.gate_table         = gen ? md.table_dev + gen->off : nullptr;
        a.vy                 = src1_q8_1.get();
        a.ids                = (const int32_t *) ids->data;
        a.dst                = (float *) out->data;
        a.counts             = md.counts_dev + en.off;
        a.gate_counts        = gen ? md.counts_dev + gen->off : nullptr;
        a.ncols_x            = (uint32_t) src0->ne[0];
        a.nrows_x            = (uint32_t) src0->ne[1];
        a.stride_row_x       = (uint32_t) (src0->nb[1] / ggml_type_size(src0->type));
        a.n_used             = (uint32_t) n_used;
        a.n_routes           = (uint32_t) n_routes;
        a.ids_stride         = ids_stride;
        a.nchannels_y        = (uint32_t) ne11;
        a.stride_channel_y   = (uint32_t) s11;
        a.stride_col_y       = (uint32_t) (ne11*s11);
        a.stride_channel_dst = (uint32_t) (out->nb[1] / sizeof(float));
        a.stride_col_dst     = (uint32_t) (out->nb[2] / sizeof(float));
        a.skip               = nullptr;

        if (!md.cpu_tried) {
            md.cpu_tried = true;
            md.cpu = md_cpu_create();
        }
        bool cpu_split = md.cpu && n_routes <= md_cpu_routes &&
            n_routes*src0->ne[0] <= md_cpu_act && n_routes*src0->ne[1] <= md_cpu_out;
        if (cpu_split) {
            const struct ggml_type_traits_cpu * tr = md.cpu->traits(src0->type);
            cpu_split = tr->vec_dot && md.cpu->traits(tr->vec_dot_type)->from_float &&
                src0->ne[0] % ggml_blck_size(tr->vec_dot_type) == 0;
        }
        if (cpu_split) {
            md_cpu_publish_kernel<<<1, 256, 0, stream>>>(md.cpu->mb_dev, md.cpu->dev,
                a.table, a.gate_table,
                (uint64_t) (uintptr_t) en.host, gen ? (uint64_t) (uintptr_t) gen->host : 0, en.nb02, src0->nb[1],
                a.ids, a.n_used, a.n_routes, ids_stride,
                (const float *) src1->data, (uint32_t) ne11, src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float),
                (int32_t) src0->type, a.ncols_x, a.nrows_x, md_cfg().cpu_frac);
            a.skip = (const uint8_t *) md.cpu->dev + offsetof(md_cpu_dev, skip);
        }
        ggml_cuda_moe_direct_mmv(src0->type, a, stream);
        if (cpu_split) {
            const dim3 grid((a.nrows_x + 255)/256, a.n_routes);
            md_cpu_merge_kernel<<<grid, 256, 0, stream>>>(md.cpu->mb_dev, md.cpu->dev, a.dst, a.n_used, a.nrows_x,
                a.stride_channel_dst, a.stride_col_dst);
        }
        return true;
    }

    if (gen) {
        return false;
    }

    // large batch: build a dense copy of the routed experts (VRAM slots for hits, PCIe for misses), then run the stock kernels
    ggml_cuda_pool_alloc<uint8_t> used(ctx.pool(), en.n_expert);
    CUDA_CHECK(cudaMemsetAsync(used.get(), 0, en.n_expert, stream));
    md_mark_kernel<<<(n_routes + 255)/256, 256, 0, stream>>>((const int32_t *) ids->data, (uint32_t) n_used, (uint32_t) n_routes, ids_stride,
        used.get(), md.counts_dev + en.off);

    ggml_tensor tmp0 = *src0;
    tmp0.buffer = mm->buffer;
    const size_t nbytes = ggml_nbytes(src0);
    const size_t alloc  = std::max(nbytes, ggml_backend_buffer_get_alloc_size(mm->buffer, &tmp0));
    ggml_cuda_pool_alloc<char> dense(ctx.pool(), alloc);
    tmp0.data = dense.get();
    if (alloc > nbytes) {
        CUDA_CHECK(cudaMemsetAsync(dense.get() + nbytes, 0, alloc - nbytes, stream));
    }

    const dim3 grid(64, (unsigned) en.n_expert);
    if (en.expert_bytes % 16 == 0 && en.nb02 % 16 == 0 && (uintptr_t) en.host % 16 == 0) {
        md_gather_kernel<int4><<<grid, 256, 0, stream>>>(md.table_dev + en.off, used.get(), dense.get(), en.nb02, en.expert_bytes/16);
    } else if (en.expert_bytes % 4 == 0 && en.nb02 % 4 == 0 && (uintptr_t) en.host % 4 == 0) {
        md_gather_kernel<int><<<grid, 256, 0, stream>>>(md.table_dev + en.off, used.get(), dense.get(), en.nb02, en.expert_bytes/4);
    } else {
        md_gather_kernel<char><<<grid, 256, 0, stream>>>(md.table_dev + en.off, used.get(), dense.get(), en.nb02, en.expert_bytes);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor tmp_mm = *mm;
    tmp_mm.src[0] = &tmp0;
    mul_mat_id(ctx, &tmp_mm);

    return true;
}

void ggml_cuda_moe_direct_free(ggml_cuda_moe_direct_ctx * md) {
    if (!md) {
        return;
    }
    ggml_cuda_set_device(md->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    md_cpu_free(md->cpu);
    md_log_stats(*md, "final");
    md_profile_save(*md);
    for (md_class & c : md->classes) {
        for (char * p : c.chunks) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    g_md_bytes[md->device] -= md->bytes;
    for (md_fill & f : md->fills) {
        CUDA_CHECK(cudaEventDestroy(f.ev));
    }
    for (cudaEvent_t ev : md->free_events) {
        CUDA_CHECK(cudaEventDestroy(ev));
    }
    CUDA_CHECK(cudaEventDestroy(md->snap_ev));
    CUDA_CHECK(cudaEventDestroy(md->evict_ev));
    CUDA_CHECK(cudaStreamDestroy(md->fill_stream));
    CUDA_CHECK(cudaFree(md->table_dev));
    CUDA_CHECK(cudaFree(md->counts_dev));
    CUDA_CHECK(cudaFreeHost(md->table_host));
    CUDA_CHECK(cudaFreeHost(md->counts_snap));
    delete md;
}
