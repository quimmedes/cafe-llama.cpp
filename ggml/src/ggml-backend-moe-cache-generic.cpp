// Generic MoE expert cache provider for backends without a dedicated one (Vulkan, Metal).
// Hot experts of host resident MUL_MAT_ID weights are copied into a slot pool on the GPU.
// A cache hit runs as ggml_mul_mat_id over the pool on the GPU, misses stay on the CPU.

#include "ggml-backend-moe-cache.h"

#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

constexpr int    gmc_min_slots      = 64;
constexpr int    gmc_max_rows       = 64;
constexpr int    gmc_max_batch      = 10;
constexpr int    gmc_fills_per_node = 2;
constexpr size_t gmc_min_expert     = 512u << 10;
constexpr size_t MiB                = 1u << 20;

ggml_backend_reg_t g_owner = nullptr;

using gmc_shape_key = std::tuple<int, int64_t, int64_t>; // type, n_in, n_out

struct gmc_shape {
    size_t expert_size = 0;
    int64_t n_expert = 0;
    std::set<const void *> tensors; // host base of every tensor with this shape
};

struct gmc_pool {
    int type = 0;
    int64_t n_in = 0;
    int64_t n_out = 0;
    size_t expert_size = 0;
    int n_slots = 0;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t = nullptr; // [n_in, n_out, n_slots]

    std::unordered_map<const void *, int> slot_of;
    std::vector<const void *> key_of;
    std::vector<uint64_t> last_use;

    ~gmc_pool() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

struct gmc_graph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * act = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out = nullptr;
    bool ok = false;

    ~gmc_graph() {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
};

struct gmc_session {
    ggml_backend_t be = nullptr;
    ggml_moe_cache_config config = {};

    std::map<gmc_shape_key, gmc_shape> census;
    bool finalized = false;
    bool disabled = false;

    std::map<gmc_shape_key, std::unique_ptr<gmc_pool>> pools;
    std::map<std::pair<gmc_pool *, int>, std::unique_ptr<gmc_graph>> graphs;

    // pinned when the device has a host buffer type, async copies from pageable memory are synchronous on Vulkan
    ggml_backend_buffer_t stage_buf = nullptr;
    std::vector<uint8_t> stage_vec;
    size_t stage_size = 0;
    uint64_t tick = 0;

    ~gmc_session() {
        if (be) {
            ggml_backend_synchronize(be);
        }
        graphs.clear();
        pools.clear();
        ggml_backend_buffer_free(stage_buf);
        ggml_backend_free(be);
    }
};

struct gmc_node {
    gmc_session * s = nullptr;
    gmc_pool * pool = nullptr;
    const char * base = nullptr;
    size_t expert_size = 0;
    int64_t n_tokens = 0;
    gmc_graph * g = nullptr;
    int n_hits = 0;
    const float * out = nullptr; // results in the staging buffer, valid after synchronize
};

std::mutex g_mu;
std::set<gmc_session *> g_sessions;
thread_local gmc_session * tl_session = nullptr;

bool gmc_is_owned_gpu(ggml_backend_dev_t dev) {
    return dev && g_owner && ggml_backend_dev_backend_reg(dev) == g_owner &&
           ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
}

size_t gmc_default_reserve(size_t total) {
    // 6% of the device, clamped to 1-3 GiB and at most a quarter, same policy as the CUDA provider
    size_t reserve = (total / 100 * 6) / (128*MiB) * (128*MiB);
    reserve = std::clamp(reserve, (size_t) 1024*MiB, (size_t) 3072*MiB);
    return std::min(reserve, total / 4);
}

int gmc_query_config(int automatic, size_t budget_mib, ggml_moe_cache_config * config) {
    GGML_UNUSED(automatic);
    if (!config || budget_mib > (SIZE_MAX >> 20)) {
        return 0;
    }
    *config = {};
    config->budget_bytes           = budget_mib * MiB;
    config->min_expert_bytes       = gmc_min_expert;
    config->max_batch              = gmc_max_batch;
    config->min_devices            = 1;
    config->overlap_cpu_rows       = -1;
    config->expert_parallel        = -1;
    return 1;
}

int gmc_query_device(void * device, const ggml_moe_cache_config * config, ggml_moe_cache_device_caps * caps) {
    ggml_backend_dev_t dev = (ggml_backend_dev_t) device;
    if (!gmc_is_owned_gpu(dev) || !config || !caps) {
        return 0;
    }
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);

    int index = 0;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(g_owner); i++) {
        if (ggml_backend_reg_dev_get(g_owner, i) == dev) {
            index = (int) i;
        }
    }
    *caps = {};
    caps->logical_device            = index;
    caps->physical_device           = index;
    caps->min_expert_bytes          = config->min_expert_explicit ? config->min_expert_bytes : gmc_min_expert;
    caps->recommended_reserve_bytes = gmc_default_reserve(total);
    return 1;
}

int gmc_query_shape(int wtype, int64_t n_in, int64_t n_out, int64_t n_expert, size_t expert_size, ggml_moe_cache_shape_caps * caps) {
    if (!caps || n_in <= 0 || n_out <= 0 || n_expert <= 0 || wtype < 0 || wtype >= GGML_TYPE_COUNT) {
        return 0;
    }
    if (ggml_row_size((ggml_type) wtype, n_in) * n_out != expert_size) {
        return 0;
    }
    caps->scratch_bytes = (size_t) gmc_max_rows * (n_in + n_out + 1) * sizeof(float);
    caps->pool_bytes    = (size_t) std::min<int64_t>(n_expert, gmc_min_slots) * expert_size;
    caps->minimum_bytes = (size_t) gmc_min_slots * expert_size;
    return 1;
}

void * gmc_session_create(void * const * backends, int n_backends, const ggml_moe_cache_config * config) {
    if (!config) {
        return nullptr;
    }
    ggml_backend_dev_t dev = nullptr;
    for (int i = 0; i < n_backends && !dev; i++) {
        ggml_backend_dev_t d = ggml_backend_get_device((ggml_backend_t) backends[i]);
        if (gmc_is_owned_gpu(d)) {
            dev = d;
        }
    }
    if (!dev) {
        return nullptr;
    }
    // a dedicated backend instance keeps cache work off the scheduler's own queue
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        return nullptr;
    }
    auto * s = new gmc_session();
    s->be = be;
    s->config = *config;

    std::lock_guard<std::mutex> lock(g_mu);
    g_sessions.insert(s);
    return s;
}

void gmc_session_destroy(void * session) {
    auto * s = (gmc_session *) session;
    if (!s) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_sessions.erase(s);
    }
    if (tl_session == s) {
        tl_session = nullptr;
    }
    delete s;
}

void gmc_session_enter(void * session) {
    tl_session = (gmc_session *) session;
}

void gmc_session_leave(void * session) {
    GGML_UNUSED(session);
    tl_session = nullptr;
}

// allocate the pools once every shape has been seen, the budget is split by expert footprint
void gmc_finalize(gmc_session * s) {
    s->finalized = true;

    size_t free = 0, total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(s->be), &free, &total);
    const size_t reserve = ggml_moe_cache_effective_reserve_bytes(s->config.reserve_explicit, s->config.reserve_bytes, gmc_default_reserve(total));
    size_t budget = free > reserve ? free - reserve : 0;
    if (s->config.budget_bytes > 0) {
        budget = std::min(budget, s->config.budget_bytes);
    }

    double footprint = 0.0;
    for (const auto & [key, shape] : s->census) {
        footprint += (double) shape.tensors.size() * shape.n_expert * shape.expert_size;
    }
    if (budget == 0 || footprint <= 0.0) {
        s->disabled = true;
        return;
    }

    size_t used = 0;
    for (const auto & [key, shape] : s->census) {
        const double share = (double) shape.tensors.size() * shape.n_expert * shape.expert_size / footprint;
        int64_t n_slots = std::min<int64_t>((int64_t) (budget * share / shape.expert_size),
                                            (int64_t) shape.tensors.size() * shape.n_expert);

        auto pool = std::make_unique<gmc_pool>();
        pool->type        = std::get<0>(key);
        pool->n_in        = std::get<1>(key);
        pool->n_out       = std::get<2>(key);
        pool->expert_size = shape.expert_size;

        // a failed allocation retries with half the slots until the 64 slot floor
        for (; n_slots >= gmc_min_slots; n_slots /= 2) {
            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            pool->ctx = ggml_init(params);
            pool->t   = ggml_new_tensor_3d(pool->ctx, (ggml_type) pool->type, pool->n_in, pool->n_out, n_slots);
            pool->buf = ggml_backend_alloc_ctx_tensors(pool->ctx, s->be);
            if (pool->buf) {
                break;
            }
            ggml_free(pool->ctx);
            pool->ctx = nullptr;
        }
        if (!pool->buf) {
            continue;
        }
        ggml_backend_buffer_set_usage(pool->buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        pool->n_slots = (int) n_slots;
        pool->key_of.assign(n_slots, nullptr);
        pool->last_use.assign(n_slots, 0);
        used += ggml_backend_buffer_get_size(pool->buf);
        s->pools.emplace(key, std::move(pool));
    }

    if (s->pools.empty()) {
        s->disabled = true;
        return;
    }
    GGML_LOG_INFO("[moe-cache] enabled on %s: %zu pools, %zu MiB\n",
                  ggml_backend_name(s->be), s->pools.size(), used / MiB);
}

gmc_graph * gmc_get_graph(gmc_session * s, gmc_pool * pool, int n) {
    auto & slot = s->graphs[{pool, n}];
    if (slot) {
        return slot->ok ? slot.get() : nullptr;
    }
    slot = std::make_unique<gmc_graph>();
    gmc_graph * g = slot.get();

    ggml_init_params params = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g->ctx = ggml_init(params);
    g->act = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, pool->n_in, 1, n);
    g->ids = ggml_new_tensor_2d(g->ctx, GGML_TYPE_I32, 1, n);
    g->out = ggml_mul_mat_id(g->ctx, pool->t, g->act, g->ids);
    if (!ggml_backend_supports_op(s->be, g->out)) {
        return nullptr;
    }
    g->gf = ggml_new_graph(g->ctx);
    ggml_build_forward_expand(g->gf, g->out);
    g->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s->be));
    g->ok = ggml_gallocr_alloc_graph(g->galloc, g->gf);
    return g->ok ? g : nullptr;
}

void * gmc_begin(const char * tensor_name, const void * host_base, size_t expert_size,
                 int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
                 int64_t n_tokens, int64_t n_rows) {
    gmc_session * s = tl_session;
    if (!s || s->disabled || !tensor_name || !host_base ||
        (!strstr(tensor_name, "_exps") && !strstr(tensor_name, "_chexps"))) {
        return nullptr;
    }
    const size_t min_expert = s->config.min_expert_explicit ? s->config.min_expert_bytes : gmc_min_expert;
    const int max_batch = s->config.max_batch > 0 ? s->config.max_batch : gmc_max_batch;
    if (expert_size < min_expert || n_tokens > max_batch || n_rows > gmc_max_rows ||
        ggml_row_size((ggml_type) wtype, n_in) * n_out != expert_size) {
        return nullptr;
    }

    const gmc_shape_key key = { wtype, n_in, n_out };
    if (!s->finalized) {
        auto & shape = s->census[key];
        shape.expert_size = expert_size;
        shape.n_expert = n_expert;
        // the first repeat of a tensor means one full pass has been seen
        if (!shape.tensors.insert(host_base).second) {
            gmc_finalize(s);
        }
        if (!s->finalized || s->disabled) {
            return nullptr;
        }
    }

    auto it = s->pools.find(key);
    if (it == s->pools.end()) {
        return nullptr;
    }
    auto * node = new gmc_node();
    node->s           = s;
    node->pool        = it->second.get();
    node->base        = (const char *) host_base;
    node->expert_size = expert_size;
    node->n_tokens    = n_tokens;
    s->tick++;
    return node;
}

int gmc_plan(void * opaque, const int32_t * ids, int n_ids, int32_t * slot_idx) {
    auto * node = (gmc_node *) opaque;
    gmc_session * s = node->s;
    gmc_pool * pool = node->pool;

    std::vector<int32_t> misses;
    int n_hits = 0;
    for (int i = 0; i < n_ids; i++) {
        slot_idx[i] = -1;
        if (ids[i] < 0) {
            continue;
        }
        const void * key = node->base + (size_t) ids[i] * node->expert_size;
        auto it = pool->slot_of.find(key);
        if (it != pool->slot_of.end()) {
            slot_idx[i] = it->second;
            pool->last_use[it->second] = s->tick;
            n_hits++;
        } else if (std::find(misses.begin(), misses.end(), ids[i]) == misses.end()) {
            misses.push_back(ids[i]);
        }
    }

    // bounded demand fill into the least recently used slots not touched by this node
    int filled = 0;
    for (int32_t id : misses) {
        if (filled >= gmc_fills_per_node) {
            break;
        }
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int j = 0; j < pool->n_slots; j++) {
            if (pool->last_use[j] < oldest && pool->last_use[j] != s->tick) {
                oldest = pool->last_use[j];
                victim = j;
            }
        }
        if (victim < 0) {
            break;
        }
        if (pool->key_of[victim]) {
            pool->slot_of.erase(pool->key_of[victim]);
        }
        const void * key = node->base + (size_t) id * node->expert_size;
        ggml_backend_tensor_set_async(s->be, pool->t, key, (size_t) victim * node->expert_size, node->expert_size);
        pool->key_of[victim] = key;
        pool->slot_of[key] = victim;
        // stays out of this node, the slot is filled behind any earlier work in queue order
        pool->last_use[victim] = s->tick;
        filled++;
    }
    return n_hits;
}

char * gmc_stage_base(gmc_session * s) {
    return s->stage_buf ? (char *) ggml_backend_buffer_get_base(s->stage_buf) : (char *) s->stage_vec.data();
}

char * gmc_stage(gmc_session * s, size_t size) {
    if (size <= s->stage_size) {
        return gmc_stage_base(s);
    }
    // the previous node was collected, so no copy still uses the old buffer
    ggml_backend_buffer_free(s->stage_buf);
    s->stage_buf = nullptr;
    s->stage_size = 0;
    ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(s->be));
    if (host) {
        s->stage_buf = ggml_backend_buft_alloc_buffer(host, size);
    }
    if (!s->stage_buf) {
        s->stage_vec.resize(size);
    }
    s->stage_size = size;
    return gmc_stage_base(s);
}

int gmc_dispatch(void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                 const int32_t * slot_idx, const float * const * act_rows) {
    auto * node = (gmc_node *) opaque;
    gmc_session * s = node->s;
    gmc_pool * pool = node->pool;
    if (n_hits <= 0 || n_hits > gmc_max_rows || wtype != pool->type || n_in != pool->n_in || n_out != pool->n_out) {
        return 0;
    }
    // a few power of two graph sizes, so changing hit counts do not switch the backend between many graphs
    int n_pad = 1;
    while (n_pad < n_hits) {
        n_pad *= 2;
    }
    gmc_graph * g = gmc_get_graph(s, pool, n_pad);
    if (!g) {
        return 0;
    }

    const size_t act_bytes = (size_t) n_pad * n_in * sizeof(float);
    const size_t ids_bytes = (size_t) n_pad * sizeof(int32_t);
    char * stage = gmc_stage(s, act_bytes + ids_bytes + (size_t) n_pad * n_out * sizeof(float));
    if (!stage) {
        return 0;
    }
    float   * act = (float *) stage;
    int32_t * ids = (int32_t *) (stage + act_bytes);
    for (int i = 0; i < n_hits; i++) {
        memcpy(act + (size_t) i * n_in, act_rows[i], n_in * sizeof(float));
        ids[i] = slot_idx[i];
    }
    // padded rows compute slot 0 on zeros, their output is never read
    memset(act + (size_t) n_hits * n_in, 0, (size_t) (n_pad - n_hits) * n_in * sizeof(float));
    for (int i = n_hits; i < n_pad; i++) {
        ids[i] = 0;
    }
    ggml_backend_tensor_set_async(s->be, g->act, act, 0, ggml_nbytes(g->act));
    ggml_backend_tensor_set_async(s->be, g->ids, ids, 0, ggml_nbytes(g->ids));
    if (ggml_backend_graph_compute_async(s->be, g->gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_synchronize(s->be);
        return 0;
    }
    ggml_backend_tensor_get_async(s->be, g->out, stage + act_bytes + ids_bytes, 0, ggml_nbytes(g->out));
    node->g = g;
    node->n_hits = n_hits;
    node->out = (const float *) (stage + act_bytes + ids_bytes);
    return 1;
}

int gmc_collect(void * opaque, int n_hits, float * const * dst_rows, int64_t n_out) {
    auto * node = (gmc_node *) opaque;
    gmc_session * s = node->s;
    if (!node->g || n_hits != node->n_hits || n_out != node->pool->n_out) {
        return 0;
    }
    ggml_backend_synchronize(s->be);
    const float * out = node->out;
    for (int i = 0; i < n_hits; i++) {
        memcpy(dst_rows[i], out + (size_t) i * n_out, n_out * sizeof(float));
    }
    return 1;
}

void gmc_end(void * opaque) {
    delete (gmc_node *) opaque;
}

void gmc_invalidate(const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    std::lock_guard<std::mutex> lock(g_mu);
    for (gmc_session * s : g_sessions) {
        bool synced = false;
        for (auto & [key, pool] : s->pools) {
            for (int j = 0; j < pool->n_slots; j++) {
                const char * k = (const char *) pool->key_of[j];
                if (!k || k + pool->expert_size <= lo || k >= hi) {
                    continue;
                }
                if (!synced) {
                    // a pending fill may still read the range
                    ggml_backend_synchronize(s->be);
                    synced = true;
                }
                pool->slot_of.erase(k);
                pool->key_of[j] = nullptr;
                pool->last_use[j] = 0;
            }
        }
    }
}

} // namespace

void ggml_moe_cache_register_generic(ggml_backend_reg_t reg) {
    if (!reg || ggml_moe_cache.owner) {
        return;
    }
    g_owner = reg;
    ggml_moe_cache.owner           = reg;
    ggml_moe_cache.query_config    = gmc_query_config;
    ggml_moe_cache.query_device    = gmc_query_device;
    ggml_moe_cache.query_shape     = gmc_query_shape;
    ggml_moe_cache.session_create  = gmc_session_create;
    ggml_moe_cache.session_destroy = gmc_session_destroy;
    ggml_moe_cache.session_enter   = gmc_session_enter;
    ggml_moe_cache.session_leave   = gmc_session_leave;
    ggml_moe_cache.begin           = gmc_begin;
    ggml_moe_cache.plan            = gmc_plan;
    ggml_moe_cache.dispatch        = gmc_dispatch;
    ggml_moe_cache.collect         = gmc_collect;
    ggml_moe_cache.end             = gmc_end;
    ggml_moe_cache.invalidate      = gmc_invalidate;
}
