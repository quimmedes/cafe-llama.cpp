#include "ggml-turbo-meansub.h"

#include <string.h>
#include <stdlib.h>
#include <mutex>
#include <atomic>

// Registry entry shape consumed by the generated data table. Compact storage: only live layers
// are stored ([n_live][max_c]) plus their layer indices; the accessor expands to a dense
// [max_l * max_c] slab (dead layers zero) on first use.
typedef struct {
    const char *  arch;
    int           n_layer;
    int           n_embd;
    int           max_l;
    int           max_c;
    int           k_live;
    int           v_live;
    const int *   klay;   // [k_live] layer indices
    const float * kval;   // [k_live * max_c] flattened
    const int *   vlay;   // [v_live]
    const float * vval;   // [v_live * max_c]
} ggml_meansub_entry;

// Before adding a baked affine-tap table, verify that the model has no learned attention sinks.
// K centering shifts only real-token logits, while a learned sink logit is inserted separately;
// V restoration likewise assumes the real-token attention weights sum to one. Supporting sinks
// requires an explicit sink-aware attention contract, not merely another calibrated mean table.
// Also verify the V-mean coordinate contract when head_dim is not a multiple of 128 and there is
// more than one KV head: encode calibration uses padded per-head rows, while graph restoration
// currently expands the table with compact kv_head*head_dim coordinates. The baked models below
// have aligned head dimensions; a padded-head model needs that mismatch fixed before adding it.
#include "ggml-turbo-meansub-data.inc"   // defines g_meansub_table[] + g_meansub_count

static_assert(
        sizeof(g_meansub_table)/sizeof(g_meansub_table[0]) < GGML_TURBO_MEANSUB_MAX_MODELS,
        "GGML_TURBO_MEANSUB_MAX_MODELS must include the unsupported-model slot");

static std::mutex g_dense_mutex;
static std::atomic<float *> g_dense_k[sizeof(g_meansub_table)/sizeof(g_meansub_table[0])] = {};
static std::atomic<float *> g_dense_v[sizeof(g_meansub_table)/sizeof(g_meansub_table[0])] = {};

GGML_API int ggml_turbo_meansub_model_id(const char * arch, int n_layer, int n_embd) {
    if (!arch || !arch[0]) {
        return 0;
    }
    for (int i = 0; i < g_meansub_count; i++) {
        const ggml_meansub_entry * e = &g_meansub_table[i];
        if (n_layer == e->n_layer && n_embd == e->n_embd && strcmp(arch, e->arch) == 0) {
            return i + 1;
        }
    }
    return 0;
}

static float * expand_dense(const ggml_meansub_entry * e, int kvsel) {
    const int     L      = e->max_l;
    const int     C      = e->max_c;
    const int     nlive  = kvsel ? e->v_live : e->k_live;
    const int *   lay    = kvsel ? e->vlay   : e->klay;
    const float * val    = kvsel ? e->vval   : e->kval;

    float * dense = (float *) calloc((size_t) L * C, sizeof(float));
    if (!dense) {
        return NULL;
    }
    for (int i = 0; i < nlive; i++) {
        const int l = lay[i];
        if (l < 0 || l >= L) {
            continue;
        }
        memcpy(dense + (size_t) l * C, val + (size_t) i * C, (size_t) C * sizeof(float));
    }
    return dense;
}

GGML_API const float * ggml_turbo_meansub_table(
        int model_id, int kvsel, int * out_max_l, int * out_max_c, int * out_live) {
    if (model_id <= 0 || model_id > g_meansub_count) {
        return NULL;
    }
    const int index = model_id - 1;
    const ggml_meansub_entry * e = &g_meansub_table[index];
    if (out_max_l) *out_max_l = e->max_l;
    if (out_max_c) *out_max_c = e->max_c;
    if (out_live)  *out_live  = kvsel ? e->v_live : e->k_live;

    std::atomic<float *> & slot = kvsel ? g_dense_v[index] : g_dense_k[index];
    float * dense = slot.load(std::memory_order_acquire);
    if (!dense) {
        std::lock_guard<std::mutex> lock(g_dense_mutex);
        dense = slot.load(std::memory_order_relaxed);
        if (!dense) {
            dense = expand_dense(e, kvsel);
            slot.store(dense, std::memory_order_release);
        }
    }
    return dense;
}
