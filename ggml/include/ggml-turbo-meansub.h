#pragma once

// Baked per-architecture KV affine-tap means (computed mu = sum/cnt, frozen from offline PFH1
// calibration). Lives in the runtime/codec, NOT in the GGUF: one table per base architecture
// transfers across finetunes and weight-quant tiers. Each loaded model retains its registry ID;
// the CUDA encode path and graph decode path use that ID to select the matching immutable slab when
// no TURBO_KMEAN_SUB / TURBO_VMEAN_SUB env override is set.

#include <stdint.h>
#include "ggml.h"

#define GGML_TURBO_MEANSUB_MAX_MODELS 16
#define GGML_TURBO_MEANSUB_MAX_L      128
#define GGML_TURBO_MEANSUB_MAX_C      2048

#ifdef __cplusplus
extern "C" {
#endif

// Resolve a stable registry ID for one model identity. arch = general.architecture string.
// The match requires arch + n_layer + n_embd to agree. Returns 0 when no table matches.
GGML_API int ggml_turbo_meansub_model_id(const char * arch, int n_layer, int n_embd);

// Dense baked mean slab [max_l * max_c] for model_id; kvsel: 0 = K, 1 = V.
// Returns NULL when model_id is 0 or invalid. Returned storage is immutable and process-lifetime;
// out_* may be NULL.
GGML_API const float * ggml_turbo_meansub_table(
        int model_id, int kvsel, int * out_max_l, int * out_max_c, int * out_live);

#ifdef __cplusplus
}
#endif
