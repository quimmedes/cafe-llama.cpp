#pragma once

#include <cstdint>

struct dspark_markov_metal;

// Legacy sequential correction only; weights are [vocabulary, rank] F32.
dspark_markov_metal * dspark_markov_metal_init(const float * w1,
                                               const float * w2,
                                               int64_t       vocabulary,
                                               int64_t       rank,
                                               int32_t       mask_id);
void                  dspark_markov_metal_free(dspark_markov_metal * context);
// Lowest vocabulary ID wins ties. Mask is excluded; nonfinite scores fail closed.
bool                  dspark_markov_metal_resample(dspark_markov_metal * context,
                                                   const float *         base_logits,
                                                   int32_t               anchor,
                                                   int32_t               slots,
                                                   int32_t *             output);
