#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    ggml_op norm_guard = GGML_OP_CLAMP; // the op on the weight sum before the division: CLAMP (a floor) or SCALE (s*sum + b)
    bool scale{};
    int  n_groups{};     // > 0: pick experts only from the top n_group_used expert groups
    int  n_group_used{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp, // the norm guard node, CLAMP or SCALE
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

// one token: the router MUL_MAT (F32 rows) of the logits and the topk of ggml_cuda_op_topk_moe as one launch, false if they do not fit
bool ggml_cuda_op_topk_moe_router(ggml_backend_cuda_context &     ctx,
                                  const ggml_tensor *             router,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_tensor *             bias,
                                  const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
