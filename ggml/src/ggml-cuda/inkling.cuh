#include "common.cuh"

#define STATE_MIX_MAX_Q 4

void ggml_cuda_op_state_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// one token: the mix and the RMS_NORM -> MUL of some of its query outputs (norms[k], muls[k]) as one kernel, false if they do not fit
bool ggml_cuda_op_state_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * mix, const ggml_tensor * const * norms, ggml_tensor * const * muls, int n_norm);
// independent mixes of one token next to each other as one launch, false if they do not fit
bool ggml_cuda_op_state_mix_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mixes, int n);
void ggml_cuda_op_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_rel_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// concat(past, cur) -> cpy(last K - 1 columns to the state) -> short_conv (-> add of a residual) as one kernel, false if the nodes do not fit
bool ggml_cuda_op_short_conv_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy, ggml_tensor * dst,
        ggml_tensor * add);
// L2_CLIP of r -> (MUL by a factor of each token) -> MUL_MAT by the bias table as one kernel, false if the nodes do not fit
bool ggml_cuda_op_rel_table(ggml_backend_cuda_context & ctx, const ggml_tensor * clip, const ggml_tensor * tok, ggml_tensor * mm);
// a few independent LOOP_CONTROL nodes of the same shape as one launch pair, false if they do not fit
bool ggml_cuda_op_loop_control_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * dst, int n_ctrl);

enum ggml_cuda_attn_prep_kind {
    GGML_CUDA_ATTN_PREP_CONV,   // a: concat, b: cpy (the tail to the state), c: short_conv
    GGML_CUDA_ATTN_PREP_ROWS,   // a: L2_CLIP or RMS_NORM, b: MUL by a token factor or nullptr, c: Hadamard MUL_MAT, d: SET_ROWS or nullptr
    GGML_CUDA_ATTN_PREP_TABLE,  // a: L2_CLIP, b: MUL by a token factor or nullptr, c: MUL_MAT by the bias table
    GGML_CUDA_ATTN_PREP_MATVEC, // a: MUL_MAT of one token by at most 64 rows of BF16 or F32
};

struct ggml_cuda_attn_prep_group {
    ggml_cuda_attn_prep_kind kind;
    const ggml_tensor * a;
    const ggml_tensor * b;
    const ggml_tensor * c;
    const ggml_tensor * d;
    bool internal = false; // CONV: only the row group of the prep reads c, so c is not stored
};

// the groups of one token's attention inputs as one launch: each conv must feed one row group, which makes its values in the same warp
// false if the groups do not fit or one writes what another reads
bool ggml_cuda_op_attn_prep(ggml_backend_cuda_context & ctx, const ggml_cuda_attn_prep_group * groups, int n_groups);

// one token: the weighted sum of the experts (see ggml_cuda_op_moe_weighted_reduction) -> concat, tail cpy and short_conv -> add of res
// as one kernel, false if the nodes do not fit
bool ggml_cuda_op_moe_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * expert_scale, const ggml_tensor * weights,
        const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * conv, ggml_tensor * add, const ggml_tensor * res);
