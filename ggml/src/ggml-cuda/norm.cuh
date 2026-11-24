#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx,
                       ggml_tensor *               dst,
                       ggml_tensor *               mul = nullptr,
                       ggml_tensor *               add = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_add_rms_norm_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

void ggml_cuda_op_add_rms_norm_scale_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, float * row_scale);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// a few independent RMS_NORM -> MUL (norms[k], muls[k]) of F32 rows as one launch, with the q8_1 of the mat-vecs that read muls[k]
// (see mmvq_q8_of); false if they do not fit
bool ggml_cuda_op_rms_norm_mul_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * norms, ggml_tensor * const * muls, int n);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
