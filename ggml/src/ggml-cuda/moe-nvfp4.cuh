#include "common.cuh"

// one token through the NVFP4 experts of mm (MUL_MAT_ID), with float activations (no q8_1 copy):
// dst[r, slot] = scale[e]*sum_k w[e][r][k]*x[slot][k], e = ids[slot], scale optional (one value per expert)
// with glu (SITU, see ggml_situ_split): dst[r, slot] = situ(gate row r, up row n_ff + r) of the same products
bool ggml_cuda_nvfp4_moe_vec_supported(const ggml_tensor * mm);
// copy_x: dst overlaps x, the kernel reads a copy of x
void ggml_cuda_nvfp4_moe_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * scale,
        const ggml_tensor * glu, ggml_tensor * dst, bool copy_x);
