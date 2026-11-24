#include "common.cuh"
#include "fwht.cuh"

#include <cstdlib>

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Large-N path. The register kernel above keeps N/warp_size floats per thread, so it stops being
// viable well before the arithmetic does: N=4096 would need 128 registers per thread and spill,
// which is why the switch below used to end at 2048 and simply decline anything larger (the whole
// op then fell back to CPU). Stage the row in shared memory instead and run all log2(N) butterfly
// stages there. One row per block; every thread handles several butterflies per stage. Slower per
// row than the register path, so it is used only where that path cannot go.
#define FWHT_SMEM_THREADS 256

template <int N, typename T, bool has_signs>
__launch_bounds__(FWHT_SMEM_THREADS, 1)
__global__ void fwht_cuda_smem(const T * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk) {
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        float v = fwht_load(src[i]) * scale;
        if (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    __syncthreads();

    // Same butterfly and the same sign convention as the register path: the low element of a pair
    // takes x + y, the high one x - y.
#pragma unroll 1
    for (int h = 1; h < N; h *= 2) {
        for (int idx = threadIdx.x; idx < N / 2; idx += FWHT_SMEM_THREADS) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        dst[i] = s[i];
    }
}


// Wide rows at small row counts (decode): one row per block instead of per warp.
// The warp kernel serialises every stage on one warp, and the shared-memory kernel above synchronises on each stage.
// Both leave most of the GPU idle at these shapes.
#define FWHT_BLOCK_THREADS 256

template <int N, int NT, typename T, bool has_signs>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = fwht_load(src[i * NT + tid]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        default:
            break;
    }
    // From 512 up, one block of FWHT_BLOCK_THREADS per row (fwht_cuda_block).
    // The older kernels were the largest single kernel of a decode step at these widths.
    // GGML_CUDA_FWHT_LEGACY=1 restores them for A/B.
#define FWHT_SMEM_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_SMEM_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
#define FWHT_BLOCK_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
    static const bool legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;
    if (legacy) {
        switch (n) {
            FWHT_CASE(512)
            FWHT_CASE(1024)
            FWHT_CASE(2048)
            FWHT_SMEM_CASE(4096)
#if !defined(GGML_USE_MUSA)
            // 32 KB of shared memory, above the MUSA limit; falls back there
            FWHT_SMEM_CASE(8192)
#endif // !defined(GGML_USE_MUSA)
            default:
                return false;
        }
    }
    switch (n) {
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        FWHT_BLOCK_CASE(2048)
        FWHT_BLOCK_CASE(4096)
#if !defined(GGML_USE_MUSA)
        FWHT_BLOCK_CASE(8192)
#endif // !defined(GGML_USE_MUSA)
#undef FWHT_CASE
#undef FWHT_SMEM_CASE
#undef FWHT_BLOCK_CASE
        default:
            return false;
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk);
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

static __global__ void __launch_bounds__(4*WARP_SIZE) k_fwht_rows(const fwht_rows_params p) {
    const int r    = blockIdx.x*blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (r >= p.n_rows) {
        return;
    }

    ggml_cuda_pdl_sync();

    float reg[FWHT_ROWS_EL];
#pragma unroll
    for (int i = 0; i < FWHT_ROWS_EL; ++i) {
        reg[i] = p.x[(int64_t) r*128 + i*WARP_SIZE + lane];
    }
    fwht_rows_row(p, r, lane, reg);
}

bool ggml_cuda_fwht_rows_plan(const ggml_tensor * pre, const ggml_tensor * tok, const ggml_tensor * mm, const ggml_tensor * set_rows,
        fwht_rows_params & p) {
    // the rows of the pre-op are [128, n_head, n_tokens]
    if (!pre) {
        return false;
    }
    const ggml_tensor * x = pre->src[0];
    if (!ggml_cuda_op_mul_mat_use_fwht(mm) || (mm->ne[0] != 128 && mm->ne[0] != 64) || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
            ggml_nelements(x) != ggml_nelements(mm) || x->ne[3] != 1) {
        return false;
    }

    p = {};
    p.x      = (const float *) x->data;
    p.n_head = x->ne[1];
    p.n_rows = ggml_nelements(x)/128;
    p.n_rot  = mm->ne[0];
    p.scale  = 1.0f/sqrtf((float) p.n_rot);
    p.s_rows = 1;
    p.pre    = FWHT_ROWS_PRE_NONE;

    if (x->ne[0] != 128 || ggml_nrows(x) != p.n_rows) {
        return false;
    }
    if (pre && pre->op == GGML_OP_L2_CLIP) {
        const ggml_tensor * s = pre->src[1];
        if (s && (s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) || s->ne[0] != 128 || s->ne[2] != 1 || s->ne[3] != 1)) {
            return false;
        }
        p.pre    = FWHT_ROWS_PRE_L2_CLIP;
        p.s      = s ? (const float *) s->data : nullptr;
        p.s_rows = s ? s->ne[1] : 1;
        p.p0     = ggml_get_op_params_f32(pre, 0);
        p.eps    = ggml_get_op_params_f32(pre, 1);
    } else if (pre && pre->op == GGML_OP_RMS_NORM) {
        p.pre = FWHT_ROWS_PRE_RMS_NORM;
        p.eps = ggml_get_op_params_f32(pre, 0);
    } else if (pre) {
        return false;
    }
    if (tok) {
        // the factor of a token: a [1, 1, n_tokens] operand of a MUL
        const ggml_tensor * f = tok->src[1]->ne[0] == 1 ? tok->src[1] : tok->src[0];
        if (f->type != GGML_TYPE_F32 || !ggml_is_contiguous(f) || ggml_nelements(f) != x->ne[2] || f->ne[0] != 1 || f->ne[1] != 1) {
            return false;
        }
        p.tok = (const float *) f->data;
    }

    if (set_rows) {
        const ggml_tensor * idx = set_rows->src[1];
        if (set_rows->type != GGML_TYPE_Q8_0 || (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) || !ggml_is_contiguous(idx) ||
                set_rows->ne[0] != 128*p.n_head || ggml_nelements(idx) != x->ne[2] || set_rows->ne[2] != 1 || set_rows->ne[3] != 1) {
            return false;
        }
        p.dst_q   = (char *) set_rows->data;
        p.dst_nb1 = set_rows->nb[1];
        p.idx     = idx->data;
        p.idx_i64 = idx->type == GGML_TYPE_I64;
    } else {
        if (mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm)) {
            return false;
        }
        p.dst_f = (float *) mm->data;
    }
    return true;
}

bool ggml_cuda_fwht_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * pre, const ggml_tensor * tok,
        const ggml_tensor * mm, const ggml_tensor * set_rows) {
    fwht_rows_params p;
    if (!ggml_cuda_fwht_rows_plan(pre, tok, mm, set_rows, p)) {
        return false;
    }

    const dim3 block_dims(WARP_SIZE, 4, 1);
    const dim3 blocks_num((p.n_rows + 3)/4, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_fwht_rows, launch_params, p);
    return true;
}

bool ggml_cuda_fwht_rows_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig, ggml_tensor * mul) {
    const ggml_tensor * x = mm->src[1];
    const ggml_tensor * g = sig->src[0];
    if (!ggml_cuda_op_mul_mat_use_fwht(mm) || (mm->ne[0] != 128 && mm->ne[0] != 64) || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
            ggml_nelements(x) % 128 != 0 || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || ggml_nelements(mul) != ggml_nelements(x) ||
            g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) || ggml_nelements(g) != ggml_nelements(x)) {
        return false;
    }

    fwht_rows_params p = {};
    p.x      = (const float *) x->data;
    p.gate   = (const float *) g->data;
    p.dst_f  = (float *) mul->data;
    p.n_head = 1;
    p.n_rows = ggml_nelements(x)/128;
    p.n_rot  = mm->ne[0];
    p.scale  = 1.0f/sqrtf((float) p.n_rot);
    p.s_rows = 1;
    p.pre    = FWHT_ROWS_PRE_NONE;

    // a mat-vec reads mul: its q8_1 too, see mmvq_shared
    const auto & shared = ctx.mmvq_shared;
    const bool   q8_1   = shared.src1 && shared.data && (shared.src1 == mul || shared.src1->view_src == mul) &&
        ggml_nelements(shared.src1) == ggml_nelements(mul) && ggml_nrows(shared.src1) == 1;
    if (q8_1) {
        p.dst_q8_1 = (block_q8_1 *) shared.data;
        p.n_q8_1   = GGML_PAD(shared.src1->ne[0], MATRIX_ROW_PADDING)/QK8_1;
    }

    const dim3 block_dims(WARP_SIZE, 4, 1);
    const dim3 blocks_num((p.n_rows + 3)/4, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_fwht_rows, launch_params, p);
    if (q8_1) {
        *shared.quantized = true;
    }
    return true;
}
