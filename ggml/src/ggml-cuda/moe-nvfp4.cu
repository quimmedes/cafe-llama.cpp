#include "moe-nvfp4.cuh"

#define NVFP4_MOE_WARPS 4
#define NVFP4_MOE_ROWS   2 // rows (with glu: gate/up row pairs) of a warp at once
#define NVFP4_MOE_GROUPS 1 // groups of rows a block runs one after the other (more were slower on a RTX 3090)

// the scales and the codes of sub-block sb of each row
template <int NR>
static __device__ __forceinline__ void nvfp4_moe_load(const char * const (&row)[NR], const int sb, uint32_t (&d)[NR], uint32_t (&q0)[NR], uint32_t (&q1)[NR]) {
    const int b = sb >> 2;
    const int s = sb & 3;
#pragma unroll
    for (int q = 0; q < NR; ++q) {
        const uint32_t * blk = (const uint32_t *) (row[q] + 36*b);
        d[q]  = __ldg(blk);
        q0[q] = __ldg(blk + 1 + 2*s);
        q1[q] = __ldg(blk + 2 + 2*s);
    }
}

// acc[q] += the products of sub-block sb of row q and the 16 inputs of sb
template <int NR>
static __device__ __forceinline__ void nvfp4_moe_dot(float (&acc)[NR], const float * lut, const float * xs, const int sb,
        const uint32_t (&d)[NR], const uint32_t (&q0)[NR], const uint32_t (&q1)[NR]) {
    const int s = sb & 3;

    // byte j of a sub-block holds value j (low nibble) and value j + 8 (high nibble)
    const float4 * xp = (const float4 *) (xs + 16*sb);
    const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    const float xv[16] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w };

#pragma unroll
    for (int q = 0; q < NR; ++q) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum += lut[(q0[q] >> (8*j    )) & 0xF]*xv[j]     + lut[(q0[q] >> (8*j + 4)) & 0xF]*xv[j + 8];
            sum += lut[(q1[q] >> (8*j    )) & 0xF]*xv[j + 4] + lut[(q1[q] >> (8*j + 4)) & 0xF]*xv[j + 12];
        }
        acc[q] += ggml_cuda_ue4m3_to_fp32((d[q] >> (8*s)) & 0xFF)*sum;
    }
}

// block (row groups, slot): each warp NVFP4_MOE_ROWS rows of each of NVFP4_MOE_GROUPS groups, its lanes a stride of the 16-value sub-blocks
template <bool glu>
static __global__ void __launch_bounds__(NVFP4_MOE_WARPS*WARP_SIZE)
k_nvfp4_moe_vec(const char * __restrict__ w, const size_t nb01, const size_t nb02, const int32_t * __restrict__ ids,
        const float * __restrict__ x, const int64_t x_stride, const float * __restrict__ scale,
        float * __restrict__ dst, const int64_t dst_stride, const int n_sub, const int n_out, const int n_ff,
        const float beta, const float lb) {
    constexpr int NR = glu ? 2*NVFP4_MOE_ROWS : NVFP4_MOE_ROWS;

    __shared__ float lut[16];

    const int slot = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    if (threadIdx.x < 16) {
        lut[threadIdx.x] = kvalues_mxfp4[threadIdx.x];
    }

    ggml_cuda_pdl_sync();

    const int e = ids[slot];
    const float * xs = x + slot*x_stride;
    const float   sc = scale ? scale[e] : 1.0f;
    __syncthreads();

    for (int g = 0; g < NVFP4_MOE_GROUPS; ++g) {
        const int r0 = ((blockIdx.x*NVFP4_MOE_GROUPS + g)*NVFP4_MOE_WARPS + warp)*NVFP4_MOE_ROWS;
        if (r0 >= n_out) {
            break;
        }

        const char * row[NR];
#pragma unroll
        for (int q = 0; q < NVFP4_MOE_ROWS; ++q) {
            const int r = min(r0 + q, n_out - 1);
            row[q] = w + e*nb02 + r*nb01;
            if constexpr (glu) {
                row[NVFP4_MOE_ROWS + q] = w + e*nb02 + (n_ff + r)*nb01;
            }
        }

        float acc[NR] = {};
        for (int sb = lane; sb < n_sub; sb += WARP_SIZE) {
            uint32_t d[NR], q0[NR], q1[NR];
            nvfp4_moe_load<NR>(row, sb, d, q0, q1);
            nvfp4_moe_dot<NR>(acc, lut, xs, sb, d, q0, q1);
        }

#pragma unroll
        for (int q = 0; q < NR; ++q) {
            acc[q] = warp_reduce_sum(acc[q]);
        }
        if (lane == 0) {
#pragma unroll
            for (int q = 0; q < NVFP4_MOE_ROWS; ++q) {
                if (r0 + q < n_out) {
                    float y = acc[q]*sc;
                    if constexpr (glu) {
                        const float u = acc[NVFP4_MOE_ROWS + q]*sc;
                        y = tanhf(y*(1.0f/beta))*beta*(1.0f/(1.0f + expf(-y))) * (lb > 0.0f ? tanhf(u*(1.0f/lb))*lb : u);
                    }
                    dst[slot*dst_stride + r0 + q] = y;
                }
            }
        }
    }
}

#define NVFP4_MMA_SPLIT_GLU  12 // warps of a block with glu, each a stride of the 64-value blocks of the rows (tuned on a RTX 3090)
#define NVFP4_MMA_SPLIT_DOWN 6  // the same without glu

// fp16x2 of the codes in the low nibbles of bytes 0 and 1 of s (the doubled E2M1 values of kvalues_mxfp4), times the half2 scale
static __device__ __forceinline__ uint32_t nvfp4_pair_h2(const uint32_t s, const uint32_t scale) {
    // the high byte of the half of each magnitude 0..7 (0, 1, 2, 3, 4, 6, 8, 12), the low bytes 0; the code sign to the half sign
    const uint32_t mag  = __byte_perm(0x42403c00u, 0x4a484644u, (s & 0x0707u) << 4);
    const uint32_t sign = ((s & 0x0808u) * 0x101000u) & 0x80008000u;
    const uint32_t h    = mag | sign;
    const half2    r    = __hmul2(*(const half2 *) &h, *(const half2 *) &scale);
    return *(const uint32_t *) &r;
}

// a lane's part of a 64-value block of its two rows: the scales and the codes of sub-block t
struct nvfp4_mma_step {
    uint32_t sa, sb, a0, a1, b0, b1;
};

static __device__ __forceinline__ nvfp4_mma_step nvfp4_mma_load(const char * a, const char * b, const int k, const int t) {
    const uint32_t * ap = (const uint32_t *) (a + 36*k);
    const uint32_t * bp = (const uint32_t *) (b + 36*k);
    nvfp4_mma_step s;
    s.sa = __ldg(ap);
    s.sb = __ldg(bp);
    s.a0 = __ldg(ap + 1 + 2*t);
    s.a1 = __ldg(ap + 2 + 2*t);
    s.b0 = __ldg(bp + 1 + 2*t);
    s.b1 = __ldg(bp + 2 + 2*t);
    return s;
}

// block (16 rows, slot) on the tensor cores (m16n8k16): with glu rows g = 8 gate rows and g + 8 = the same up rows, else 16 rows.
// Lane (g, t) = (lane/4, lane%4): sub-block t of rows g and g + 8 of each 64-value block, one MMA a byte pair m of the sub-block:
// its low nibbles (values 2m, 2m + 1) in k slots 2t, 2t + 1, its high nibbles (values 2m + 8, 2m + 9) in slots 2t + 8, 2t + 9.
// B: the token as two FP16 limbs, hi (column 0) + lo (column 1) = x; code*scale is exact in FP16, the sums are FP32.
template <bool glu, int SPLIT>
static __global__ void __launch_bounds__(SPLIT*WARP_SIZE)
k_nvfp4_moe_mma(const char * __restrict__ w, const size_t nb01, const size_t nb02, const int32_t * __restrict__ ids,
        const float * __restrict__ x, const int64_t x_stride, const float * __restrict__ scale,
        float * __restrict__ dst, const int64_t dst_stride, const int K, const int n_ff, const float beta, const float lb) {
#ifdef AMPERE_MMA_AVAILABLE
    extern __shared__ half xh[]; // [2][K]: hi, lo
    __shared__ float2 part[SPLIT][8];

    const int lane  = threadIdx.x % WARP_SIZE;
    const int split = threadIdx.x / WARP_SIZE;
    const int g     = lane / 4;
    const int t     = lane % 4;
    const int slot  = blockIdx.y;

    ggml_cuda_pdl_sync();

    const int    e  = ids[slot];
    const int    r  = blockIdx.x*(glu ? 8 : 16) + g;
    const char * a  = w + e*nb02 + r*nb01;
    const char * b  = glu ? a + n_ff*nb01 : a + 8*nb01;
    const int    nk = K/64;

    // the first weights in flight while the block converts the token
    nvfp4_mma_step cur = nvfp4_mma_load(a, b, split, t);

    const float4 * xs4 = (const float4 *) (x + slot*x_stride);
    for (int i = threadIdx.x; i < K/4; i += blockDim.x) {
        const float4 v4 = xs4[i];
        const float  v[4] = { v4.x, v4.y, v4.z, v4.w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const half hi = __float2half(v[j]);
            xh[4*i + j]     = hi;
            xh[K + 4*i + j] = __float2half(v[j] - __half2float(hi));
        }
    }
    __syncthreads();

    const half * xc = xh + (g == 1 ? K : 0) + 16*t;

    float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int k = split; k < nk; k += SPLIT) {
        // the next step's loads before this step's math
        nvfp4_mma_step nxt = cur;
        if (k + SPLIT < nk) {
            nxt = nvfp4_mma_load(a, b, k + SPLIT, t);
        }
        const half2 sa2 = __float2half2_rn(ggml_cuda_ue4m3_to_fp32((cur.sa >> (8*t)) & 0xFF));
        const half2 sb2 = __float2half2_rn(ggml_cuda_ue4m3_to_fp32((cur.sb >> (8*t)) & 0xFF));
        const uint32_t sa = *(const uint32_t *) &sa2;
        const uint32_t sb = *(const uint32_t *) &sb2;
        const uint32_t wa[2] = { cur.a0, cur.a1 };
        const uint32_t wb[2] = { cur.b0, cur.b1 };
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m/2] >> (16*(m%2));
            const uint32_t qb = wb[m/2] >> (16*(m%2));
            const uint32_t a0 = nvfp4_pair_h2(qa,      sa);
            const uint32_t a1 = nvfp4_pair_h2(qb,      sb);
            const uint32_t a2 = nvfp4_pair_h2(qa >> 4, sa);
            const uint32_t a3 = nvfp4_pair_h2(qb >> 4, sb);
            const uint32_t x0 = *(const uint32_t *) (xc + 64*k + 2*m);
            const uint32_t x1 = *(const uint32_t *) (xc + 64*k + 2*m + 8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(x0), "r"(x1));
        }
        cur = nxt;
    }

    // lanes t == 0 hold columns 0 (hi) and 1 (lo) of rows g and g + 8
    if (t == 0) {
        part[split][g] = make_float2(v[0] + v[1], v[2] + v[3]);
    }
    __syncthreads();
    if (split != 0 || t != 0) {
        return;
    }
    float2 tot = part[0][g];
#pragma unroll
    for (int i = 1; i < SPLIT; ++i) {
        tot.x += part[i][g].x;
        tot.y += part[i][g].y;
    }
    const float sc = scale ? scale[e] : 1.0f;
    if constexpr (glu) {
        const float y = tot.x*sc;
        const float u = tot.y*sc;
        dst[slot*dst_stride + r] = tanhf(y*(1.0f/beta))*beta*(1.0f/(1.0f + expf(-y))) * (lb > 0.0f ? tanhf(u*(1.0f/lb))*lb : u);
    } else {
        dst[slot*dst_stride + r]     = tot.x*sc;
        dst[slot*dst_stride + r + 8] = tot.y*sc;
    }
#else
    GGML_UNUSED_VARS(w, nb01, nb02, ids, x, x_stride, scale, dst, dst_stride, K, n_ff, beta, lb);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

bool ggml_cuda_nvfp4_moe_vec_supported(const ggml_tensor * mm) {
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * x   = mm->src[1];
    const ggml_tensor * ids = mm->src[2];
    return mm->op == GGML_OP_MUL_MAT_ID && w->type == GGML_TYPE_NVFP4 && x->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
        w->ne[0] % 64 == 0 && w->nb[1] % 4 == 0 && w->nb[2] % 4 == 0 && w->ne[3] == 1 &&
        x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) && (x->ne[1] == 1 || x->nb[1] % 16 == 0) &&
        ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && ggml_is_contiguous(ids) && ggml_is_contiguous(mm) &&
        (uintptr_t) x->data % 16 == 0 && (uintptr_t) w->data % 4 == 0;
}

void ggml_cuda_nvfp4_moe_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * scale,
        const ggml_tensor * glu, ggml_tensor * dst, const bool copy_x) {
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * x   = mm->src[1];
    const ggml_tensor * ids = mm->src[2];

    GGML_ASSERT(ggml_cuda_nvfp4_moe_vec_supported(mm) && ggml_is_contiguous(dst));
    GGML_ASSERT(!scale || (scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && ggml_nelements(scale) == w->ne[2]));

    const int n_slots = ids->ne[0];
    const int n_rows  = w->ne[1];
    const int n_ff    = glu ? n_rows/2 : 0;
    const int n_out   = glu ? n_ff : n_rows;

    GGML_ASSERT(dst->ne[0] == n_out && dst->ne[1] == n_slots && n_slots <= UINT16_MAX);

    const int rows_per_block = NVFP4_MOE_GROUPS*NVFP4_MOE_WARPS*NVFP4_MOE_ROWS;
    const dim3 blocks_num((n_out + rows_per_block - 1)/rows_per_block, n_slots, 1);
    const dim3 block_dims(NVFP4_MOE_WARPS*WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};

    const int64_t x_stride   = x->ne[1] == 1 ? 0 : x->nb[1]/sizeof(float);
    const int64_t dst_stride = dst->nb[1]/sizeof(float);
    const float   beta       = glu ? ggml_get_op_params_f32(glu, 2) : 0.0f;
    const float   lb         = glu ? ggml_get_op_params_f32(glu, 3) : 0.0f;

    // dst overlaps x: the blocks read a copy of x
    ggml_cuda_pool_alloc<char> x_copy(ctx.pool());
    const float * x_d = (const float *) x->data;
    if (copy_x) {
        CUDA_CHECK(cudaMemcpyAsync(x_copy.alloc(ggml_nbytes(x)), x->data, ggml_nbytes(x), cudaMemcpyDeviceToDevice, ctx.stream()));
        x_d = (const float *) x_copy.get();
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int K  = w->ne[0];
    if (ampere_mma_available(cc) && n_out % (glu ? 8 : 16) == 0 && 2*K*sizeof(half) <= 48*1024) {
        const int split = glu ? NVFP4_MMA_SPLIT_GLU : NVFP4_MMA_SPLIT_DOWN;
        const ggml_cuda_kernel_launch_params mma_params = ggml_cuda_kernel_launch_params{
            dim3(n_out/(glu ? 8 : 16), n_slots, 1), dim3(split*WARP_SIZE, 1, 1), 2*K*sizeof(half), ctx.stream()};
        if (glu) {
            ggml_cuda_kernel_launch(k_nvfp4_moe_mma<true, NVFP4_MMA_SPLIT_GLU>, mma_params,
                (const char *) w->data, w->nb[1], w->nb[2], (const int32_t *) ids->data, x_d, x_stride,
                scale ? (const float *) scale->data : nullptr, (float *) dst->data, dst_stride, K, n_ff, beta, lb);
        } else {
            ggml_cuda_kernel_launch(k_nvfp4_moe_mma<false, NVFP4_MMA_SPLIT_DOWN>, mma_params,
                (const char *) w->data, w->nb[1], w->nb[2], (const int32_t *) ids->data, x_d, x_stride,
                scale ? (const float *) scale->data : nullptr, (float *) dst->data, dst_stride, K, n_ff, beta, lb);
        }
        return;
    }

    if (glu) {
        ggml_cuda_kernel_launch(k_nvfp4_moe_vec<true>, launch_params,
            (const char *) w->data, w->nb[1], w->nb[2], (const int32_t *) ids->data, x_d, x_stride,
            scale ? (const float *) scale->data : nullptr, (float *) dst->data, dst_stride, (int) (w->ne[0]/16), n_out, n_ff, beta, lb);
    } else {
        ggml_cuda_kernel_launch(k_nvfp4_moe_vec<false>, launch_params,
            (const char *) w->data, w->nb[1], w->nb[2], (const int32_t *) ids->data, x_d, x_stride,
            scale ? (const float *) scale->data : nullptr, (float *) dst->data, dst_stride, (int) (w->ne[0]/16), n_out, n_ff, beta, lb);
    }
}
