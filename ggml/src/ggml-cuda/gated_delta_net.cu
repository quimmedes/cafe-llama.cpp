#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

static __global__ void gdn_precompute_exp(const float * g, float * g_exp, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x*gridDim.x) {
        g_exp[i] = expf(g[i]);
    }
}

constexpr int gdn_cols_per_warp = 4;

// RAW: beta and g arrive pre-activation (ggml_gated_delta_net_set_raw_gates); the kernel applies
// sigmoid(beta) and raw_a[h] * softplus(g + raw_dt_bias[h]) with the unary kernels' formulas.
// G_PRECOMPUTED: g already holds exp(g) (GB10 long-prompt path); only used with RAW == false.
template <int S_v, bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * raw_dt_bias,
                                     const float * raw_a,
                                     const float * curr_state,
                                     const int32_t * tree_parent,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    const float * s0_col = curr_state + state_in_offset + col * S_v;
    curr_state = s0_col;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        if constexpr (keep_rs_t) {
            // token tree: start from the state after the parent token, written by this thread
            if (tree_parent != nullptr && t > 0 && tree_parent[t] != t - 1) {
                const int p = tree_parent[t];
                const float * src = p < 0 ? s0_col : state + (int64_t) ((int) n_tokens - 1 - p) * state_slot_stride + col * S_v;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = src[r * lanes_per_col + lane_in_col];
                }
            }
        }

        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = *beta_t;
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
            float g0 = *g_t;
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h_idx];
                g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
            const float g_val = G_PRECOMPUTED ? g0 : expf(g0);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * rb_d, const float * ra_d, const float * s_d, const int32_t * tp_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, tp_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, tp_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, tp_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, tp_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const bool    raw  = ggml_get_op_params_i32(dst, 1) != 0;
    const float * rb_d = raw ? (const float *) dst->src[7]->data : nullptr;
    const float * ra_d = raw ? (const float *) dst->src[8]->data : nullptr;
    GGML_ASSERT(!(raw && kda)); // raw gates are defined for the scalar gate only

    const int32_t * tp_d = dst->src[9] ? (const int32_t *) dst->src[9]->data : nullptr;
    GGML_ASSERT(tp_d == nullptr || (keep_rs && K >= n_tokens) || n_tokens == 1);

    // the MMA path expects activated gates and a linear token order
    if (!raw && tp_d == nullptr) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }

    // GB10 long-prompt path: exp(g) once per (token, head) instead of once per column.
    // Only for activated gates; with raw gates (#165) the activation happens inside the kernel.
    ggml_cuda_pool_alloc<float> g_exp_alloc(ctx.pool());
    bool g_precomputed = false;
    if (!kda && !raw && S_v == 128 && n_tokens >= 32 &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_DGX_SPARK) {
        const int64_t n_g = ggml_nelements(src_g);
        g_exp_alloc.alloc(n_g);
        const int block = 256;
        const int grid = std::min<int64_t>((n_g + block - 1)/block, 4096);
        gdn_precompute_exp<<<grid, block, 0, stream>>>(g_d, g_exp_alloc.ptr, n_g);
        g_d = g_exp_alloc.ptr;
        g_precomputed = true;
    }

#define GDN_LAUNCH(KDA_, KEEP_, RAW_, PRE_)                                                       \
    launch_gated_delta_net<KDA_, KEEP_, RAW_, PRE_>(q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, tp_d, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                    \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream)
    if (kda) {
        if (keep_rs) { GDN_LAUNCH(true,  true,  false, false); } else { GDN_LAUNCH(true,  false, false, false); }
    } else if (raw) {
        if (keep_rs) { GDN_LAUNCH(false, true,  true,  false); } else { GDN_LAUNCH(false, false, true,  false); }
    } else {
        if (g_precomputed) {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, true);  } else { GDN_LAUNCH(false, false, false, true);  }
        } else {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, false); } else { GDN_LAUNCH(false, false, false, false); }
        }
    }
#undef GDN_LAUNCH
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
