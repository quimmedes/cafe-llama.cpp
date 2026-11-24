#include "inkling.cuh"
#include "convert.cuh"
#include "fwht.cuh"

#define STATE_MIX_MAX_STATES (GGML_MAX_SRC - 1)

struct state_mix_params {
    const float * s[STATE_MIX_MAX_STATES];
    int64_t       s_stride[STATE_MIX_MAX_STATES]; // floats from a token to the next
    const float * q;
    float       * dst;
    int64_t       dst_stride_q;
    int64_t       dst_stride_t;
    int           n;
    int           n_q;
    int           n_states;
    float         scale;
    float         eps;
    // one token: query i also as rms_norm(.)*norm_w[i] to norm_dst[i] (if set), and its q8_1 row of norm_q8_n blocks to norm_q8[i] (if set)
    float       * norm_dst[STATE_MIX_MAX_Q];
    const float * norm_w[STATE_MIX_MAX_Q];
    block_q8_1  * norm_q8[STATE_MIX_MAX_Q];
    int           norm_q8_n;
    float         norm_eps;
};

#define STATE_MIX_NV        (STATE_MIX_MAX_STATES*(1 + STATE_MIX_MAX_Q))
#define STATE_MIX_BLOCK     256

// block (t, split): the sums of squares and the dot products of a token over one channel slice
static __global__ void k_state_mix_partial(const state_mix_params p, float * partials) {
    constexpr int MS = STATE_MIX_MAX_STATES;
    constexpr int MQ = STATE_MIX_MAX_Q;
    constexpr int NV = STATE_MIX_NV;
    constexpr int n_warps = STATE_MIX_BLOCK/WARP_SIZE;

    __shared__ float partial[NV][n_warps];

    const int t   = blockIdx.x;
    const int c   = blockIdx.y*STATE_MIX_BLOCK + threadIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    float acc[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        acc[v] = 0.0f;
    }
    if (c < p.n) {
        float qv[MQ];
#pragma unroll
        for (int i = 0; i < MQ; ++i) {
            qv[i] = i < p.n_q ? p.q[i*p.n + c] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            if (j < p.n_states) {
                const float x = p.s[j][t*p.s_stride[j] + c];
                acc[j] = x*x;
#pragma unroll
                for (int i = 0; i < MQ; ++i) {
                    acc[MS + j*MQ + i] = x*qv[i];
                }
            }
        }
    }

    const int warp = tid/WARP_SIZE;
    const int lane = tid%WARP_SIZE;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        const float r = warp_reduce_sum(acc[v]);
        if (lane == 0) {
            partial[v][warp] = r;
        }
    }
    __syncthreads();

    if (tid < NV) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < n_warps; ++w) {
            sum += partial[tid][w];
        }
        partials[((int64_t) t*gridDim.y + blockIdx.y)*NV + tid] = sum;
    }
}

// block (t, split): the softmax weights of a token from all slices' partials, then the weighted sum over one channel slice
static __global__ void k_state_mix_out(const state_mix_params p, const float * partials, const int n_split) {
    constexpr int MS = STATE_MIX_MAX_STATES;
    constexpr int MQ = STATE_MIX_MAX_Q;
    constexpr int NV = STATE_MIX_NV;

    __shared__ float weight[MQ][MS];

    const int t   = blockIdx.x;
    const int c   = blockIdx.y*STATE_MIX_BLOCK + threadIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    if (tid < p.n_q) {
        const int i = tid;
        const float * pt = partials + (int64_t) t*n_split*NV;
        float rms[MS];
        float score[MS];
        float max = -INFINITY;
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            float ss  = 0.0f;
            float dot = 0.0f;
            if (j < p.n_states) {
                for (int k = 0; k < n_split; ++k) {
                    ss  += pt[k*NV + j];
                    dot += pt[k*NV + MS + j*MQ + i];
                }
            }
            rms[j]   = rsqrtf(ss/p.n + p.eps);
            score[j] = j < p.n_states ? p.scale*rms[j]*dot : -INFINITY;
            max      = fmaxf(max, score[j]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            score[j] = j < p.n_states ? expf(score[j] - max) : 0.0f;
            sum     += score[j];
        }
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            weight[i][j] = score[j]/sum*rms[j];
        }
    }
    __syncthreads();

    if (c >= p.n) {
        return;
    }
    float x[MS];
#pragma unroll
    for (int j = 0; j < MS; ++j) {
        x[j] = j < p.n_states ? p.s[j][t*p.s_stride[j] + c] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < MQ; ++i) {
        if (i < p.n_q) {
            float y = 0.0f;
#pragma unroll
            for (int j = 0; j < MS; ++j) {
                if (j < p.n_states) {
                    y += weight[i][j]*x[j];
                }
            }
            p.dst[t*p.dst_stride_t + i*p.dst_stride_q + c] = y;
        }
    }
}

#define STATE_MIX_SINGLE_BLOCK 512
#define STATE_MIX_SINGLE_CPT   5
#define STATE_MIX_MULTI_MAX    12 // mixes of one launch, the params of all of them fit in 4 KB

struct state_mix_multi {
    state_mix_params m[STATE_MIX_MULTI_MAX];
};
static_assert(sizeof(state_mix_multi) <= 4096, "kernel params of at most 4 KB");

// block (t, split, mix) for a few tokens: each block sums over all channels itself, then writes its own channel slice, in one launch
template <int NQ>
static __global__ void k_state_mix_single(const state_mix_multi mp) {
    constexpr int MS      = STATE_MIX_MAX_STATES;
    constexpr int NV      = MS*(1 + NQ);
    constexpr int n_warps = STATE_MIX_SINGLE_BLOCK/WARP_SIZE;

    __shared__ float partial[NV][n_warps];
    __shared__ float weight[NQ][MS];

    const state_mix_params & p = mp.m[blockIdx.z];

    const int t   = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    float acc[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        acc[v] = 0.0f;
    }
    // all channels of a thread at once (n <= STATE_MIX_SINGLE_CPT*STATE_MIX_SINGLE_BLOCK), so that its loads are in flight together
    float qv[STATE_MIX_SINGLE_CPT][NQ];
#pragma unroll
    for (int k = 0; k < STATE_MIX_SINGLE_CPT; ++k) {
        const int c = tid + k*STATE_MIX_SINGLE_BLOCK;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            qv[k][i] = c < p.n ? p.q[i*p.n + c] : 0.0f;
        }
    }
    // the values of all states stay in registers for the outputs
    float xs[MS][STATE_MIX_SINGLE_CPT];
#pragma unroll
    for (int j = 0; j < MS; ++j) {
#pragma unroll
        for (int k = 0; k < STATE_MIX_SINGLE_CPT; ++k) {
            const int c = tid + k*STATE_MIX_SINGLE_BLOCK;
            xs[j][k] = j < p.n_states && c < p.n ? p.s[j][t*p.s_stride[j] + c] : 0.0f;
        }
    }
#pragma unroll
    for (int j = 0; j < MS; ++j) {
#pragma unroll
        for (int k = 0; k < STATE_MIX_SINGLE_CPT; ++k) {
            acc[j] += xs[j][k]*xs[j][k];
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                acc[MS + j*NQ + i] += xs[j][k]*qv[k][i];
            }
        }
    }

    const int warp = tid/WARP_SIZE;
    const int lane = tid%WARP_SIZE;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        const float r = warp_reduce_sum(acc[v]);
        if (lane == 0) {
            partial[v][warp] = r;
        }
    }
    __syncthreads();

    if (tid < NV) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < n_warps; ++w) {
            sum += partial[tid][w];
        }
        partial[tid][0] = sum;
    }
    __syncthreads();

    if (tid < NQ) {
        const int i = tid;
        float rms[MS];
        float score[MS];
        float max = -INFINITY;
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            rms[j]   = rsqrtf(partial[j][0]/p.n + p.eps);
            score[j] = j < p.n_states ? p.scale*rms[j]*partial[MS + j*NQ + i][0] : -INFINITY;
            max      = fmaxf(max, score[j]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            score[j] = j < p.n_states ? expf(score[j] - max) : 0.0f;
            sum     += score[j];
        }
#pragma unroll
        for (int j = 0; j < MS; ++j) {
            weight[i][j] = score[j]/sum*rms[j];
        }
    }
    __syncthreads();

    // the outputs of a thread's channels: all of them with one block, else the one of this block
    float y[STATE_MIX_SINGLE_CPT][NQ];
#pragma unroll
    for (int k = 0; k < STATE_MIX_SINGLE_CPT; ++k) {
        const int  c    = tid + k*STATE_MIX_SINGLE_BLOCK;
        const bool mine = gridDim.y == 1 || k == (int) blockIdx.y;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            y[k][i] = 0.0f;
#pragma unroll
            for (int j = 0; j < MS; ++j) {
                if (j < p.n_states) {
                    y[k][i] += weight[i][j]*xs[j][k];
                }
            }
            if (mine && c < p.n) {
                p.dst[t*p.dst_stride_t + i*p.dst_stride_q + c] = y[k][i];
            }
        }
    }

    if (!p.norm_dst[0]) {
        return;
    }

    // the rms norms, one token: each block sums the squares of each query output over all channels
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < STATE_MIX_SINGLE_CPT; ++k) {
            ss += y[k][i]*y[k][i];
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) {
            partial[i][warp] = ss;
        }
    }
    __syncthreads();
    if (tid < NQ) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < n_warps; ++w) {
            sum += partial[tid][w];
        }
        weight[tid][0] = rsqrtf(sum/p.n + p.norm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        if (!p.norm_dst[i]) {
            continue;
        }
        const float f = weight[i][0];
#pragma unroll
        for (int k = 0; k < STATE_MIX_SINGLE_CPT; ++k) {
            // each block writes its own channels of the norms, all blocks have all the sums
            if (gridDim.y > 1 && k != (int) blockIdx.y) {
                continue;
            }
            const int c = k*STATE_MIX_SINGLE_BLOCK + tid;
            const float v = c < p.n ? f*y[k][i]*p.norm_w[i][c] : 0.0f;
            if (c < p.n) {
                p.norm_dst[i][c] = v;
            }
            if (p.norm_q8[i] && k*STATE_MIX_SINGLE_BLOCK < p.n) {
                // as quantize_q8_1, a warp a block of 32 channels (n % 32 == 0)
                const float  amax = warp_reduce_max(fabsf(v));
                const float  d    = amax / 127.0f;
                const int8_t q    = amax == 0.0f ? 0 : roundf(v / d);
                const int    sum  = warp_reduce_sum((int) q);
                if (c < p.n) {
                    block_q8_1 * b = p.norm_q8[i] + c/QK8_1;
                    b->qs[lane] = q;
                    if (lane == 0) {
                        b->ds = make_half2(d, d * sum);
                    }
                }
            }
        }
        if (p.norm_q8[i] && blockIdx.y == 0) {
            for (int b = p.n/QK8_1 + tid; b < p.norm_q8_n; b += STATE_MIX_SINGLE_BLOCK) {
                p.norm_q8[i][b].ds = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int j = 0; j < QK8_1/4; ++j) {
                    ((int *) p.norm_q8[i][b].qs)[j] = 0;
                }
            }
        }
    }
}

static state_mix_params state_mix_params_of(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];

    state_mix_params p = {};
    p.n_states = 0;
    while (p.n_states < STATE_MIX_MAX_STATES && dst->src[1 + p.n_states]) {
        const ggml_tensor * s = dst->src[1 + p.n_states];
        p.s[p.n_states]        = (const float *) s->data;
        p.s_stride[p.n_states] = s->nb[1]/sizeof(float);
        p.n_states++;
    }
    p.q            = (const float *) q->data;
    p.dst          = (float *) dst->data;
    p.dst_stride_q = dst->nb[1]/sizeof(float);
    p.dst_stride_t = dst->nb[2]/sizeof(float);
    p.n            = q->ne[0];
    p.n_q          = q->ne[1];
    p.scale        = ggml_get_op_params_f32(dst, 0);
    p.eps          = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(p.n_q >= 1 && p.n_q <= STATE_MIX_MAX_Q);
    return p;
}

// n_mix mixes of the same number of queries in one launch
static void state_mix_single_launch(ggml_backend_cuda_context & ctx, const state_mix_params * ps, const int n_mix, const int n_tokens, const int n_split) {
    GGML_ASSERT(n_split <= UINT16_MAX && n_mix >= 1 && n_mix <= STATE_MIX_MULTI_MAX);
    state_mix_multi mp = {};
    for (int i = 0; i < n_mix; ++i) {
        GGML_ASSERT(ps[i].n_q == ps[0].n_q);
        mp.m[i] = ps[i];
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{
        dim3(n_tokens, n_split, n_mix), dim3(STATE_MIX_SINGLE_BLOCK, 1, 1), 0, ctx.stream()};
    switch (ps[0].n_q) {
        case 1: ggml_cuda_kernel_launch(k_state_mix_single<1>, launch_params, mp); break;
        case 2: ggml_cuda_kernel_launch(k_state_mix_single<2>, launch_params, mp); break;
        case 3: ggml_cuda_kernel_launch(k_state_mix_single<3>, launch_params, mp); break;
        case 4: ggml_cuda_kernel_launch(k_state_mix_single<4>, launch_params, mp); break;
        default: GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_op_state_mix_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mixes, const int n) {
    if (n < 2 || n > STATE_MIX_MULTI_MAX) {
        return false;
    }
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    state_mix_params ps[STATE_MIX_MULTI_MAX];
    for (int i = 0; i < n; ++i) {
        ps[i] = state_mix_params_of(mixes[i]);
        if (mixes[i]->ne[2] != 1 || ps[i].n_q != ps[0].n_q || ps[i].n != ps[0].n || ps[i].n > STATE_MIX_SINGLE_CPT*STATE_MIX_SINGLE_BLOCK) {
            return false;
        }
        // the mixes run at once: no output overlaps an input of another one
        for (int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            for (int s = 0; s < GGML_MAX_SRC && mixes[j]->src[s]; ++s) {
                if (overlap(mixes[i], mixes[j]->src[s])) {
                    return false;
                }
            }
            if (j < i && overlap(mixes[i], mixes[j])) {
                return false;
            }
        }
    }
    state_mix_single_launch(ctx, ps, n, 1, (ps[0].n + STATE_MIX_SINGLE_BLOCK - 1)/STATE_MIX_SINGLE_BLOCK);
    return true;
}

void ggml_cuda_op_state_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const state_mix_params p = state_mix_params_of(dst);

    const int64_t n_tokens = dst->ne[2];
    GGML_ASSERT(n_tokens <= INT32_MAX);

    // a few tokens: one launch, each block repeats the sums over all channels (they are in L2)
    if (n_tokens <= 8 && p.n <= STATE_MIX_SINGLE_CPT*STATE_MIX_SINGLE_BLOCK) {
        state_mix_single_launch(ctx, &p, 1, n_tokens, (p.n + STATE_MIX_SINGLE_BLOCK - 1)/STATE_MIX_SINGLE_BLOCK);
        return;
    }

    const int n_split = (p.n + STATE_MIX_BLOCK - 1)/STATE_MIX_BLOCK;
    GGML_ASSERT(n_split <= UINT16_MAX);

    ggml_cuda_pool_alloc<float> partials(ctx.pool(), n_tokens*n_split*STATE_MIX_NV);

    const dim3 blocks_num(n_tokens, n_split, 1);
    const dim3 block_dims(STATE_MIX_BLOCK, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};
    ggml_cuda_kernel_launch(k_state_mix_partial, launch_params, p, partials.get());
    ggml_cuda_kernel_launch(k_state_mix_out,     launch_params, p, (const float *) partials.get(), n_split);
}

bool ggml_cuda_op_state_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * mix, const ggml_tensor * const * norms, ggml_tensor * const * muls, const int n_norm) {
    state_mix_params p = state_mix_params_of(mix);
    if (mix->ne[2] != 1 || p.n > STATE_MIX_SINGLE_CPT*STATE_MIX_SINGLE_BLOCK || p.n % QK8_1 != 0 || n_norm < 1 || n_norm > p.n_q) {
        return false;
    }
    bool * quantized[STATE_MIX_MAX_Q] = {};
    for (int k = 0; k < n_norm; ++k) {
        // rms_norm of the view of query i, times a weight of n values
        const ggml_tensor * x   = norms[k]->src[0];
        const ggml_tensor * mul = muls[k];
        const ggml_tensor * w   = mul->src[0] == norms[k] ? mul->src[1] : mul->src[0];
        if (x->view_src != mix || x->ne[0] != p.n || ggml_nelements(x) != p.n || x->view_offs % (p.dst_stride_q*sizeof(float)) != 0 ||
                w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != p.n ||
                mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || ggml_nelements(mul) != p.n) {
            return false;
        }
        const int i = x->view_offs/(p.dst_stride_q*sizeof(float));
        if (i >= p.n_q || p.norm_dst[i] || (k > 0 && ggml_get_op_params_f32(norms[k], 0) != p.norm_eps)) {
            return false;
        }
        p.norm_dst[i] = (float *) mul->data;
        p.norm_w[i]   = (const float *) w->data;
        p.norm_eps    = ggml_get_op_params_f32(norms[k], 0);
        // the q8_1 of the mat-vecs that read the norm
        const auto q8 = ctx.mmvq_q8_of ? ctx.mmvq_q8_of(mul) : ggml_backend_cuda_context::mmvq_shared_q8_1{};
        if (q8.data && ggml_nrows(q8.src1) == 1 && ggml_nelements(q8.src1) == p.n) {
            p.norm_q8[i] = (block_q8_1 *) q8.data;
            p.norm_q8_n  = GGML_PAD(p.n, MATRIX_ROW_PADDING)/QK8_1;
            quantized[i] = q8.quantized;
        }
    }
    // the kernel checks query 0 for the norms
    if (!p.norm_dst[0]) {
        return false;
    }

    // the norm outputs must not overlap the inputs or the mix output
    auto overlap = [](const char * a0, size_t na, const char * b0, size_t nb) {
        return a0 < b0 + nb && b0 < a0 + na;
    };
    for (int k = 0; k < n_norm; ++k) {
        const char * o = (const char *) muls[k]->data;
        const size_t no = ggml_nbytes(muls[k]);
        if (overlap(o, no, (const char *) mix->data, ggml_nbytes(mix)) || overlap(o, no, (const char *) mix->src[0]->data, ggml_nbytes(mix->src[0]))) {
            return false;
        }
        for (int j = 0; j < p.n_states; ++j) {
            if (overlap(o, no, (const char *) mix->src[1 + j]->data, ggml_nbytes(mix->src[1 + j]))) {
                return false;
            }
        }
        for (int j = 0; j < k; ++j) {
            if (overlap(o, no, (const char *) muls[j]->data, ggml_nbytes(muls[j]))) {
                return false;
            }
        }
    }

    state_mix_single_launch(ctx, &p, 1, 1, (p.n + STATE_MIX_SINGLE_BLOCK - 1)/STATE_MIX_SINGLE_BLOCK);
    for (bool * q : quantized) {
        if (q) {
            *q = true;
        }
    }
    return true;
}

// x - tanh(f)*x + tanh(g)*m, one block a row
static __global__ void k_loop_gate(
        const float * x, const float * m, const float * g, const float * f, float * dst, const int ne0, const int ne1, const int ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3, const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3, const int64_t sf1, const int64_t sf2, const int64_t sf3) {
    const int64_t r  = blockIdx.x;
    const int64_t i1 = r % ne1;
    const int64_t i2 = (r/ne1) % ne2;
    const int64_t i3 = r/(ne1*ne2);

    ggml_cuda_pdl_sync();

    const float * xr = x + i1*sx1 + i2*sx2 + i3*sx3;
    const float * mr = m + i1*sm1 + i2*sm2 + i3*sm3;
    const float * gr = g + i1*sg1 + i2*sg2 + i3*sg3;
    const float * fr = f + i1*sf1 + i2*sf2 + i3*sf3;
    float       * y  = dst + r*ne0;

    for (int c = threadIdx.x; c < ne0; c += blockDim.x) {
        const float xi = xr[c];
        y[c] = (xi - tanhf(fr[c])*xi) + tanhf(gr[c])*mr[c];
    }
}

void ggml_cuda_op_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * m = dst->src[1];
    const ggml_tensor * g = dst->src[2];
    const ggml_tensor * f = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t nrows = ggml_nrows(dst);
    const dim3 blocks_num(nrows, 1, 1);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};
    ggml_cuda_kernel_launch(k_loop_gate, launch_params,
        (const float *) x->data, (const float *) m->data, (const float *) g->data, (const float *) f->data, (float *) dst->data,
        (int) dst->ne[0], (int) dst->ne[1], (int) dst->ne[2],
        x->nb[1]/4, x->nb[2]/4, x->nb[3]/4, m->nb[1]/4, m->nb[2]/4, m->nb[3]/4,
        g->nb[1]/4, g->nb[2]/4, g->nb[3]/4, f->nb[1]/4, f->nb[2]/4, f->nb[3]/4);
}

#define LOOP_CONTROL_BLOCK 256
#define LOOP_CONTROL_MAX   3

// the operands of a few loop controls of the same shape, control blockIdx.z (strides in floats)
struct loop_control_args {
    const float * x[LOOP_CONTROL_MAX];
    const float * m[LOOP_CONTROL_MAX];
    const float * wd[LOOP_CONTROL_MAX];
    const float * wg[LOOP_CONTROL_MAX];
    const float * wf[LOOP_CONTROL_MAX];
    float       * dst[LOOP_CONTROL_MAX];
    int64_t sx1[LOOP_CONTROL_MAX];
    int64_t sm1[LOOP_CONTROL_MAX];
    int64_t sd1[LOOP_CONTROL_MAX];
};

// block (k, t, control): z[t][k] = w_down[k]*[rms_norm(x_t); rms_norm(m_t)], the norms and the dot products in one pass
static __global__ void k_loop_control_down(const loop_control_args a, float * z, const int n, const int r, const int n_tokens, const float eps) {
    constexpr int n_warps = LOOP_CONTROL_BLOCK/WARP_SIZE;
    __shared__ float partial[4][n_warps];

    const int k   = blockIdx.x;
    const int t   = blockIdx.y;
    const int i   = blockIdx.z;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const float * xr = a.x[i] + t*a.sx1[i];
    const float * mr = a.m[i] + t*a.sm1[i];
    const float * w  = a.wd[i] + (int64_t) k*2*n;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int c = tid; c < n; c += LOOP_CONTROL_BLOCK) {
        const float xv = xr[c];
        const float mv = mr[c];
        acc[0] += xv*xv;
        acc[1] += mv*mv;
        acc[2] += w[c]*xv;
        acc[3] += w[n + c]*mv;
    }

    const int warp = tid/WARP_SIZE;
    const int lane = tid%WARP_SIZE;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const float s = warp_reduce_sum(acc[v]);
        if (lane == 0) {
            partial[v][warp] = s;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int v = 0; v < 4; ++v) {
#pragma unroll
            for (int w = 0; w < n_warps; ++w) {
                s[v] += partial[v][w];
            }
        }
        z[((int64_t) i*n_tokens + t)*r + k] = s[2]*rsqrtf(s[0]/n + eps) + s[3]*rsqrtf(s[1]/n + eps);
    }
}

// block (channel group, t, control), one warp a channel: g = w_gain*z, f = w_forget*z, then the gated update
static __global__ void k_loop_control_out(const loop_control_args a, const float * z, const int n, const int r, const int n_tokens) {
    extern __shared__ float zs[];

    const int t    = blockIdx.y;
    const int i    = blockIdx.z;
    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;
    const int c    = blockIdx.x*(LOOP_CONTROL_BLOCK/WARP_SIZE) + warp;

    ggml_cuda_pdl_sync();

    for (int k = threadIdx.x; k < r; k += LOOP_CONTROL_BLOCK) {
        zs[k] = z[((int64_t) i*n_tokens + t)*r + k];
    }
    __syncthreads();

    if (c >= n) {
        return;
    }

    const float * g = a.wg[i] + (int64_t) c*r;
    const float * f = a.wf[i] + (int64_t) c*r;
    float gs = 0.0f;
    float fs = 0.0f;
    for (int k = lane; k < r; k += WARP_SIZE) {
        gs += g[k]*zs[k];
        fs += f[k]*zs[k];
    }
    gs = warp_reduce_sum(gs);
    fs = warp_reduce_sum(fs);

    if (lane == 0) {
        const float xc = a.x[i][t*a.sx1[i] + c];
        a.dst[i][t*a.sd1[i] + c] = (xc - tanhf(fs)*xc) + tanhf(gs)*a.m[i][t*a.sm1[i] + c];
    }
}

static void loop_control_launch(ggml_backend_cuda_context & ctx, ggml_tensor * const * dst, const int n_ctrl) {
    GGML_ASSERT(n_ctrl >= 1 && n_ctrl <= LOOP_CONTROL_MAX);

    const int     n        = dst[0]->ne[0];
    const int64_t n_tokens = dst[0]->ne[1];
    const int     r        = dst[0]->src[2]->ne[1];
    const float   eps      = ggml_get_op_params_f32(dst[0], 0);

    GGML_ASSERT(n_tokens <= UINT16_MAX);

    loop_control_args a = {};
    for (int i = 0; i < n_ctrl; ++i) {
        a.x[i]   = (const float *) dst[i]->src[0]->data;
        a.m[i]   = (const float *) dst[i]->src[1]->data;
        a.wd[i]  = (const float *) dst[i]->src[2]->data;
        a.wg[i]  = (const float *) dst[i]->src[3]->data;
        a.wf[i]  = (const float *) dst[i]->src[4]->data;
        a.dst[i] = (float *) dst[i]->data;
        a.sx1[i] = dst[i]->src[0]->nb[1]/4;
        a.sm1[i] = dst[i]->src[1]->nb[1]/4;
        a.sd1[i] = dst[i]->nb[1]/4;
    }

    ggml_cuda_pool_alloc<float> z(ctx.pool(), n_ctrl*n_tokens*r);

    const dim3 block_dims(LOOP_CONTROL_BLOCK, 1, 1);
    const ggml_cuda_kernel_launch_params down_params = ggml_cuda_kernel_launch_params{dim3(r, n_tokens, n_ctrl), block_dims, 0, ctx.stream()};
    ggml_cuda_kernel_launch(k_loop_control_down, down_params, a, z.get(), n, r, (int) n_tokens, eps);

    constexpr int n_warps = LOOP_CONTROL_BLOCK/WARP_SIZE;
    const ggml_cuda_kernel_launch_params out_params =
        ggml_cuda_kernel_launch_params{dim3((n + n_warps - 1)/n_warps, n_tokens, n_ctrl), block_dims, r*sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(k_loop_control_out, out_params, a, (const float *) z.get(), n, r, (int) n_tokens);
}

void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    loop_control_launch(ctx, &dst, 1);
}

bool ggml_cuda_op_loop_control_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * dst, const int n_ctrl) {
    if (n_ctrl > LOOP_CONTROL_MAX) {
        return false;
    }
    for (int i = 1; i < n_ctrl; ++i) {
        if (!ggml_are_same_shape(dst[i], dst[0]) || dst[i]->src[2]->ne[1] != dst[0]->src[2]->ne[1] ||
                ggml_get_op_params_f32(dst[i], 0) != ggml_get_op_params_f32(dst[0], 0)) {
            return false;
        }
    }
    loop_control_launch(ctx, dst, n_ctrl);
    return true;
}

// the conv input of a sequence as two strided parts: the K - 1 earlier columns, then the new ones (strides in floats)
struct short_conv_src {
    const float * past;
    const float * cur;
    int64_t p0, p1, p2;
    int64_t c0, c1, c2;
    int     n_past;

    __device__ float operator()(const int col, const int ch, const int s) const {
        return col < n_past ? past[col*p0 + ch*p1 + s*p2] : cur[(col - n_past)*c0 + ch*c1 + s*c2];
    }
};

// one thread an output: scale*(x_t + sum_j c[j]*x[t + j] + sum_lag d[lag]*x[t + K - 1 - lag]), plus res[i] if any
// threads past the outputs copy the last K - 1 inputs of each sequence to tail (if any), [(K - 1)*n_c, n_s]
static __global__ void k_short_conv(
        const short_conv_src x, const float * c, const float * d, const float * res, float * dst, float * tail, const int64_t tail_s1,
        const int n_conv, const int n_c, const int n_t, const int n_s, const int64_t sd1, const int head_c, const float scale) {
    int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();

    const int64_t n_out = (int64_t) n_c*n_t*n_s;
    if (i >= n_out) {
        i -= n_out;
        if (!tail || i >= (int64_t) (n_conv - 1)*n_c*n_s) {
            return;
        }
        const int j  = i % (n_conv - 1);
        const int ch = (i/(n_conv - 1)) % n_c;
        const int s  = i/((int64_t) (n_conv - 1)*n_c);
        tail[s*tail_s1 + (int64_t) ch*(n_conv - 1) + j] = x(n_t + j, ch, s);
        return;
    }
    const int ch = i % n_c;
    const int t  = (i/n_c) % n_t;
    const int s  = i/((int64_t) n_c*n_t);

    const float * w = c + ch*n_conv;

    float out = 0.0f;
    for (int j = 0; j < n_conv; ++j) {
        out += x(t + j, ch, s)*w[j];
    }
    if (d) {
        const float * tap = d + (int64_t) (s*n_t + t)*sd1 + (ch/head_c)*n_conv;
        for (int lag = 0; lag < n_conv; ++lag) {
            out += x(t + n_conv - 1 - lag, ch, s)*tap[lag];
        }
    }
    const float y = (x(t + n_conv - 1, ch, s) + out)*scale;
    dst[i] = res ? res[i] + y : y;
}

static void short_conv_launch(ggml_backend_cuda_context & ctx, const short_conv_src & x, ggml_tensor * dst, float * tail, const int64_t tail_s1,
        const float * res = nullptr, float * out = nullptr) {
    const ggml_tensor * c = dst->src[1];
    const ggml_tensor * d = dst->src[2];

    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(c));

    const int n_conv  = c->ne[0];
    const int n_c     = dst->ne[0];
    const int n_t     = dst->ne[1];
    const int n_s     = dst->ne[2];
    const int n_heads = d ? d->ne[0]/n_conv : 1;

    const int64_t n = (int64_t) n_c*n_t*n_s + (tail ? (int64_t) (n_conv - 1)*n_c*n_s : 0);
    const dim3 blocks_num((n + 255)/256, 1, 1);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};
    ggml_cuda_kernel_launch(k_short_conv, launch_params,
        x, (const float *) c->data, d ? (const float *) d->data : nullptr, res, out ? out : (float *) dst->data, tail, tail_s1,
        n_conv, n_c, n_t, n_s, d ? d->nb[1]/4 : 0, n_c/n_heads, ggml_get_op_params_f32(dst, 0));
}

static short_conv_src short_conv_src_of(const ggml_tensor * past, const ggml_tensor * cur) {
    short_conv_src x;
    x.past   = (const float *) past->data;
    x.cur    = (const float *) cur->data;
    x.p0     = past->nb[0]/4;
    x.p1     = past->nb[1]/4;
    x.p2     = past->nb[2]/4;
    x.c0     = cur->nb[0]/4;
    x.c1     = cur->nb[1]/4;
    x.c2     = cur->nb[2]/4;
    x.n_past = past->ne[0];
    return x;
}

void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    short_conv_launch(ctx, short_conv_src_of(dst->src[0], dst->src[3]), dst, nullptr, 0);
}

// the concat must join the inputs of the conv, the cpy must store its last K - 1 columns [K - 1, n_c, n_s] to F32 cells [(K - 1)*n_c, n_s]
static bool short_conv_concat_ok(const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * dst) {
    const ggml_tensor * last = cpy->src[0];
    const ggml_tensor * cell = cpy->src[1];

    const int64_t n_past = dst->src[0]->ne[0];
    const int64_t n_c    = dst->ne[0];
    const int64_t n_t    = dst->ne[1];
    const int64_t n_s    = dst->ne[2];

    return concat->src[0] == dst->src[0] && concat->src[1] == dst->src[3] && cell->type == GGML_TYPE_F32 &&
        last->view_src == concat && last->ne[0] == n_past && last->ne[1] == n_c && last->ne[2] == n_s &&
        last->nb[1] == concat->nb[1] && last->nb[2] == concat->nb[2] && last->view_offs == (size_t) n_t*sizeof(float) &&
        cell->nb[0] == sizeof(float) && cell->ne[0] == n_past*n_c && cell->ne[1] == n_s && cell->ne[2] == 1 && cell->ne[3] == 1;
}

bool ggml_cuda_op_short_conv_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy, ggml_tensor * dst,
        ggml_tensor * add) {
    if (!short_conv_concat_ok(concat, cpy, dst)) {
        return false;
    }
    const ggml_tensor * cell = cpy->src[1];

    // add: dst + res, written to add, the output of the conv itself is not stored
    const ggml_tensor * res = nullptr;
    if (add) {
        res = add->src[0] == dst || add->src[0]->view_src == dst ? add->src[1] : add->src[0];
        if (add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || ggml_nelements(add) != ggml_nelements(dst) ||
                res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != ggml_nelements(dst)) {
            return false;
        }
    }

    short_conv_launch(ctx, short_conv_src_of(dst->src[0], dst->src[3]), dst, (float *) cell->data, cell->nb[1]/4,
        res ? (const float *) res->data : nullptr, add ? (float *) add->data : nullptr);
    return true;
}

#define SHORT_CONV_JOB_K 4 // the conv width of a short_conv_job

// a short conv of one token of one sequence (width SHORT_CONV_JOB_K), its tail stored to the state
struct short_conv_job {
    short_conv_src x;
    const float  * c;
    const float  * d;
    float        * dst;
    float        * tail;
    int            head_c;
    float          scale;
};

// the inputs of channel ch, no stores: a lane loads all its channels before it stores any
struct short_conv_in {
    float  x[SHORT_CONV_JOB_K];
    float4 w;
    float4 tap;
};

static __device__ __forceinline__ short_conv_in short_conv_load(const short_conv_job & j, const int ch) {
    static_assert(SHORT_CONV_JOB_K == 4, "the weights are a float4");
    short_conv_in in;
#pragma unroll
    for (int k = 0; k < SHORT_CONV_JOB_K; ++k) {
        in.x[k] = j.x(k, ch, 0);
    }
    in.w   = *(const float4 *) (j.c + ch*SHORT_CONV_JOB_K);
    in.tap = j.d ? *(const float4 *) (j.d + (ch/j.head_c)*SHORT_CONV_JOB_K) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return in;
}

// channel ch as k_short_conv makes it, also stores the tail of ch
static __device__ __forceinline__ float short_conv_store(const short_conv_job & j, const int ch, const short_conv_in & in) {
    const float w[4]   = { in.w.x,   in.w.y,   in.w.z,   in.w.w   };
    const float tap[4] = { in.tap.x, in.tap.y, in.tap.z, in.tap.w };

    float out = 0.0f;
#pragma unroll
    for (int k = 0; k < SHORT_CONV_JOB_K; ++k) {
        out += in.x[k]*w[k];
    }
    if (j.d) {
#pragma unroll
        for (int lag = 0; lag < SHORT_CONV_JOB_K; ++lag) {
            out += in.x[SHORT_CONV_JOB_K - 1 - lag]*tap[lag];
        }
    }
    const float y = (in.x[SHORT_CONV_JOB_K - 1] + out)*j.scale;

#pragma unroll
    for (int k = 0; k < SHORT_CONV_JOB_K - 1; ++k) {
        j.tail[(int64_t) ch*(SHORT_CONV_JOB_K - 1) + k] = in.x[1 + k];
    }
    if (j.dst) {
        j.dst[ch] = y;
    }
    return y;
}

static bool short_conv_job_plan(const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * dst, short_conv_job & j) {
    const ggml_tensor * c = dst->src[1];
    const ggml_tensor * d = dst->src[2];
    if (!short_conv_concat_ok(concat, cpy, dst) || dst->ne[1] != 1 || dst->ne[2] != 1 || !ggml_is_contiguous(dst) || !ggml_is_contiguous(c) ||
            c->ne[0] != SHORT_CONV_JOB_K || (uintptr_t) c->data % 16 != 0 || (d && ((uintptr_t) d->data % 16 != 0 || d->nb[0] != sizeof(float)))) {
        return false;
    }
    j = {};
    j.x      = short_conv_src_of(dst->src[0], dst->src[3]);
    j.c      = (const float *) c->data;
    j.d      = d ? (const float *) d->data : nullptr;
    j.dst    = (float *) dst->data;
    j.tail   = (float *) cpy->src[1]->data;
    j.head_c = dst->ne[0]/(d ? d->ne[0]/SHORT_CONV_JOB_K : 1);
    j.scale  = ggml_get_op_params_f32(dst, 0);
    return true;
}

// one token: the weighted sum of the expert outputs (as moe_weighted_reduction_f32) -> short conv (tail to the state) -> + res
static __global__ void k_moe_conv(const short_conv_job j, const float * __restrict__ experts, const float * __restrict__ scale,
        const float * __restrict__ weights, const int n_used, const float * res, float * dst, const int n) {
    const int ch = blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();

    if (ch >= n) {
        return;
    }

    short_conv_in in;
#pragma unroll
    for (int k = 0; k < SHORT_CONV_JOB_K - 1; ++k) {
        in.x[k] = j.x(k, ch, 0);
    }
    in.w   = *(const float4 *) (j.c + ch*SHORT_CONV_JOB_K);
    in.tap = j.d ? *(const float4 *) (j.d + (ch/j.head_c)*SHORT_CONV_JOB_K) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float sum = (experts[ch]*(scale ? scale[0] : 1.0f))*weights[0];
    for (int e = 1; e < n_used; ++e) {
        sum += (experts[(int64_t) e*n + ch]*(scale ? scale[e] : 1.0f))*weights[e];
    }
    in.x[SHORT_CONV_JOB_K - 1] = sum;

    const float y = short_conv_store(j, ch, in);
    dst[ch] = res[ch] + y;
}

bool ggml_cuda_op_moe_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * expert_scale, const ggml_tensor * weights,
        const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * conv, ggml_tensor * add, const ggml_tensor * res) {
    short_conv_job j;
    if (!short_conv_job_plan(concat, cpy, conv, j)) {
        return false;
    }
    const int64_t n      = conv->ne[0];
    const int64_t n_used = experts->ne[1];
    if (experts->type != GGML_TYPE_F32 || !ggml_is_contiguous(experts) || experts->ne[0] != n || ggml_nelements(experts) != n*n_used ||
            weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) || ggml_nelements(weights) != n_used ||
            (expert_scale && (expert_scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(expert_scale) || ggml_nelements(expert_scale) != n_used)) ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || ggml_nelements(add) != n ||
            res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != n) {
        return false;
    }

    // the outputs (the add, the tail) must not overlap the inputs, but for the residual read in place
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    const ggml_tensor * cell = cpy->src[1];
    for (const ggml_tensor * w : { (const ggml_tensor *) add, cell }) {
        for (const ggml_tensor * r : std::initializer_list<const ggml_tensor *>{ experts, expert_scale, weights, conv->src[0], conv->src[2], res }) {
            if (r && overlap(w, r) && !(w == add && r == res && add->data == res->data)) {
                return false;
            }
        }
    }
    if (overlap(add, cell)) {
        return false;
    }

    j.dst = nullptr;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{dim3((n + 255)/256, 1, 1), dim3(256, 1, 1), 0, ctx.stream()};
    ggml_cuda_kernel_launch(k_moe_conv, launch_params, j, (const float *) experts->data, expert_scale ? (const float *) expert_scale->data : nullptr,
        (const float *) weights->data, (int) n_used, (const float *) res->data, (float *) add->data, (int) n);
    return true;
}

// the clip of row (head, token) of r (times the factor of the token), dot with each row of the bias
struct rel_table_params {
    const float * r;
    const float * s;
    const float * tok;
    const float * bias;
    float       * dst;
    int           s_rows;
    int           n_rel;
    int           n_head;
    int           n_tokens;
    int           n_extent;
    float         threshold;
    float         eps;
};

// the bias rows chunk*nthreads + tid of (h, t), rc: shared memory of 2*WARP_SIZE floats
static __device__ __forceinline__ void rel_table_block(const rel_table_params & p, const int chunk, const int h, const int t, const int tid,
        const int nthreads, float * rc) {
    if (tid < WARP_SIZE) {
        const float * rr = p.r + ((int64_t) t*p.n_head + h)*p.n_rel;
        const float * sr = p.s ? p.s + (int64_t) (h % p.s_rows)*p.n_rel : nullptr;

        float v[2];
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int k = tid + i*WARP_SIZE;
            v[i] = k < p.n_rel ? (sr ? rr[k]*sr[k] : rr[k]) : 0.0f;
            sum += v[i]*v[i];
        }
        sum = warp_reduce_sum(sum);

        const float f = fminf(fmaxf(p.threshold/(sqrtf(sum) + p.eps), 0.0f), 1.0f);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int k = tid + i*WARP_SIZE;
            if (k < p.n_rel) {
                rc[k] = p.tok ? v[i]*f*p.tok[t] : v[i]*f;
            }
        }
    }
    __syncthreads();

    const int d = chunk*nthreads + tid;
    if (d >= p.n_extent) {
        return;
    }
    const float * b = p.bias + (int64_t) d*p.n_rel;
    float acc = 0.0f;
    if (p.n_rel % 4 == 0) {
        for (int k = 0; k < p.n_rel; k += 4) {
            const float4 b4 = *(const float4 *) (b + k);
            acc += b4.x*rc[k] + b4.y*rc[k + 1] + b4.z*rc[k + 2] + b4.w*rc[k + 3];
        }
    } else {
        for (int k = 0; k < p.n_rel; ++k) {
            acc += b[k]*rc[k];
        }
    }
    p.dst[((int64_t) t*p.n_head + h)*p.n_extent + d] = acc;
}

// block (extent chunk, head, token)
static __global__ void k_rel_table(const rel_table_params p) {
    __shared__ float rc[2*WARP_SIZE];

    ggml_cuda_pdl_sync();

    rel_table_block(p, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, blockDim.x, rc);
}

static bool rel_table_plan(const ggml_tensor * clip, const ggml_tensor * tok, const ggml_tensor * mm, rel_table_params & p) {
    const ggml_tensor * r    = clip->src[0];
    const ggml_tensor * s    = clip->src[1];
    const ggml_tensor * bias = mm->src[0];

    const int64_t n_rel    = r->ne[0];
    const int64_t n_head   = r->ne[1];
    const int64_t n_tokens = r->ne[2];
    const int64_t n_extent = bias->ne[1];

    if (n_rel > 2*WARP_SIZE || n_tokens > 8 || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || r->ne[3] != 1 ||
            bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) || bias->ne[0] != n_rel || bias->ne[2] != 1 || bias->ne[3] != 1 ||
            mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || mm->ne[0] != n_extent || mm->ne[1] != n_head || mm->ne[2] != n_tokens) {
        return false;
    }
    if (s && (s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) || s->ne[0] != n_rel || s->ne[2] != 1 || s->ne[3] != 1)) {
        return false;
    }
    const ggml_tensor * f = nullptr;
    if (tok) {
        // the factor of a token: a [1, 1, n_tokens] operand of a MUL
        f = tok->src[1]->ne[0] == 1 ? tok->src[1] : tok->src[0];
        if (f->type != GGML_TYPE_F32 || !ggml_is_contiguous(f) || ggml_nelements(f) != n_tokens || f->ne[0] != 1 || f->ne[1] != 1) {
            return false;
        }
    }

    p = {};
    p.r         = (const float *) r->data;
    p.s         = s ? (const float *) s->data : nullptr;
    p.tok       = f ? (const float *) f->data : nullptr;
    p.bias      = (const float *) bias->data;
    p.dst       = (float *) mm->data;
    p.s_rows    = s ? s->ne[1] : 1;
    p.n_rel     = n_rel;
    p.n_head    = n_head;
    p.n_tokens  = n_tokens;
    p.n_extent  = n_extent;
    p.threshold = ggml_get_op_params_f32(clip, 0);
    p.eps       = ggml_get_op_params_f32(clip, 1);
    return true;
}

bool ggml_cuda_op_rel_table(ggml_backend_cuda_context & ctx, const ggml_tensor * clip, const ggml_tensor * tok, ggml_tensor * mm) {
    rel_table_params p;
    if (!rel_table_plan(clip, tok, mm, p)) {
        return false;
    }
    const dim3 blocks_num((p.n_extent + 255)/256, p.n_head, p.n_tokens);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};
    ggml_cuda_kernel_launch(k_rel_table, launch_params, p);
    return true;
}

#define ATTN_PREP_MAX_ROWS 4
#define ATTN_PREP_MAX_CONV 2

// one launch for the attention inputs of a token: rows of 128 (fwht_rows_row, the K and V ones from their conv) and the rel table
struct attn_prep_params {
    fwht_rows_params f[ATTN_PREP_MAX_ROWS];
    int              f_conv[ATTN_PREP_MAX_ROWS];       // the conv of the values of row job f, or -1: the values are in f.x
    int              f_block0[ATTN_PREP_MAX_ROWS + 1]; // the first block of row job f, f_block0[n_f]: the blocks of all row jobs
    int              n_f;
    short_conv_job   conv[ATTN_PREP_MAX_CONV];
    rel_table_params tab;
    int              tab_chunks; // blocks of 4*WARP_SIZE bias rows a (head, token), 0: no table
    // a small matvec of one token, a warp a row: mv_dst[r] = mv_w row r (BF16 or F32, mv_s1 values apart) . mv_x
    const void     * mv_w;
    const float    * mv_x;
    float          * mv_dst;
    int64_t          mv_s1;
    int              mv_bf16;
    int              mv_rows; // 0: no matvec
    int              mv_k;
};

static __global__ void __launch_bounds__(4*WARP_SIZE) k_attn_prep(const attn_prep_params p) {
    __shared__ float rc[2*WARP_SIZE];

    const int lane = threadIdx.x;
    const int b    = blockIdx.x;

    ggml_cuda_pdl_sync();

    if (b < p.f_block0[p.n_f]) {
        int f = 0;
        while (b >= p.f_block0[f + 1]) {
            ++f;
        }
        const fwht_rows_params & q = p.f[f];
        const int r = (b - p.f_block0[f])*4 + threadIdx.y;
        if (r >= q.n_rows) {
            return;
        }
        float reg[FWHT_ROWS_EL];
        if (p.f_conv[f] >= 0) {
            const short_conv_job & j = p.conv[p.f_conv[f]];
            short_conv_in in[FWHT_ROWS_EL];
#pragma unroll
            for (int i = 0; i < FWHT_ROWS_EL; ++i) {
                in[i] = short_conv_load(j, r*128 + i*WARP_SIZE + lane);
            }
#pragma unroll
            for (int i = 0; i < FWHT_ROWS_EL; ++i) {
                reg[i] = short_conv_store(j, r*128 + i*WARP_SIZE + lane, in[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < FWHT_ROWS_EL; ++i) {
                reg[i] = q.x[(int64_t) r*128 + i*WARP_SIZE + lane];
            }
        }
        fwht_rows_row(q, r, lane, reg);
        return;
    }

    int tb = b - p.f_block0[p.n_f];
    if (tb < p.tab_chunks*p.tab.n_head) {
        const int chunk = tb % p.tab_chunks;
        tb /= p.tab_chunks;
        rel_table_block(p.tab, chunk, tb % p.tab.n_head, tb / p.tab.n_head, threadIdx.y*WARP_SIZE + lane, 4*WARP_SIZE, rc);
        return;
    }

    const int r = (tb - p.tab_chunks*p.tab.n_head)*4 + threadIdx.y;
    if (r >= p.mv_rows) {
        return;
    }
    float sum = 0.0f;
    if (p.mv_bf16) {
        const nv_bfloat162 * w2 = (const nv_bfloat162 *) ((const nv_bfloat16 *) p.mv_w + r*p.mv_s1);
        const float2       * x2 = (const float2 *) p.mv_x;
        for (int k = lane; k < p.mv_k/2; k += WARP_SIZE) {
            const float2 wf = __bfloat1622float2(w2[k]);
            const float2 xf = x2[k];
            sum += wf.x*xf.x + wf.y*xf.y;
        }
    } else {
        const float * wr = (const float *) p.mv_w + r*p.mv_s1;
        for (int k = lane; k < p.mv_k; k += WARP_SIZE) {
            sum += wr[k]*p.mv_x[k];
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        p.mv_dst[r] = sum;
    }
}

bool ggml_cuda_op_attn_prep(ggml_backend_cuda_context & ctx, const ggml_cuda_attn_prep_group * groups, const int n_groups) {
    attn_prep_params p = {};
    const ggml_tensor * conv_dst[ATTN_PREP_MAX_CONV] = {};
    bool conv_used[ATTN_PREP_MAX_CONV] = {};
    int  n_conv = 0;
    bool has_tab = false;

    // the tensors each group reads and writes, for the overlap check
    std::vector<std::pair<int, const ggml_tensor *>> reads;
    std::vector<std::pair<int, const ggml_tensor *>> writes;

    for (int g = 0; g < n_groups; ++g) {
        const ggml_cuda_attn_prep_group & gr = groups[g];
        if (gr.kind == GGML_CUDA_ATTN_PREP_CONV) {
            if (n_conv == ATTN_PREP_MAX_CONV || !short_conv_job_plan(gr.a, gr.b, gr.c, p.conv[n_conv])) {
                return false;
            }
            conv_dst[n_conv] = gr.c;
            if (gr.internal) {
                p.conv[n_conv].dst = nullptr;
            } else {
                writes.push_back({ g, gr.c });
            }
            n_conv++;
            for (int s : { 0, 2, 3 }) {
                if (gr.c->src[s]) {
                    reads.push_back({ g, gr.c->src[s] });
                }
            }
            writes.push_back({ g, gr.b->src[1] });
        } else if (gr.kind == GGML_CUDA_ATTN_PREP_ROWS) {
            if (p.n_f == ATTN_PREP_MAX_ROWS) {
                return false;
            }
            fwht_rows_params & q = p.f[p.n_f];
            if (!ggml_cuda_fwht_rows_plan(gr.a, gr.b, gr.c, gr.d, q)) {
                return false;
            }
            const ggml_tensor * x = gr.a->src[0];
            if (x->ne[2] != 1) {
                return false;
            }
            // the values of a conv of the prep: made in the same warp
            p.f_conv[p.n_f] = -1;
            for (int k = 0; k < n_conv; ++k) {
                if (x == conv_dst[k] || x->view_src == conv_dst[k]) {
                    if (conv_used[k] || x->view_offs != 0 || ggml_nelements(x) != ggml_nelements(conv_dst[k])) {
                        return false;
                    }
                    conv_used[k]    = true;
                    p.f_conv[p.n_f] = k;
                }
            }
            if (p.f_conv[p.n_f] < 0) {
                reads.push_back({ g, x });
            }
            if (gr.a->src[1]) {
                reads.push_back({ g, gr.a->src[1] });
            }
            if (gr.b) {
                reads.push_back({ g, gr.b->src[1]->ne[0] == 1 ? gr.b->src[1] : gr.b->src[0] });
            }
            if (gr.d) {
                reads.push_back({ g, gr.d->src[1] });
            }
            writes.push_back({ g, gr.d ? gr.d : gr.c });

            const int n_blocks = (q.n_rows + 3)/4;
            p.f_block0[p.n_f + 1] = p.f_block0[p.n_f] + n_blocks;
            p.n_f++;
        } else if (gr.kind == GGML_CUDA_ATTN_PREP_MATVEC) {
            const ggml_tensor * mm = gr.a;
            const ggml_tensor * w  = mm->src[0];
            const ggml_tensor * x  = mm->src[1];
            if (p.mv_rows || (w->type != GGML_TYPE_BF16 && w->type != GGML_TYPE_F32) || w->nb[0] != ggml_type_size(w->type) || w->ne[2] != 1 || w->ne[3] != 1 ||
                    w->nb[1] % 4 != 0 || (uintptr_t) w->data % 4 != 0 || w->ne[0] % 2 != 0 || w->ne[1] > 64 ||
                    x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nelements(x) != w->ne[0] || (uintptr_t) x->data % 8 != 0 ||
                    mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || ggml_nelements(mm) != w->ne[1]) {
                return false;
            }
            p.mv_w    = w->data;
            p.mv_x    = (const float *) x->data;
            p.mv_dst  = (float *) mm->data;
            p.mv_s1   = w->nb[1]/ggml_type_size(w->type);
            p.mv_bf16 = w->type == GGML_TYPE_BF16;
            p.mv_rows = w->ne[1];
            p.mv_k    = w->ne[0];
            reads.push_back({ g, x });
            writes.push_back({ g, mm });
        } else {
            if (has_tab || !rel_table_plan(gr.a, gr.b, gr.c, p.tab) || p.tab.n_tokens != 1) {
                return false;
            }
            has_tab = true;
            reads.push_back({ g, gr.a->src[0] });
            if (gr.a->src[1]) {
                reads.push_back({ g, gr.a->src[1] });
            }
            if (gr.b) {
                reads.push_back({ g, gr.b->src[1]->ne[0] == 1 ? gr.b->src[1] : gr.b->src[0] });
            }
            writes.push_back({ g, gr.c });
        }
    }
    for (int k = 0; k < n_conv; ++k) {
        if (!conv_used[k]) {
            return false;
        }
    }

    // no group may write what another group reads; a row job may write its own input in place
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    for (const auto & [gw, w] : writes) {
        for (const auto & [gr, r] : reads) {
            const bool in_place = gw == gr && groups[gw].kind == GGML_CUDA_ATTN_PREP_ROWS && w->data == r->data && ggml_nbytes(w) == ggml_nbytes(r);
            if (overlap(w, r) && !in_place) {
                return false;
            }
        }
    }
    for (size_t a = 0; a < writes.size(); ++a) {
        for (size_t b = a + 1; b < writes.size(); ++b) {
            if (overlap(writes[a].second, writes[b].second)) {
                return false;
            }
        }
    }

    p.tab_chunks = has_tab ? (p.tab.n_extent + 4*WARP_SIZE - 1)/(4*WARP_SIZE) : 0;
    const int n_blocks = p.f_block0[p.n_f] + (has_tab ? p.tab_chunks*p.tab.n_head : 0) + (p.mv_rows + 3)/4;

    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dim3(n_blocks, 1, 1), block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_attn_prep, launch_params, p);
    return true;
}

// block (kv chunk, t, h): mask + bias of the distance - sink, token t is row t % n_rows of stream t / n_rows
template <typename T>
static __global__ void k_rel_mask(
        const float * table, const int32_t * dist, const float * sinks, const T * mask, T * dst, const int n_kv, const int n_rows, const int n_extent, const int h0,
        const int64_t st1, const int64_t st2, const int64_t sd1, const int64_t ss1, const int64_t sm1, const int64_t sm3,
        const int64_t sy1, const int64_t sy2, const int64_t sy3) {
    const int kv = blockIdx.x*blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    const int h  = blockIdx.z;
    const int r  = t % n_rows;
    const int s  = t / n_rows;

    ggml_cuda_pdl_sync();

    if (kv >= n_kv) {
        return;
    }
    const float * tab  = table + (h0 + h)*st1 + t*st2;
    const float   sink = sinks ? sinks[h0 + h + t*ss1] : 0.0f;
    const int32_t d    = dist[t*sd1 + kv];
    const float   b    = (d >= 0 && d < n_extent ? tab[d] : 0.0f) - sink;

    dst[r*sy1 + h*sy2 + s*sy3 + kv] = ggml_cuda_cast<T>(ggml_cuda_cast<float>(mask[r*sm1 + s*sm3 + kv]) + b);
}

void ggml_cuda_op_rel_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * dist  = dst->src[1];
    const ggml_tensor * sinks = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int n_kv     = dst->ne[0];
    const int n_rows   = dst->ne[1];
    const int nh       = dst->ne[2];
    const int n_tokens = n_rows*dst->ne[3];
    const int h0       = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(n_tokens <= UINT16_MAX && nh <= UINT16_MAX);

    const size_t ts = ggml_type_size(dst->type);
    const dim3 blocks_num((n_kv + 255)/256, n_tokens, nh);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};
    auto launch = [&](auto type_tag) {
        using T = decltype(type_tag);
        ggml_cuda_kernel_launch(k_rel_mask<T>, launch_params,
            (const float *) table->data, (const int32_t *) dist->data, sinks ? (const float *) sinks->data : nullptr,
            (const T *) mask->data, (T *) dst->data, n_kv, n_rows, (int) table->ne[0], h0,
            table->nb[1]/4, table->nb[2]/4, dist->nb[1]/4, sinks ? sinks->nb[1]/4 : 0, mask->nb[1]/ts, mask->nb[3]/ts,
            dst->nb[1]/ts, dst->nb[2]/ts, dst->nb[3]/ts);
    };
    if (dst->type == GGML_TYPE_F16) {
        launch(half());
    } else {
        launch(float());
    }
}
