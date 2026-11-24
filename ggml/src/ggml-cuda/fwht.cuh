#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

enum fwht_rows_pre {
    FWHT_ROWS_PRE_NONE,
    FWHT_ROWS_PRE_L2_CLIP,
    FWHT_ROWS_PRE_RMS_NORM,
};

// rows of 128 values, one warp a row: an optional row norm, an optional factor of the token, the Walsh-Hadamard transform
// of fwht_cuda on blocks of n_rot values, then either F32 rows or q8_0 rows of a set_rows destination (row t of the tokens to row idx[t])
struct fwht_rows_params {
    const float * x;
    const float * s;      // L2 clip scale, [128, s_rows], or nullptr
    const float * tok;    // a factor of each token, or nullptr
    float       * dst_f;  // F32 rows, or nullptr
    const float * gate;   // multiply the F32 rows by sigmoid(gate), or nullptr
    block_q8_1  * dst_q8_1; // the F32 rows also as q8_1 of n_q8_1 blocks (zero blocks past the rows), or nullptr
    int           n_q8_1;
    char        * dst_q;  // q8_0 rows, or nullptr
    const void  * idx;
    int64_t       dst_nb1;
    int           idx_i64;
    int           n_head;
    int           s_rows;
    int           n_rows;
    int           n_rot;  // the transform size, 64 or 128: one or two transforms a row
    int           pre;    // fwht_rows_pre
    float         p0;     // L2 clip threshold
    float         eps;
    float         scale;
};

#define FWHT_ROWS_EL (128/WARP_SIZE) // values of a row in each lane: value i*WARP_SIZE + lane in register i

// pre (L2_CLIP or RMS_NORM of [128, n_head, n_tokens]) -> tok (optional MUL by a factor of each token) -> mm (Hadamard MUL_MAT)
// -> set_rows (optional, q8_0): one kernel, false if the nodes do not fit
bool ggml_cuda_fwht_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * pre, const ggml_tensor * tok,
        const ggml_tensor * mm, const ggml_tensor * set_rows);

// mm (Hadamard MUL_MAT of rows of 128) -> mul (MUL by sig, a SIGMOID): one kernel, false if the nodes do not fit
bool ggml_cuda_fwht_rows_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig, ggml_tensor * mul);

// row r of p from its raw values (reg[i] = x[i*WARP_SIZE + lane]): the steps after the load of k_fwht_rows
static __device__ __forceinline__ void fwht_rows_row(const fwht_rows_params & p, const int r, const int lane, float (&reg)[FWHT_ROWS_EL]) {
    constexpr int N    = 128;
    constexpr int el_w = FWHT_ROWS_EL;

    const int h = r % p.n_head;
    const int t = r / p.n_head;

    if (p.s) {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            reg[i] *= p.s[(h % p.s_rows)*N + i*WARP_SIZE + lane];
        }
    }

    if (p.pre != FWHT_ROWS_PRE_NONE) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            sum += reg[i]*reg[i];
        }
        sum = warp_reduce_sum(sum);
        const float f = p.pre == FWHT_ROWS_PRE_L2_CLIP ? fminf(fmaxf(p.p0/(sqrtf(sum) + p.eps), 0.0f), 1.0f) : rsqrtf(sum/N + p.eps);
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            reg[i] = p.pre == FWHT_ROWS_PRE_L2_CLIP ? reg[i]*f : f*reg[i];
        }
    }
    if (p.tok) {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            reg[i] *= p.tok[t];
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] *= p.scale;
    }

#pragma unroll
    for (int hh = 1; hh < WARP_SIZE; hh *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, val, hh, WARP_SIZE);
            reg[j] = (lane & hh) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int hh = WARP_SIZE; hh < p.n_rot; hh *= 2) {
        const int step = hh / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float a = reg[j + k];
                const float b = reg[j + k + step];
                reg[j + k]        = a + b;
                reg[j + k + step] = a - b;
            }
        }
    }

    if (p.dst_f) {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            const int64_t k = (int64_t) r*N + i*WARP_SIZE + lane;
            const float   v = p.gate ? reg[i]*(1.0f/(1.0f + expf(-p.gate[k]))) : reg[i];
            p.dst_f[k] = v;
            if (p.dst_q8_1) {
                // as quantize_q8_1
                const float  amax = warp_reduce_max(fabsf(v));
                const float  d    = amax / 127.0f;
                const int8_t q    = amax == 0.0f ? 0 : roundf(v / d);
                const int    sum  = warp_reduce_sum((int) q);
                block_q8_1 * y    = p.dst_q8_1 + (int64_t) r*(N/QK8_1) + i;
                y->qs[lane] = q;
                if (lane == 0) {
                    y->ds = make_half2(d, d * sum);
                }
            }
        }
        if (p.dst_q8_1 && r == 0) {
            for (int b = p.n_rows*(N/QK8_1) + lane; b < p.n_q8_1; b += WARP_SIZE) {
                p.dst_q8_1[b].ds = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int j = 0; j < QK8_1/4; ++j) {
                    ((int *) p.dst_q8_1[b].qs)[j] = 0;
                }
            }
        }
        return;
    }

    // q8_0 block i of the head is register i of all lanes, as quantize_f32_q8_0_block
    const int64_t row = p.idx_i64 ? ((const int64_t *) p.idx)[t] : ((const int32_t *) p.idx)[t];
    block_q8_0 * y = (block_q8_0 *) (p.dst_q + row*p.dst_nb1) + h*(N/QK8_0);
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const float amax = warp_reduce_max(fabsf(reg[i]));
        const float d    = amax / ((1 << 7) - 1);
        const float id   = d ? 1.0f/d : 0.0f;
        y[i].qs[lane] = roundf(reg[i]*id);
        if (lane == 0) {
            y[i].d = d;
        }
    }
}

// the params of ggml_cuda_fwht_rows_fused, false if the nodes do not fit
bool ggml_cuda_fwht_rows_plan(const ggml_tensor * pre, const ggml_tensor * tok, const ggml_tensor * mm, const ggml_tensor * set_rows,
        fwht_rows_params & p);
