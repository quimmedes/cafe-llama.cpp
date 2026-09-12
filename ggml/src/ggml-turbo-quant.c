/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit) and GGML_TYPE_TURBO4_0 (4-bit)
 * for use as --cache-type-k turbo3 --cache-type-v turbo3 in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#if defined(_WIN32)
#define _USE_MATH_DEFINES // for M_PI
#endif 

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* 4-bit: Lloyd-Max for N(0, 1/sqrt(128)), 16 centroids */
static const float CENTROIDS_4BIT[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static const float MIDPOINTS_4BIT[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

/* 8-bit: Lloyd-Max for N(0,1) scaled by 1/sqrt(128) (matches N(0,1/128) convention), 256 centroids */
static const float CENTROIDS_8BIT[256] = {
    -0.34189706f, -0.29884648f, -0.27157635f, -0.25121314f, -0.23484838f, -0.22114745f, -0.20937878f, -0.19909346f,
    -0.19000012f, -0.18188631f, -0.17459596f, -0.16801505f, -0.16205003f, -0.15662252f, -0.15166721f, -0.14712896f,
    -0.14295785f, -0.13910911f, -0.13554055f, -0.13222068f, -0.12912571f, -0.12622918f, -0.12350196f, -0.12092574f,
    -0.11848717f, -0.11616507f, -0.11394363f, -0.11181760f, -0.10977627f, -0.10780649f, -0.10590563f, -0.10406567f,
    -0.10227606f, -0.10053416f, -0.09883731f, -0.09718554f, -0.09557616f, -0.09400389f, -0.09246874f, -0.09096270f,
    -0.08947787f, -0.08801425f, -0.08657184f, -0.08515064f, -0.08375065f, -0.08237188f, -0.08101431f, -0.07967795f,
    -0.07836547f, -0.07707417f, -0.07580144f, -0.07454460f, -0.07330103f, -0.07206804f, -0.07084568f, -0.06963391f,
    -0.06843275f, -0.06724219f, -0.06606225f, -0.06489290f, -0.06373417f, -0.06258603f, -0.06144851f, -0.06032158f,
    -0.05920527f, -0.05810222f, -0.05700977f, -0.05592792f, -0.05485668f, -0.05379604f, -0.05274602f, -0.05170659f,
    -0.05067778f, -0.04965957f, -0.04864930f, -0.04764700f, -0.04664999f, -0.04565829f, -0.04467189f, -0.04369080f,
    -0.04271500f, -0.04174452f, -0.04077933f, -0.03981945f, -0.03886486f, -0.03791293f, -0.03696631f, -0.03602498f,
    -0.03508631f, -0.03415029f, -0.03321957f, -0.03229416f, -0.03137139f, -0.03045128f, -0.02953382f, -0.02861901f,
    -0.02770950f, -0.02680265f, -0.02589845f, -0.02499689f, -0.02409534f, -0.02319644f, -0.02230284f, -0.02141190f,
    -0.02052095f, -0.01963266f, -0.01874701f, -0.01786137f, -0.01697838f, -0.01609804f, -0.01522035f, -0.01434266f,
    -0.01346497f, -0.01258993f, -0.01171755f, -0.01084516f, -0.00997278f, -0.00910304f, -0.00823331f, -0.00736357f,
    -0.00649649f, -0.00562941f, -0.00476233f, -0.00389524f, -0.00302816f, -0.00216373f, -0.00129930f, -0.00043487f,
     0.00043222f,  0.00129930f,  0.00216373f,  0.00302816f,  0.00389524f,  0.00476233f,  0.00562941f,  0.00649649f,
     0.00736357f,  0.00823331f,  0.00910304f,  0.00997278f,  0.01084516f,  0.01171755f,  0.01258993f,  0.01346497f,
     0.01434266f,  0.01522035f,  0.01609804f,  0.01697838f,  0.01786137f,  0.01874701f,  0.01963266f,  0.02052095f,
     0.02141190f,  0.02230284f,  0.02319644f,  0.02409534f,  0.02499689f,  0.02589845f,  0.02680265f,  0.02770950f,
     0.02861901f,  0.02953382f,  0.03045128f,  0.03137139f,  0.03229416f,  0.03321957f,  0.03415029f,  0.03508631f,
     0.03602498f,  0.03696631f,  0.03791293f,  0.03886486f,  0.03981945f,  0.04077933f,  0.04174452f,  0.04271500f,
     0.04369080f,  0.04467189f,  0.04565829f,  0.04664999f,  0.04764700f,  0.04864930f,  0.04965957f,  0.05067778f,
     0.05170659f,  0.05274602f,  0.05379604f,  0.05485668f,  0.05592792f,  0.05700977f,  0.05810222f,  0.05920527f,
     0.06032158f,  0.06144851f,  0.06258603f,  0.06373417f,  0.06489290f,  0.06606225f,  0.06724219f,  0.06843275f,
     0.06963391f,  0.07084568f,  0.07206804f,  0.07330103f,  0.07454460f,  0.07580144f,  0.07707417f,  0.07836547f,
     0.07967795f,  0.08101431f,  0.08237188f,  0.08375065f,  0.08515064f,  0.08657184f,  0.08801425f,  0.08947787f,
     0.09096270f,  0.09246874f,  0.09400389f,  0.09557616f,  0.09718554f,  0.09883731f,  0.10053416f,  0.10227606f,
     0.10406567f,  0.10590563f,  0.10780649f,  0.10977627f,  0.11181760f,  0.11394363f,  0.11616507f,  0.11848717f,
     0.12092574f,  0.12350196f,  0.12622918f,  0.12912571f,  0.13222068f,  0.13554055f,  0.13910911f,  0.14295785f,
     0.14712896f,  0.15166721f,  0.15662252f,  0.16205003f,  0.16801505f,  0.17459596f,  0.18188631f,  0.19000012f,
     0.19909346f,  0.20937878f,  0.22114745f,  0.23484838f,  0.25121314f,  0.27157635f,  0.29884648f,  0.34189706f,
};
static const float MIDPOINTS_8BIT[255] = {
    -0.32037177f, -0.28521142f, -0.26139474f, -0.24303076f, -0.22799792f, -0.21526312f, -0.20423612f, -0.19454679f,
    -0.18594321f, -0.17824113f, -0.17130551f, -0.16503254f, -0.15933627f, -0.15414486f, -0.14939809f, -0.14504341f,
    -0.14103348f, -0.13732483f, -0.13388062f, -0.13067319f, -0.12767744f, -0.12486557f, -0.12221385f, -0.11970645f,
    -0.11732612f, -0.11505435f, -0.11288061f, -0.11079693f, -0.10879138f, -0.10685606f, -0.10498565f, -0.10317087f,
    -0.10140511f, -0.09968573f, -0.09801143f, -0.09638085f, -0.09479002f, -0.09323631f, -0.09171572f, -0.09022028f,
    -0.08874606f, -0.08729305f, -0.08586124f, -0.08445065f, -0.08306126f, -0.08169309f, -0.08034613f, -0.07902171f,
    -0.07771982f, -0.07643781f, -0.07517302f, -0.07392282f, -0.07268454f, -0.07145686f, -0.07023979f, -0.06903333f,
    -0.06783747f, -0.06665222f, -0.06547757f, -0.06431353f, -0.06316010f, -0.06201727f, -0.06088505f, -0.05976343f,
    -0.05865375f, -0.05755599f, -0.05646884f, -0.05539230f, -0.05432636f, -0.05327103f, -0.05222630f, -0.05119219f,
    -0.05016867f, -0.04915443f, -0.04814815f, -0.04714850f, -0.04615414f, -0.04516509f, -0.04418135f, -0.04320290f,
    -0.04222976f, -0.04126192f, -0.04029939f, -0.03934216f, -0.03838890f, -0.03743962f, -0.03649565f, -0.03555565f,
    -0.03461830f, -0.03368493f, -0.03275686f, -0.03183277f, -0.03091134f, -0.02999255f, -0.02907641f, -0.02816426f,
    -0.02725608f, -0.02635055f, -0.02544767f, -0.02454612f, -0.02364589f, -0.02274964f, -0.02185737f, -0.02096642f,
    -0.02007680f, -0.01918983f, -0.01830419f, -0.01741987f, -0.01653821f, -0.01565919f, -0.01478150f, -0.01390381f,
    -0.01302745f, -0.01215374f, -0.01128136f, -0.01040897f, -0.00953791f, -0.00866818f, -0.00779844f, -0.00693003f,
    -0.00606295f, -0.00519587f, -0.00432878f, -0.00346170f, -0.00259595f, -0.00173151f, -0.00086708f, -0.00000133f,
     0.00086576f,  0.00173151f,  0.00259595f,  0.00346170f,  0.00432878f,  0.00519587f,  0.00606295f,  0.00693003f,
     0.00779844f,  0.00866818f,  0.00953791f,  0.01040897f,  0.01128136f,  0.01215374f,  0.01302745f,  0.01390381f,
     0.01478150f,  0.01565919f,  0.01653821f,  0.01741987f,  0.01830419f,  0.01918983f,  0.02007680f,  0.02096642f,
     0.02185737f,  0.02274964f,  0.02364589f,  0.02454612f,  0.02544767f,  0.02635055f,  0.02725608f,  0.02816426f,
     0.02907641f,  0.02999255f,  0.03091134f,  0.03183277f,  0.03275686f,  0.03368493f,  0.03461830f,  0.03555565f,
     0.03649565f,  0.03743962f,  0.03838890f,  0.03934216f,  0.04029939f,  0.04126192f,  0.04222976f,  0.04320290f,
     0.04418135f,  0.04516509f,  0.04615414f,  0.04714850f,  0.04814815f,  0.04915443f,  0.05016867f,  0.05119219f,
     0.05222630f,  0.05327103f,  0.05432636f,  0.05539230f,  0.05646884f,  0.05755599f,  0.05865375f,  0.05976343f,
     0.06088505f,  0.06201727f,  0.06316010f,  0.06431353f,  0.06547757f,  0.06665222f,  0.06783747f,  0.06903333f,
     0.07023979f,  0.07145686f,  0.07268454f,  0.07392282f,  0.07517302f,  0.07643781f,  0.07771982f,  0.07902171f,
     0.08034613f,  0.08169309f,  0.08306126f,  0.08445065f,  0.08586124f,  0.08729305f,  0.08874606f,  0.09022028f,
     0.09171572f,  0.09323631f,  0.09479002f,  0.09638085f,  0.09801143f,  0.09968573f,  0.10140511f,  0.10317087f,
     0.10498565f,  0.10685606f,  0.10879138f,  0.11079693f,  0.11288061f,  0.11505435f,  0.11732612f,  0.11970645f,
     0.12221385f,  0.12486557f,  0.12767744f,  0.13067319f,  0.13388062f,  0.13732483f,  0.14103348f,  0.14504341f,
     0.14939809f,  0.15414486f,  0.15933627f,  0.16503254f,  0.17130551f,  0.17824113f,  0.18594321f,  0.19454679f,
     0.20423612f,  0.21526312f,  0.22799792f,  0.24303076f,  0.26139474f,  0.28521142f,  0.32037177f,
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, binary search on midpoints */
    if (val < MIDPOINTS_4BIT[7]) {
        if (val < MIDPOINTS_4BIT[3]) {
            if (val < MIDPOINTS_4BIT[1]) return val < MIDPOINTS_4BIT[0] ? 0 : 1;
            else                         return val < MIDPOINTS_4BIT[2] ? 2 : 3;
        } else {
            if (val < MIDPOINTS_4BIT[5]) return val < MIDPOINTS_4BIT[4] ? 4 : 5;
            else                         return val < MIDPOINTS_4BIT[6] ? 6 : 7;
        }
    } else {
        if (val < MIDPOINTS_4BIT[11]) {
            if (val < MIDPOINTS_4BIT[9])  return val < MIDPOINTS_4BIT[8] ? 8 : 9;
            else                          return val < MIDPOINTS_4BIT[10] ? 10 : 11;
        } else {
            if (val < MIDPOINTS_4BIT[13]) return val < MIDPOINTS_4BIT[12] ? 12 : 13;
            else                          return val < MIDPOINTS_4BIT[14] ? 14 : 15;
        }
    }
}

static int nearest_centroid_8bit(float val) {
    /* 256 monotonic centroids, binary search on 255 midpoints (lower_bound) */
    int lo = 0, hi = 255;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (val < MIDPOINTS_8BIT[mid]) hi = mid; else lo = mid + 1;
    }
    return lo;
}

/* ---------- TURBO2_0: 2-bit PolarQuant, no QJL ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO2; j++) norm += x[i*QK_TURBO2 + j] * x[i*QK_TURBO2 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO2 / 4);
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_0: 2-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles quantize on GPU. CPU path is simplified.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) norm += x[i*QK_TURBO3 + j] * x[i*QK_TURBO3 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO3 / 4);
        memset(y[i].signs, 0, QK_TURBO3 / 8);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_TCQ: Trellis-Coded Quantization ---------- */

void quantize_row_turbo3_tcq_ref(const float * GGML_RESTRICT x, block_turbo3_tcq * GGML_RESTRICT y, int64_t k) {
    // Stub — CUDA kernel handles TCQ quantize (Viterbi). CPU path zeros out.
    assert(k % QK_TURBO3_TCQ == 0);
    const int nb = k / QK_TURBO3_TCQ;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3_TCQ; j++) norm += x[i*QK_TURBO3_TCQ + j] * x[i*QK_TURBO3_TCQ + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, 49);
    }
}

void dequantize_row_turbo3_tcq(const block_turbo3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_UNUSED(x);
    assert(k % QK_TURBO3_TCQ == 0);
    const int nb = k / QK_TURBO3_TCQ;
    for (int block = 0; block < nb; block++) {
        for (int j = 0; j < QK_TURBO3_TCQ; j++) {
            y[block * QK_TURBO3_TCQ + j] = 0.0f;
        }
    }
}

size_t quantize_turbo3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3_TCQ == 0);

    size_t row_size = (n_per_row / QK_TURBO3_TCQ) * sizeof(block_turbo3_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_tcq_ref(
            src + row * n_per_row,
            (block_turbo3_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_TCQ: 2-bit Trellis-Coded Quantization ---------- */

void quantize_row_turbo2_tcq_ref(const float * GGML_RESTRICT x, block_turbo2_tcq * GGML_RESTRICT y, int64_t k) {
	// Stub — CUDA kernel handles TCQ quantize (Viterbi). CPU path zeros out.
	assert(k % QK_TURBO2_TCQ == 0);
	const int nb = k / QK_TURBO2_TCQ;
	for (int i = 0; i < nb; i++) {
		float norm = 0.0f;
		for (int j = 0; j < QK_TURBO2_TCQ; j++) norm += x[i*QK_TURBO2_TCQ + j] * x[i*QK_TURBO2_TCQ + j];
		y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
		memset(y[i].qs, 0, 33);
	}
}

void dequantize_row_turbo2_tcq(const block_turbo2_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
	GGML_UNUSED(x);
	assert(k % QK_TURBO2_TCQ == 0);
	const int nb = k / QK_TURBO2_TCQ;
	for (int block = 0; block < nb; block++) {
		for (int j = 0; j < QK_TURBO2_TCQ; j++) {
			y[block * QK_TURBO2_TCQ + j] = 0.0f;
		}
	}
}

size_t quantize_turbo2_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
	GGML_UNUSED(imatrix);
	assert(n_per_row % QK_TURBO2_TCQ == 0);

	size_t row_size = (n_per_row / QK_TURBO2_TCQ) * sizeof(block_turbo2_tcq);
	for (int64_t row = 0; row < nrows; row++) {
		quantize_row_turbo2_tcq_ref(
			src + row * n_per_row,
			(block_turbo2_tcq *)((char *)dst + row * row_size),
			n_per_row
		);
	}
	return nrows * row_size;
}

/* ---------- TURBO4_0: 4-bit PolarQuant (16 centroids, no QJL) ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Rotate */
        float rotated[TURBO_D];
        matvec(turbo_rotation, normalized, rotated, d);

        /* Step 3: 4-bit quantization — find nearest of 16 centroids */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Step 4: Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float r = CENTROIDS_4BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        /* Pack 4-bit indices: 2 per byte, low nibble first */
        for (int i = 0; i < d; i += 2) {
            y[block].qs[i / 2] = (uint8_t)((indices[i + 1] << 4) | (indices[i] & 0xF));
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 4-bit indices and reconstruct in rotated space */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (i & 1) ? (x[block].qs[i / 2] >> 4) : (x[block].qs[i / 2] & 0xF);
            rotated_recon[i] = CENTROIDS_4BIT[idx];
        }

        /* Inverse rotate */
        float * dst = y + block * d;
        matvec(turbo_rotation_t, rotated_recon, dst, d);

        /* Scale by norm */
        for (int i = 0; i < d; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO8_0: 8-bit Lloyd-Max (256 centroids, no QJL) ---------- */

void quantize_row_turbo8_0_ref(const float * GGML_RESTRICT x, block_turbo8_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO8 == 0);
    const int nb = k / QK_TURBO8;
    const int d  = QK_TURBO8;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Rotate */
        float rotated[TURBO_D];
        matvec(turbo_rotation, normalized, rotated, d);

        /* Step 3: 8-bit quantization — find nearest of 256 centroids */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_8bit(rotated[i]);
        }

        /* Step 4: Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float r = CENTROIDS_8BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        for (int i = 0; i < d; i++) y[block].qs[i] = indices[i];
    }
}

void dequantize_row_turbo8_0(const block_turbo8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO8 == 0);
    const int nb = k / QK_TURBO8;
    const int d  = QK_TURBO8;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Reconstruct in rotated space */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_8BIT[x[block].qs[i]];
        }

        /* Inverse rotate */
        float * dst = y + block * d;
        matvec(turbo_rotation_t, rotated_recon, dst, d);

        /* Scale by norm */
        for (int i = 0; i < d; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_turbo8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO8 == 0);

    size_t row_size = (n_per_row / QK_TURBO8) * sizeof(block_turbo8_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo8_0_ref(
            src + row * n_per_row,
            (block_turbo8_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

// turbo1_tcq CPU ref: STUB (the 256-state trellis codebook + Viterbi are CUDA-only; CPU path unused for KV decode).
void quantize_row_turbo1_tcq_ref(const float * GGML_RESTRICT x, block_turbo1_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO1_TCQ == 0);
    const int nb = k / QK_TURBO1_TCQ;
    for (int b = 0; b < nb; b++) {
        const float * src = x + b * QK_TURBO1_TCQ;
        float ns = 0.0f; for (int i = 0; i < QK_TURBO1_TCQ; i++) ns += src[i]*src[i];
        y[b].norm = GGML_FP32_TO_FP16(sqrtf(ns) * 0.08838834764831845f);
        for (int i = 0; i < 17; i++) y[b].qs[i] = 0;
    }
}
void dequantize_row_turbo1_tcq(const block_turbo1_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO1_TCQ == 0);
    memset(y, 0, (size_t)k * sizeof(float)); // stub; real decode is the CUDA trellis lookup + inverse FWHT
    (void)x;
}
size_t quantize_turbo1_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO1_TCQ == 0);
    size_t row_size = (n_per_row / QK_TURBO1_TCQ) * sizeof(block_turbo1_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo1_tcq_ref(src + row * n_per_row,
            (block_turbo1_tcq *)((char *)dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}
