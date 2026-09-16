// TurboQuant shared constants: Lloyd-Max codebooks and the FWHT rotation signs.
// Values mirror ggml-cuda/turbo-quant-cuda.cuh so the Vulkan encode/decode match
// the CUDA implementation bit for bit.
#ifndef TURBO_GLSL
#define TURBO_GLSL

const float k_turbo_centroids_2bit[4] = float[4](
    -0.133462, -0.039994, 0.039994, 0.133462
);

const float k_turbo_centroids_3bit[8] = float[8](
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685
);

const float k_turbo_centroids_4bit[16] = float[16](
    -0.241556, -0.182907, -0.143047, -0.111065,
    -0.083317, -0.058069, -0.034311, -0.011353,
     0.011353,  0.034311,  0.058069,  0.083317,
     0.111065,  0.143047,  0.182907,  0.241556
);

const float k_turbo_wht_signs1[128] = float[128](
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0
);

const float k_turbo_wht_signs2[128] = float[128](
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0
);

#define TURBO_FWHT_INV_SQRT_128 0.08838834764831845

void turbo_fwht_128(inout float x[128]) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; i++) {
        x[i] *= TURBO_FWHT_INV_SQRT_128;
    }
}

void turbo_rotate_forward(inout float x[128]) {
    for (int i = 0; i < 128; i++) x[i] *= k_turbo_wht_signs1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= k_turbo_wht_signs2[i];
}

// The normalized FWHT is an involution, so the inverse is the forward rotation
// with the two sign arrays swapped.
void turbo_rotate_inverse(inout float x[128]) {
    for (int i = 0; i < 128; i++) x[i] *= k_turbo_wht_signs2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= k_turbo_wht_signs1[i];
}

uint turbo_find_nearest_4bit(float val) {
    // CUDA picks the upper centroid on ties (val < mid), mirror that with strict <.
    for (uint i = 0u; i < 15u; i++) {
        if (val < 0.5 * (k_turbo_centroids_4bit[i] + k_turbo_centroids_4bit[i + 1u])) return i;
    }
    return 15u;
}

uint turbo_find_nearest_3bit(float val) {
    for (uint i = 0u; i < 7u; i++) {
        if (val < 0.5 * (k_turbo_centroids_3bit[i] + k_turbo_centroids_3bit[i + 1u])) return i;
    }
    return 7u;
}

uint turbo_find_nearest_2bit(float val) {
    for (uint i = 0u; i < 3u; i++) {
        if (val < 0.5 * (k_turbo_centroids_2bit[i] + k_turbo_centroids_2bit[i + 1u])) return i;
    }
    return 3u;
}

#endif // TURBO_GLSL
