#include "../fattn-vec.cuh"

// turbo K rows are 128 wide, so there is no 64 instance
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
