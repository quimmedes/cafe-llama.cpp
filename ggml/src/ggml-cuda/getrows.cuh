#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream);

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define GGML_CUDA_GET_ROWS_MULTI_MAX 64

// true if dst is a GET_ROWS of a few F32 rows that ggml_cuda_op_get_rows_multi can do
bool ggml_cuda_get_rows_multi_ok(const ggml_tensor * dst);

// independent GET_ROWS (see ggml_cuda_get_rows_multi_ok) as one launch
void ggml_cuda_op_get_rows_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n);

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
