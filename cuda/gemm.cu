// glmserve — dense GEMM via cuBLAS.
//
// Computes the HF linear y[n,out] = x[n,in] @ W[out,in]^T (+bias). All buffers
// are row-major; we map to cuBLAS (column-major) by treating the row-major
// [out,in] weight as a column-major [in,out] matrix and transposing it.
#include "common.cuh"
#include "kernels.cuh"

#include <cublas_v2.h>

namespace glmserve {
namespace cuda {

static cublasHandle_t g_handle = nullptr;
static cublasHandle_t handle() {
    if (!g_handle) {
        if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[glmserve][cuda] cublasCreate failed\n");
            std::abort();
        }
    }
    return g_handle;
}

__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ b,
                                int64_t n, int64_t out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * out) y[idx] += b[idx % out];
}

static void maybe_bias(float* y, const float* bias, int64_t n, int64_t out, cudaStream_t s) {
    if (!bias) return;
    int64_t total = n * out;
    add_bias_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(y, bias, n, out);
}

void gemm_fp32(const float* x, const float* W, const float* bias, float* y,
               int64_t n, int64_t in, int64_t out, cudaStream_t s) {
    cublasSetStream(handle(), s);
    const float alpha = 1.0f, beta = 0.0f;
    // C(out x n) = W^T(out x in) * X(in x n)
    cublasSgemm(handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out, (int)n, (int)in,
                &alpha, W, (int)in, x, (int)in, &beta, y, (int)out);
    maybe_bias(y, bias, n, out, s);
}

void gemm_fp16(const half* x, const half* W, const float* bias, half* y,
               int64_t n, int64_t in, int64_t out, cudaStream_t s) {
    cublasSetStream(handle(), s);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                 (int)out, (int)n, (int)in,
                 &alpha, W, CUDA_R_16F, (int)in, x, CUDA_R_16F, (int)in,
                 &beta, y, CUDA_R_16F, (int)out,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    // (fp16 bias add omitted; callers using fp16 add bias in their own epilogue)
    (void)bias;
}

}  // namespace cuda
}  // namespace glmserve
