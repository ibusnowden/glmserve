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

// n-invariant fp32 gemv: one warp per output row, lane-strided K with a fixed
// warp-shuffle reduce, the token loop just repeating the identical per-token
// order. cuBLAS picks different kernels (and thus summation orders) for
// different n, so a verify chunk's rows would drift (last-bit) from the n=1
// decode steps they re-check — the F32 MoE router gemv was the residual
// ulp-level source of spec-vs-greedy divergence after the all-reduce fix.
// Decode and verify-sized calls (n <= kMlaParityMaxQ) take this kernel;
// prefill stays on cuBLAS.
__global__ void gemv_fp32_kernel(const float* __restrict__ x, const float* __restrict__ W,
                                 const float* __restrict__ bias, float* __restrict__ y,
                                 int64_t n, int64_t in, int64_t out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t o = (int64_t)blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= out) return;
    const float* wrow = W + o * in;
    for (int64_t t = 0; t < n; ++t) {
        const float* xt = x + t * in;
        float acc = 0.0f;
        for (int64_t i = lane; i < in; i += 32) acc += wrow[i] * xt[i];
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) y[t * out + o] = bias ? acc + bias[o] : acc;
    }
}

void gemm_fp32(const float* x, const float* W, const float* bias, float* y,
               int64_t n, int64_t in, int64_t out, cudaStream_t s) {
    if (n <= kMlaParityMaxQ) {
        const int warps = 8;
        const unsigned blocks = (unsigned)((out + warps - 1) / warps);
        gemv_fp32_kernel<<<blocks, warps * 32, 0, s>>>(x, W, bias, y, n, in, out);
        return;
    }
    cublasSetStream(handle(), s);
    const float alpha = 1.0f, beta = 0.0f;
    // C(out x n) = W^T(out x in) * X(in x n)
    cublasSgemm(handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out, (int)n, (int)in,
                &alpha, W, (int)in, x, (int)in, &beta, y, (int)out);
    maybe_bias(y, bias, n, out, s);
}

// Pedantic fp32 GEMM: identical to gemm_fp32 but forces CUBLAS_PEDANTIC_MATH
// on the cuBLAS path, so cublasSgemm uses true fp32 (no TF32 tensor cores).
// On Ada/Ampere the default math mode rounds each fp32 product to TF32's
// 10-bit mantissa before accumulating — a systematic error large enough to
// flip DSA top-k selections (a 128-dim dot accumulates ~1e-3 of relative
// TF32 error vs ~1e-6 for fp32 summation-order noise). The DSA selector must
// match the scalar kernel's fp32 domain so the greedy stream agrees with the
// scalar path; only the (ULP-level) summation order then differs. The n <=
// kMlaParityMaxQ gemv branch is already true fp32 (no cuBLAS), so it is
// unaffected and just delegates.
void gemm_fp32_pedantic(const float* x, const float* W, const float* bias, float* y,
                        int64_t n, int64_t in, int64_t out, cudaStream_t s) {
    if (n <= kMlaParityMaxQ) {
        gemm_fp32(x, W, bias, y, n, in, out, s);
        return;
    }
    cublasHandle_t h = handle();
    cublasMath_t saved;
    cublasGetMathMode(h, &saved);
    cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
    cublasSetStream(h, s);
    const float alpha = 1.0f, beta = 0.0f;
    // C(out x n) = W^T(out x in) * X(in x n)
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out, (int)n, (int)in,
                &alpha, W, (int)in, x, (int)in, &beta, y, (int)out);
    cublasSetMathMode(h, saved);
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
