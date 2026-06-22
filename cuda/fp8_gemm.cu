// glmserve — FP8 (e4m3) weight GEMM: dequantize weights on the fly x fp32 acts.
//
// GLM-5.2 ships an FP8 checkpoint (spec §4); on 96GB GPUs this is the native
// path. Each block computes one output feature for a small token batch,
// decoding e4m3 bytes to float inline. (For large prefills a CUTLASS/cuBLASLt
// mixed-input GEMM would replace this; this kernel is the correctness baseline.)
#include "common.cuh"
#include "kernels.cuh"
#include <cuda_fp8.h>

namespace glmserve {
namespace cuda {

__global__ void fp8_gemm_kernel(const float* __restrict__ x, const uint8_t* __restrict__ fp8W,
                                const float* __restrict__ w_scale, const float* __restrict__ bias,
                                float* __restrict__ y, int n, int in, int out) {
    extern __shared__ float smem[];
    int o = blockIdx.x;
    const uint8_t* wrow = fp8W + (size_t)o * in;
    float sc = w_scale ? w_scale[o] : 1.0f;   // per-output-row scale (or 1.0)

    for (int t = 0; t < n; ++t) {
        const float* xt = x + (size_t)t * in;
        float local = 0.0f;
        for (int i = threadIdx.x; i < in; i += blockDim.x) {
            __nv_fp8_e4m3 e;
            e.__x = wrow[i];
            float w = (float)e * sc;
            local += w * xt[i];
        }
        float acc = block_reduce_sum(local, smem);
        if (threadIdx.x == 0) y[(size_t)t * out + o] = bias ? acc + bias[o] : acc;
        __syncthreads();
    }
}

void gemm_fp8(const float* x, const uint8_t* fp8W, const float* w_scale, const float* bias,
              float* y, int64_t n, int64_t in, int64_t out, cudaStream_t s) {
    int threads = 256;
    size_t shmem = 32 * sizeof(float);
    fp8_gemm_kernel<<<(unsigned)out, threads, shmem, s>>>(x, fp8W, w_scale, bias, y,
                                                          (int)n, (int)in, (int)out);
}

}  // namespace cuda
}  // namespace glmserve
