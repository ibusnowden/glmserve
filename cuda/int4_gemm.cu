// glmserve — W4A16 GEMM: packed-int4 weights (group-symmetric) x fp activations.
//
// This is the memory path that makes GLM-5.2 fit on 16x48GB (spec §15.1): the
// weight stays 4-bit in VRAM and is dequantized on the fly inside the kernel.
// Layout: qW packs two signed nibbles (zero-point 8) per byte for W[out,in];
// scales[out * num_groups], group_size columns share one scale. Each block
// computes one output column for a tile of tokens.
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

// grid.x = out (one output feature per block), threads cooperate over `in`.
__global__ void w4a16_kernel(const float* __restrict__ x, const uint8_t* __restrict__ qW,
                             const float* __restrict__ scales, const float* __restrict__ bias,
                             float* __restrict__ y, int n, int in, int out, int group_size) {
    extern __shared__ float smem[];
    int o = blockIdx.x;                       // output feature
    int num_groups = (in + group_size - 1) / group_size;
    const uint8_t* wrow = qW + (size_t)o * ((in + 1) / 2);  // packed row
    const float* srow = scales + (size_t)o * num_groups;

    for (int t = 0; t < n; ++t) {             // tokens (small batch in decode)
        const float* xt = x + (size_t)t * in;
        float local = 0.0f;
        for (int i = threadIdx.x; i < in; i += blockDim.x) {
            uint8_t byte = wrow[i >> 1];
            int q = (i & 1) ? (byte >> 4) : (byte & 0x0F);
            float w = (float)(q - 8) * srow[i / group_size];
            local += w * xt[i];
        }
        float acc = block_reduce_sum(local, smem);
        if (threadIdx.x == 0) {
            y[(size_t)t * out + o] = bias ? acc + bias[o] : acc;
        }
        __syncthreads();
    }
}

void gemm_w4a16(const float* x, const uint8_t* qW, const float* scales, const float* bias,
                float* y, int64_t n, int64_t in, int64_t out, int64_t group_size,
                cudaStream_t s) {
    int threads = 256;
    size_t shmem = 32 * sizeof(float);
    w4a16_kernel<<<(unsigned)out, threads, shmem, s>>>(x, qW, scales, bias, y, (int)n,
                                                       (int)in, (int)out, (int)group_size);
}

}  // namespace cuda
}  // namespace glmserve
