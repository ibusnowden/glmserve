// glmserve — shared CUDA utilities (error checking, warp reductions, dtype).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace glmserve {
namespace cuda {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[glmserve][cuda] %s:%d %s: %s\n", __FILE__,   \
                         __LINE__, #call, cudaGetErrorString(_e));              \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarp = 32;

// Sum-reduce a value across a warp.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Max-reduce a value across a warp.
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Block-wide sum using shared memory across warps. `smem` must hold >= 32 floats.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int lane = threadIdx.x % kWarp;
    int wid  = threadIdx.x / kWarp;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + kWarp - 1) / kWarp;
    v = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

}  // namespace cuda
}  // namespace glmserve
