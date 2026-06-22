// glmserve — RMSNorm kernels (full-row and per-head qk-norm).
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

// One block per row; threads stride over the feature dim.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ out, int64_t d, float eps) {
    extern __shared__ float smem[];
    const float* xr = x + blockIdx.x * d;
    float* outr = out + blockIdx.x * d;

    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        float v = xr[i];
        local += v * v;
    }
    float ss = block_reduce_sum(local, smem);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / d + eps);
    __syncthreads();

    for (int64_t i = threadIdx.x; i < d; i += blockDim.x)
        outr[i] = xr[i] * inv * w[i];
}

void rmsnorm(const float* x, const float* w, float* out, int64_t n, int64_t d,
             float eps, cudaStream_t s) {
    int threads = 256;
    size_t shmem = 32 * sizeof(float);
    rmsnorm_kernel<<<(unsigned)n, threads, shmem, s>>>(x, w, out, d, eps);
}

// One block per (row, head); normalize each head's head_dim vector in place.
__global__ void per_head_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                        int64_t n_heads, int64_t head_dim, float eps) {
    extern __shared__ float smem[];
    int64_t row = blockIdx.x / n_heads;
    int64_t head = blockIdx.x % n_heads;
    float* v = x + (row * n_heads + head) * head_dim;

    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < head_dim; i += blockDim.x) local += v[i] * v[i];
    float ss = block_reduce_sum(local, smem);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / head_dim + eps);
    __syncthreads();
    for (int64_t i = threadIdx.x; i < head_dim; i += blockDim.x) v[i] = v[i] * inv * w[i];
}

void per_head_rmsnorm(float* x, const float* w, int64_t n, int64_t n_heads,
                      int64_t head_dim, float eps, cudaStream_t s) {
    int threads = 64;
    size_t shmem = 32 * sizeof(float);
    per_head_rmsnorm_kernel<<<(unsigned)(n * n_heads), threads, shmem, s>>>(
        x, w, n_heads, head_dim, eps);
}

}  // namespace cuda
}  // namespace glmserve
