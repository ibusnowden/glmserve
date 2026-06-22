// glmserve — sampling kernels: argmax (greedy) and in-place softmax/temperature.
// Top-k / top-p narrowing for the GPU path is done host-side on the small
// candidate set after these reductions (see src/sampler.cpp for the policy).
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

__global__ void argmax_kernel(const float* __restrict__ logits, int vocab, int* __restrict__ out) {
    extern __shared__ float smem[];
    int* sidx = (int*)(smem + blockDim.x);
    float best = -1e30f; int best_i = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (logits[i] > best) { best = logits[i]; best_i = i; }
    }
    smem[threadIdx.x] = best;
    sidx[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride] > smem[threadIdx.x]) {
                smem[threadIdx.x] = smem[threadIdx.x + stride];
                sidx[threadIdx.x] = sidx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = sidx[0];
}

void argmax(const float* logits, int vocab, int* out_id, cudaStream_t s) {
    int threads = 256;
    size_t shmem = threads * (sizeof(float) + sizeof(int));
    argmax_kernel<<<1, threads, shmem, s>>>(logits, vocab, out_id);
}

__global__ void softmax_kernel(float* __restrict__ logits, int vocab, float inv_t) {
    extern __shared__ float smem[];
    float m = -1e30f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) m = fmaxf(m, logits[i]);
    smem[threadIdx.x] = m; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    float gmax = smem[0]; __syncthreads();
    float local = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float e = __expf((logits[i] - gmax) * inv_t);
        logits[i] = e; local += e;
    }
    smem[threadIdx.x] = local; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float sum = smem[0]; __syncthreads();
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) logits[i] /= sum;
}

void softmax_inplace(float* logits, int vocab, float temperature, cudaStream_t s) {
    int threads = 256;
    size_t shmem = threads * sizeof(float);
    float inv_t = temperature > 0.0f ? 1.0f / temperature : 1.0f;
    softmax_kernel<<<1, threads, shmem, s>>>(logits, vocab, inv_t);
}

}  // namespace cuda
}  // namespace glmserve
