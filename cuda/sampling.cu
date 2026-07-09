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

// Per-row argmax with EXACT first-max (lowest index) tie semantics, matching a
// serial `if (x[i] > best)` host scan. Emits float2{val, (float)idx} per row so
// TP ranks can combine shard winners with one tiny all-reduce (idx < 2^24, so
// the float round-trip is exact). One block per row.
__global__ void argmax_rows_kernel(const float* __restrict__ y, int ncols, int64_t ld,
                                   float2* __restrict__ out) {
    extern __shared__ float smem[];
    int* sidx = (int*)(smem + blockDim.x);
    const float* row = y + (int64_t)blockIdx.x * ld;
    float best = -1e30f;
    int best_i = 0;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        const float v = row[i];
        if (v > best) { best = v; best_i = i; }
    }
    smem[threadIdx.x] = best;
    sidx[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float vo = smem[threadIdx.x + stride];
            const int io = sidx[threadIdx.x + stride];
            if (vo > smem[threadIdx.x] ||
                (vo == smem[threadIdx.x] && io < sidx[threadIdx.x])) {
                smem[threadIdx.x] = vo;
                sidx[threadIdx.x] = io;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = make_float2(smem[0], (float)sidx[0]);
}

void argmax_rows(const float* y, int nrows, int ncols, int64_t ld, float2* out_pairs,
                 cudaStream_t s) {
    const int threads = 256;
    const size_t shmem = threads * (sizeof(float) + sizeof(int));
    argmax_rows_kernel<<<nrows, threads, shmem, s>>>(y, ncols, ld, out_pairs);
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
