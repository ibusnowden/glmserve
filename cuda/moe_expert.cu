// glmserve — MoE expert FFN: for each selected (token, expert) pair compute
// down(silu(gate(x)) * up(x)) and accumulate weight * result into the token's
// output. One block per (token, slot); threads split the moe_inter dim.
//
// Weight layout (contiguous per expert):
//   gate_w[E, moe_inter, hidden], up_w[E, moe_inter, hidden], down_w[E, hidden, moe_inter]
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

__global__ void moe_expert_kernel(const float* __restrict__ x, const int* __restrict__ topk_ids,
                                  const float* __restrict__ topk_w, const float* __restrict__ gate_w,
                                  const float* __restrict__ up_w, const float* __restrict__ down_w,
                                  int topk, int hidden, int moe_inter, float* __restrict__ out) {
    int t = blockIdx.x;
    int slot = blockIdx.y;
    int e = topk_ids[t * topk + slot];
    float weight = topk_w[t * topk + slot];

    const float* xt = x + (size_t)t * hidden;
    const float* gw = gate_w + (size_t)e * moe_inter * hidden;
    const float* uw = up_w + (size_t)e * moe_inter * hidden;
    const float* dw = down_w + (size_t)e * hidden * moe_inter;

    extern __shared__ float h_act[];   // moe_inter activations (silu(gate)*up)

    // compute h_act[f] for f assigned to this thread
    for (int f = threadIdx.x; f < moe_inter; f += blockDim.x) {
        float g = 0.0f, u = 0.0f;
        const float* gwf = gw + (size_t)f * hidden;
        const float* uwf = uw + (size_t)f * hidden;
        for (int i = 0; i < hidden; ++i) { g += gwf[i] * xt[i]; u += uwf[i] * xt[i]; }
        h_act[f] = silu(g) * u;
    }
    __syncthreads();

    // down projection: out[o] += weight * sum_f down[o,f] * h_act[f]
    for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        const float* dwo = dw + (size_t)o * moe_inter;
        float acc = 0.0f;
        for (int f = 0; f < moe_inter; ++f) acc += dwo[f] * h_act[f];
        atomicAdd(&out[(size_t)t * hidden + o], weight * acc);
    }
}

void moe_expert_ffn(const float* x, const int* topk_ids, const float* topk_weights,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int n, int topk, int hidden, int moe_inter, int E, float* out,
                    cudaStream_t s) {
    (void)E;
    dim3 grid((unsigned)n, (unsigned)topk);
    int threads = 128;
    size_t shmem = (size_t)moe_inter * sizeof(float);
    moe_expert_kernel<<<grid, threads, shmem, s>>>(x, topk_ids, topk_weights, gate_w,
                                                   up_w, down_w, topk, hidden, moe_inter, out);
}

}  // namespace cuda
}  // namespace glmserve
