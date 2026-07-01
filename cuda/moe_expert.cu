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

__device__ inline float q4_weight(const uint8_t* qbase, const float* sbase,
                                  int row, int col, int in, int group_size) {
    int packed_in = (in + 1) / 2;
    int groups = (in + group_size - 1) / group_size;
    const uint8_t* qrow = qbase + (size_t)row * packed_in;
    uint8_t byte = qrow[col >> 1];
    int q = (col & 1) ? (byte >> 4) : (byte & 0x0F);
    return (float)(q - 8) * sbase[(size_t)row * groups + col / group_size];
}

__global__ void moe_expert_w4_kernel(const float* __restrict__ x,
                                     const int* __restrict__ topk_ids,
                                     const float* __restrict__ topk_w,
                                     const uint8_t* __restrict__ gate_q,
                                     const float* __restrict__ gate_sc,
                                     const uint8_t* __restrict__ up_q,
                                     const float* __restrict__ up_sc,
                                     const uint8_t* __restrict__ down_q,
                                     const float* __restrict__ down_sc,
                                     int topk, int hidden, int moe_inter,
                                     int group_size, float* __restrict__ out) {
    int t = blockIdx.x;
    int slot = blockIdx.y;
    int e = topk_ids[t * topk + slot];
    float weight = topk_w[t * topk + slot];

    const float* xt = x + (size_t)t * hidden;
    const int gate_packed = (hidden + 1) / 2;
    const int gate_groups = (hidden + group_size - 1) / group_size;
    const int down_packed = (moe_inter + 1) / 2;
    const int down_groups = (moe_inter + group_size - 1) / group_size;
    const uint8_t* gq = gate_q + (size_t)e * moe_inter * gate_packed;
    const uint8_t* uq = up_q + (size_t)e * moe_inter * gate_packed;
    const uint8_t* dq = down_q + (size_t)e * hidden * down_packed;
    const float* gs = gate_sc + (size_t)e * moe_inter * gate_groups;
    const float* us = up_sc + (size_t)e * moe_inter * gate_groups;
    const float* ds = down_sc + (size_t)e * hidden * down_groups;

    extern __shared__ float h_act[];

    for (int f = threadIdx.x; f < moe_inter; f += blockDim.x) {
        float g = 0.0f, u = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            g += q4_weight(gq, gs, f, i, hidden, group_size) * xt[i];
            u += q4_weight(uq, us, f, i, hidden, group_size) * xt[i];
        }
        h_act[f] = silu(g) * u;
    }
    __syncthreads();

    for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        for (int f = 0; f < moe_inter; ++f)
            acc += q4_weight(dq, ds, o, f, moe_inter, group_size) * h_act[f];
        atomicAdd(&out[(size_t)t * hidden + o], weight * acc);
    }
}

void moe_expert_ffn_w4a16(const float* x, const int* topk_ids, const float* topk_weights,
                          const uint8_t* gate_q, const float* gate_sc,
                          const uint8_t* up_q, const float* up_sc,
                          const uint8_t* down_q, const float* down_sc,
                          int n, int topk, int hidden, int moe_inter, int E,
                          int group_size, float* out, cudaStream_t s) {
    (void)E;
    dim3 grid((unsigned)n, (unsigned)topk);
    int threads = 128;
    size_t shmem = (size_t)moe_inter * sizeof(float);
    moe_expert_w4_kernel<<<grid, threads, shmem, s>>>(x, topk_ids, topk_weights,
                                                      gate_q, gate_sc, up_q, up_sc,
                                                      down_q, down_sc, topk, hidden,
                                                      moe_inter, group_size, out);
}

}  // namespace cuda
}  // namespace glmserve
