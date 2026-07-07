// glmserve — MoE router: sigmoid scoring, aux-loss-free top-k selection,
// optional normalization, routed_scaling_factor. One warp per token.
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

// Warp-parallel top-k (E up to 256, topk up to 8). Lane l owns experts
// l, l+32, l+64, ... and keeps their scores in registers; each of the topk
// rounds does a lane-local scan plus a warp shuffle arg-max. Semantics are
// bit-identical to the serial reference: strict > comparison with
// lower-expert-index tie-break, gate weight = original sigmoid score (not
// bias-corrected), wsum accumulated in selection order. The serial
// one-thread-per-token predecessor cost ~0.15 ms per decode layer (a single
// thread walking 8x160 scores in global memory).
__global__ void moe_router_kernel(const float* __restrict__ logits, const float* __restrict__ e_bias,
                                  int E, int topk, bool norm_topk, float routed_scale,
                                  int* __restrict__ topk_ids, float* __restrict__ topk_w) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const float* lt = logits + (size_t)t * E;

    // E <= 256 => at most 8 lane-owned experts.
    float score[8];
    float choose[8];   // score + bias; set to -inf once selected
    int cnt = 0;
    for (int e = lane; e < E; e += 32) {
        const float sc = dsigmoid(lt[e]);
        score[cnt] = sc;
        choose[cnt] = sc + (e_bias ? e_bias[e] : 0.0f);
        ++cnt;
    }

    int* ids = topk_ids + (size_t)t * topk;
    float* w = topk_w + (size_t)t * topk;
    float wsum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        // Lane-local best among not-yet-selected (keeps lowest index on ties:
        // strict > over ascending j == ascending expert index within the lane).
        float bc = -1e30f;
        int bj = -1;
        for (int j = 0; j < cnt; ++j)
            if (choose[j] > bc) { bc = choose[j]; bj = j; }
        int be = bj >= 0 ? lane + (bj << 5) : E;
        float bs = bj >= 0 ? score[bj] : 0.0f;
        // Warp arg-max: higher choose wins, equal choose -> lower expert index
        // (matches the serial scan order, which visits experts ascending).
        for (int off = 16; off > 0; off >>= 1) {
            const float oc = __shfl_down_sync(0xffffffffu, bc, off);
            const int   oe = __shfl_down_sync(0xffffffffu, be, off);
            const float os = __shfl_down_sync(0xffffffffu, bs, off);
            if (oc > bc || (oc == bc && oe < be)) { bc = oc; be = oe; bs = os; }
        }
        be = __shfl_sync(0xffffffffu, be, 0);
        bs = __shfl_sync(0xffffffffu, bs, 0);
        // Owning lane retires the winner.
        if ((be & 31) == lane) choose[be >> 5] = -1e30f;
        if (lane == 0) {
            ids[k] = be;
            w[k] = bs;
            wsum += bs;
        }
    }
    if (lane == 0) {
        for (int k = 0; k < topk; ++k) {
            float v = w[k];
            if (norm_topk && wsum > 0.0f) v /= wsum;
            w[k] = v * routed_scale;
        }
    }
}

void moe_router(const float* logits, const float* e_bias, int n, int E, int topk,
                bool norm_topk, float routed_scale, int* topk_ids, float* topk_weights,
                cudaStream_t s) {
    moe_router_kernel<<<(unsigned)n, 32, 0, s>>>(logits, e_bias, E, topk, norm_topk,
                                                 routed_scale, topk_ids, topk_weights);
}

}  // namespace cuda
}  // namespace glmserve
