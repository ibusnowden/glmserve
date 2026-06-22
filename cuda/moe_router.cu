// glmserve — MoE router: sigmoid scoring, aux-loss-free top-k selection,
// optional normalization, routed_scaling_factor. One block per token.
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

// Single-thread-per-token selection (E up to 256, topk up to 8 — tiny work).
__global__ void moe_router_kernel(const float* __restrict__ logits, const float* __restrict__ e_bias,
                                  int E, int topk, bool norm_topk, float routed_scale,
                                  int* __restrict__ topk_ids, float* __restrict__ topk_w) {
    int t = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float* lt = logits + (size_t)t * E;

    // sigmoid scores + selection score (score + bias)
    // iterative top-k (E and topk are small)
    int* ids = topk_ids + (size_t)t * topk;
    float* w  = topk_w + (size_t)t * topk;
    bool used[256];
    for (int e = 0; e < E; ++e) used[e] = false;

    float wsum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int best = -1;
        float best_choose = -1e30f;
        for (int e = 0; e < E; ++e) {
            if (used[e]) continue;
            float score = dsigmoid(lt[e]);
            float choose = score + (e_bias ? e_bias[e] : 0.0f);
            if (choose > best_choose) { best_choose = choose; best = e; }
        }
        used[best] = true;
        float score = dsigmoid(lt[best]);
        ids[k] = best;
        w[k] = score;       // gate uses original score, not bias-corrected
        wsum += score;
    }
    for (int k = 0; k < topk; ++k) {
        float v = w[k];
        if (norm_topk && wsum > 0.0f) v /= wsum;
        w[k] = v * routed_scale;
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
