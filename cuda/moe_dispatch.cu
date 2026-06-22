// glmserve — MoE dispatch: group the (token, slot) pairs by expert so the
// expert FFN can run as grouped GEMMs instead of scattered per-token matvecs.
//
// Given topk_ids[n*topk], produce:
//   expert_offsets[E+1]      prefix sum of per-expert row counts
//   sorted_token_ids[n*topk] token index for each grouped row
//   sorted_row[n*topk]       which of the token's topk slots the row is
// This is a counting sort over E buckets (E<=256), done single-threaded for the
// small decode batches V0 targets; a parallel histogram replaces it at scale.
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

__global__ void moe_dispatch_kernel(const int* __restrict__ topk_ids, int n, int topk, int E,
                                    int* __restrict__ expert_offsets,
                                    int* __restrict__ sorted_token_ids,
                                    int* __restrict__ sorted_row) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int total = n * topk;

    // histogram
    for (int e = 0; e <= E; ++e) expert_offsets[e] = 0;
    for (int i = 0; i < total; ++i) expert_offsets[topk_ids[i] + 1]++;
    for (int e = 0; e < E; ++e) expert_offsets[e + 1] += expert_offsets[e];

    // scatter (use a moving cursor seeded from offsets)
    extern __shared__ int cursor[];   // size E
    for (int e = 0; e < E; ++e) cursor[e] = expert_offsets[e];
    for (int t = 0; t < n; ++t) {
        for (int k = 0; k < topk; ++k) {
            int e = topk_ids[t * topk + k];
            int dst = cursor[e]++;
            sorted_token_ids[dst] = t;
            sorted_row[dst] = k;
        }
    }
}

void moe_dispatch(const int* topk_ids, int n, int topk, int E, int* expert_offsets,
                  int* sorted_token_ids, int* sorted_row, cudaStream_t s) {
    moe_dispatch_kernel<<<1, 32, (size_t)E * sizeof(int), s>>>(
        topk_ids, n, topk, E, expert_offsets, sorted_token_ids, sorted_row);
}

}  // namespace cuda
}  // namespace glmserve
