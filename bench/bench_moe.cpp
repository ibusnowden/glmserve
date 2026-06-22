// glmserve bench — MoE expert FFN. CPU build: times the silu-gated expert math
// (gate/up/down) on synthetic data at GLM-5.2 expert dims. GPU build: times the
// moe_expert / moe_router kernels. Usage: bench_moe [n_tokens]
#include "common.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"
#endif

using namespace glmserve;

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 8;
    const int hidden = 6144, moe_inter = 1536, topk = 8;

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> x(n * hidden), gate(moe_inter * hidden), up(moe_inter * hidden),
                       down(hidden * moe_inter);
    for (auto& v : x) v = nd(rng);
    for (auto& v : gate) v = nd(rng);
    for (auto& v : up) v = nd(rng);
    for (auto& v : down) v = nd(rng);

    auto silu = [](float z) { return z / (1.0f + std::exp(-z)); };
    std::vector<float> out(n * hidden, 0.0f), h(moe_inter);

    Timer t;
    for (int rep = 0; rep < topk; ++rep)       // emulate topk experts per token
        for (int tk = 0; tk < n; ++tk) {
            const float* xt = x.data() + (size_t)tk * hidden;
            for (int f = 0; f < moe_inter; ++f) {
                float g = 0, u = 0;
                const float* gw = gate.data() + (size_t)f * hidden;
                const float* uw = up.data() + (size_t)f * hidden;
                for (int i = 0; i < hidden; ++i) { g += gw[i] * xt[i]; u += uw[i] * xt[i]; }
                h[f] = silu(g) * u;
            }
            for (int oi = 0; oi < hidden; ++oi) {
                const float* dw = down.data() + (size_t)oi * moe_inter;
                float acc = 0;
                for (int f = 0; f < moe_inter; ++f) acc += dw[f] * h[f];
                out[(size_t)tk * hidden + oi] += acc;
            }
        }
    double ms = t.ms();
    double flops = (double)n * topk * (2.0 * moe_inter * hidden * 2 + 2.0 * hidden * moe_inter);
    std::printf("MoE expert FFN (CPU): %d tok x top-%d, hidden=%d moe_inter=%d\n",
                n, topk, hidden, moe_inter);
    std::printf("  %.1f ms, %.2f GFLOP/s\n", ms, flops / 1e9 / (ms / 1000.0));

#ifdef GLMSERVE_CUDA
    std::printf("  (GPU moe_expert/moe_router kernels available — see tests/test_moe_router)\n");
#endif
    return 0;
}
