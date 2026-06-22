// glmserve test — MoE router CUDA kernel (sigmoid top-k + bias + norm + scale)
// vs CPU reference. Checks selected expert ids and gate weights.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "cuda_test_utils.hpp"
#endif

static float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

int main() {
    const int n = 7, E = 32, topk = 4;
    const float scale = 2.5f; const bool norm = true;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> logits(n * E), ebias(E);
    for (auto& v : logits) v = nd(rng);
    for (auto& v : ebias) v = 0.05f * nd(rng);

    std::vector<int> ref_ids(n * topk);
    std::vector<float> ref_w(n * topk);
    for (int t = 0; t < n; ++t) {
        std::vector<int> idx(E); std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> sc(E), ch(E);
        for (int e = 0; e < E; ++e) { sc[e] = sigm(logits[t * E + e]); ch[e] = sc[e] + ebias[e]; }
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int a, int b) { return ch[a] > ch[b]; });
        float s = 0; for (int k = 0; k < topk; ++k) s += sc[idx[k]];
        for (int k = 0; k < topk; ++k) {
            ref_ids[t * topk + k] = idx[k];
            float w = sc[idx[k]];
            if (norm && s > 0) w /= s;
            ref_w[t * topk + k] = w * scale;
        }
    }

#ifdef GLMSERVE_CUDA
    const char* test_name = "test_moe_router";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    using namespace glmserve::cuda;
    float *dl = nullptr, *db = nullptr, *dw = nullptr; int* di = nullptr;
    CUDA_TEST_CHECK(cudaMalloc(&dl, logits.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&db, E * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dw, n * topk * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&di, n * topk * sizeof(int)));
    CUDA_TEST_CHECK(cudaMemcpy(dl, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(db, ebias.data(), E * sizeof(float), cudaMemcpyHostToDevice));
    moe_router(dl, db, n, E, topk, norm, scale, di, dw);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<int> got_i(n * topk); std::vector<float> got_w(n * topk);
    CUDA_TEST_CHECK(cudaMemcpy(got_i.data(), di, got_i.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TEST_CHECK(cudaMemcpy(got_w.data(), dw, got_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dl); cudaFree(db); cudaFree(dw); cudaFree(di);
    int rc = 0; float mw = 0;
    for (int i = 0; i < n * topk; ++i) {
        if (got_i[i] != ref_ids[i]) rc = 1;
        mw = std::max(mw, std::fabs(got_w[i] - ref_w[i]));
    }
    std::printf("  id-match=%s  max weight diff=%.3e\n", rc ? "no" : "yes", mw);
    if (mw > 1e-4f) rc = 1;
    std::printf("test_moe_router: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    std::printf("test_moe_router: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
