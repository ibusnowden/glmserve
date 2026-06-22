// glmserve test — DSA sparse attention wrapper.
//
// Checks both modes:
//   * ctx <= index_topk: exact dense attention
//   * ctx > index_topk: current V1 recent-window sparse baseline
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "cuda_test_utils.hpp"
#endif

static void cpu_attn_window(const std::vector<float>& q,
                            const std::vector<float>& kc,
                            const std::vector<float>& vc,
                            std::vector<float>& ref,
                            int n, int H, int KVH, int hd, int topk,
                            float scale) {
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < H; ++h) {
            const float* qh = q.data() + (t * H + h) * hd;
            int lo = (t + 1 > topk) ? (t + 1 - topk) : 0;
            std::vector<float> s(t - lo + 1);
            float mx = -1e30f;
            for (int j = lo; j <= t; ++j) {
                float d = 0.0f;
                const float* kj = kc.data() + (j * KVH + h) * hd;
                for (int e = 0; e < hd; ++e) d += qh[e] * kj[e];
                s[j - lo] = d * scale;
                mx = std::max(mx, s[j - lo]);
            }
            float sum = 0.0f;
            for (float& v : s) { v = std::exp(v - mx); sum += v; }
            float* o = ref.data() + (t * H + h) * hd;
            for (int j = lo; j <= t; ++j) {
                const float* vj = vc.data() + (j * KVH + h) * hd;
                float w = s[j - lo] / sum;
                for (int e = 0; e < hd; ++e) o[e] += w * vj[e];
            }
        }
}

int main() {
    const int n = 9, H = 4, KVH = 4, hd = 16, block_size = 16, topk = 3;
    const float scale = 1.0f / std::sqrt((float)hd);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);

    std::vector<float> q(n * H * hd);
    std::vector<float> kc(block_size * KVH * hd), vc(block_size * KVH * hd);
    for (auto& v : q) v = nd(rng);
    for (int j = 0; j < n; ++j)
        for (int x = 0; x < KVH * hd; ++x) {
            kc[j * KVH * hd + x] = nd(rng);
            vc[j * KVH * hd + x] = nd(rng);
        }

    std::vector<float> ref(n * H * hd, 0.0f);
    cpu_attn_window(q, kc, vc, ref, n, H, KVH, hd, topk, scale);

#ifdef GLMSERVE_CUDA
    const char* test_name = "test_attention_dsa";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    using namespace glmserve::cuda;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    int* dbt = nullptr;
    int bt = 0;
    CUDA_TEST_CHECK(cudaMalloc(&dq, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dk, kc.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dv, vc.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dout, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dbt, sizeof(int)));
    CUDA_TEST_CHECK(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dk, kc.data(), kc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dv, vc.data(), vc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dbt, &bt, sizeof(int), cudaMemcpyHostToDevice));
    attention_dsa_paged(dq, dk, dv, dbt, n, 0, H, KVH, hd, block_size, topk, scale, dout);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(q.size());
    CUDA_TEST_CHECK(cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dbt);

    float md = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) md = std::max(md, std::fabs(got[i] - ref[i]));
    std::printf("  max abs diff = %.3e\n", md);
    int rc = md <= 1e-4f ? 0 : 1;
    std::printf("test_attention_dsa: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    std::printf("test_attention_dsa: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
