// glmserve test — dense paged attention CUDA kernel vs CPU reference.
// Single sequence, one KV block large enough to hold the prompt.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "cuda_test_utils.hpp"
#endif

int main() {
    const int n = 6, H = 4, KVH = 4, hd = 16, block_size = 16;
    const float scale = 1.0f / std::sqrt((float)hd);
    std::mt19937 rng(4);
    std::normal_distribution<float> nd(0, 1);

    std::vector<float> q(n * H * hd);
    // KV cache: [1 block, block_size, KVH, hd]
    std::vector<float> kc(block_size * KVH * hd), vc(block_size * KVH * hd);
    for (auto& v : q) v = nd(rng);
    for (int j = 0; j < n; ++j)
        for (int x = 0; x < KVH * hd; ++x) {
            kc[j * KVH * hd + x] = nd(rng);
            vc[j * KVH * hd + x] = nd(rng);
        }

    std::vector<float> ref(n * H * hd, 0.0f);
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < H; ++h) {
            const float* qh = q.data() + (t * H + h) * hd;
            std::vector<float> s(t + 1);
            float mx = -1e30f;
            for (int j = 0; j <= t; ++j) {
                float d = 0; const float* kj = kc.data() + (j * KVH + h) * hd;
                for (int e = 0; e < hd; ++e) d += qh[e] * kj[e];
                s[j] = d * scale; mx = std::max(mx, s[j]);
            }
            float sum = 0; for (int j = 0; j <= t; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            float* o = ref.data() + (t * H + h) * hd;
            for (int j = 0; j <= t; ++j) {
                const float* vj = vc.data() + (j * KVH + h) * hd;
                for (int e = 0; e < hd; ++e) o[e] += (s[j] / sum) * vj[e];
            }
        }

#ifdef GLMSERVE_CUDA
    const char* test_name = "test_attention";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    using namespace glmserve::cuda;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr; int* dbt = nullptr;
    int bt = 0;  // one block, physical id 0
    CUDA_TEST_CHECK(cudaMalloc(&dq, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dk, kc.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dv, vc.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dout, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dbt, sizeof(int)));
    CUDA_TEST_CHECK(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dk, kc.data(), kc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dv, vc.data(), vc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dbt, &bt, sizeof(int), cudaMemcpyHostToDevice));
    attention_dense_paged(dq, dk, dv, dbt, n, 0, H, KVH, hd, block_size, scale, dout);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(q.size());
    CUDA_TEST_CHECK(cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dbt);
    float md = 0; for (size_t i = 0; i < got.size(); ++i) md = std::max(md, std::fabs(got[i] - ref[i]));
    std::printf("  max abs diff = %.3e\n", md);
    int rc = md <= 1e-4f ? 0 : 1;
    std::printf("test_attention: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    std::printf("test_attention: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
