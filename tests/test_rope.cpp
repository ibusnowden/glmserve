// glmserve test — partial-rotary RoPE CUDA kernel vs CPU reference.
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
    const int n = 5, n_heads = 4, head_dim = 16, rot = 8;
    const double theta = 10000.0;
    std::mt19937 rng(2);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> x(n * n_heads * head_dim), ref;
    for (auto& v : x) v = nd(rng);
    std::vector<long long> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i;
    ref = x;

    int half = rot / 2;
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < n_heads; ++h) {
            float* v = ref.data() + (t * n_heads + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                double freq = std::pow(theta, (2.0 * i) / rot);
                double ang = pos[t] / freq;
                float c = std::cos(ang), s = std::sin(ang);
                float x0 = v[i], x1 = v[i + half];
                v[i] = x0 * c - x1 * s;
                v[i + half] = x1 * c + x0 * s;
            }
        }

#ifdef GLMSERVE_CUDA
    const char* test_name = "test_rope";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    using namespace glmserve::cuda;
    float* dx = nullptr; long long* dp = nullptr;
    CUDA_TEST_CHECK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dp, n * sizeof(long long)));
    CUDA_TEST_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dp, pos.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    rope(dx, (const int64_t*)dp, n, n_heads, head_dim, rot, theta);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(x.size());
    CUDA_TEST_CHECK(cudaMemcpy(got.data(), dx, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dp);
    float md = 0; for (size_t i = 0; i < got.size(); ++i) md = std::max(md, std::fabs(got[i] - ref[i]));
    std::printf("  max abs diff = %.3e\n", md);
    int rc = md <= 1e-4f ? 0 : 1;
    std::printf("test_rope: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    std::printf("test_rope: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
