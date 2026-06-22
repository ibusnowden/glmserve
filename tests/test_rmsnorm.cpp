// glmserve test — RMSNorm CUDA kernel vs CPU reference.
// CPU-only build: compiles and reports "skipped". GPU build: runs the kernel.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "cuda_test_utils.hpp"
#endif

static int check(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    float md = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) md = std::max(md, std::fabs(a[i] - b[i]));
    std::printf("  max abs diff = %.3e (tol %.1e)\n", md, tol);
    return md <= tol ? 0 : 1;
}

int main() {
    const int n = 17, d = 256;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> x(n * d), w(d), ref(n * d);
    for (auto& v : x) v = nd(rng);
    for (auto& v : w) v = 1.0f + 0.1f * nd(rng);
    const float eps = 1e-5f;

    for (int r = 0; r < n; ++r) {
        double ss = 0;
        for (int i = 0; i < d; ++i) ss += (double)x[r * d + i] * x[r * d + i];
        float inv = 1.0f / std::sqrt(ss / d + eps);
        for (int i = 0; i < d; ++i) ref[r * d + i] = x[r * d + i] * inv * w[i];
    }

#ifdef GLMSERVE_CUDA
    const char* test_name = "test_rmsnorm";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    using namespace glmserve::cuda;
    float *dx = nullptr, *dw = nullptr, *dy = nullptr;
    CUDA_TEST_CHECK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dw, w.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dy, x.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    rmsnorm(dx, dw, dy, n, d, eps);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(n * d);
    CUDA_TEST_CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    int rc = check(got, ref, 1e-4f);
    std::printf("test_rmsnorm: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    (void)check;
    std::printf("test_rmsnorm: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
