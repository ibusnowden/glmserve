// glmserve test — tensor-parallel row-linear reconstruction.
//
// Each TP rank owns one contiguous slice of the input dimension and the matching
// slice of a full output projection weight. Local GEMM produces a partial
// output; TP all-reduce reconstructs the dense reference output on every rank.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>

#include "cuda_test_utils.hpp"
#include "kernels.cuh"
#include "nccl_comm.hpp"
#endif

int main() {
#ifdef GLMSERVE_CUDA
    const char* test_name = "test_tp_linear";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    glmserve::DistConfig cfg = glmserve::dist_config_from_env();
    if (cfg.world_size <= 1) {
        setenv("GLMSERVE_NCCL_SINGLE", "1", 1);
        cfg.rank = 0;
        cfg.world_size = 1;
        cfg.local_rank = 0;
        cfg.tp_size = 1;
        cfg.pp_size = 1;
    }

    glmserve::Communicator comm(cfg);
    if (!comm.active()) {
        std::printf("test_tp_linear: FAIL (communicator inactive)\n");
        return 1;
    }

    const int n = 3;
    const int local_in = 4;
    const int in = local_in * cfg.tp_size;
    const int out = 5;
    const int tp0 = cfg.tp_rank() * local_in;

    std::vector<float> x_full(n * in);
    std::vector<float> w_full(out * in);
    for (int i = 0; i < n * in; ++i) {
        x_full[i] = 0.01f * static_cast<float>((i % 13) - 6);
    }
    for (int i = 0; i < out * in; ++i) {
        w_full[i] = 0.02f * static_cast<float>((i % 17) - 8);
    }

    std::vector<float> x_local(n * local_in);
    std::vector<float> w_local(out * local_in);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < local_in; ++col) {
            x_local[row * local_in + col] = x_full[row * in + tp0 + col];
        }
    }
    for (int row = 0; row < out; ++row) {
        for (int col = 0; col < local_in; ++col) {
            w_local[row * local_in + col] = w_full[row * in + tp0 + col];
        }
    }

    std::vector<float> want(n * out, 0.0f);
    for (int row = 0; row < n; ++row) {
        for (int o = 0; o < out; ++o) {
            float acc = 0.0f;
            for (int col = 0; col < in; ++col) {
                acc += x_full[row * in + col] * w_full[o * in + col];
            }
            want[row * out + o] = acc;
        }
    }

    float *dx = nullptr, *dw = nullptr, *dy = nullptr;
    CUDA_TEST_CHECK(cudaMalloc(&dx, x_local.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dw, w_local.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dy, want.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMemcpy(dx, x_local.data(), x_local.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dw, w_local.data(), w_local.size() * sizeof(float),
                               cudaMemcpyHostToDevice));

    glmserve::cuda::gemm_fp32(dx, dw, nullptr, dy, n, local_in, out);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    comm.all_reduce_sum(dy, static_cast<int64_t>(want.size()));

    std::vector<float> got(want.size());
    CUDA_TEST_CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);

    float max_diff = 0.0f;
    for (size_t i = 0; i < want.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(got[i] - want[i]));
    }
    if (max_diff > 1e-6f) {
        std::printf("test_tp_linear: FAIL rank=%d max_diff=%.9g\n", cfg.rank, max_diff);
        return 1;
    }

    std::printf("test_tp_linear: PASS rank=%d tp_rank=%d max_diff=%.9g\n",
                cfg.rank, cfg.tp_rank(), max_diff);
    return 0;
#else
    std::printf("test_tp_linear: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
