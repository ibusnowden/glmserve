// glmserve test — NCCL communicator smoke test.
//
// On a GPU build with a visible device, forces a one-rank NCCL communicator so
// the test proves ncclCommInitRank + barrier execute on real hardware. Multi-rank
// TP/PP tests can reuse the same Communicator with launcher-provided env vars.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "cuda_test_utils.hpp"
#include "nccl_comm.hpp"
#endif

int main() {
#ifdef GLMSERVE_CUDA
    const char* test_name = "test_nccl_comm";
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
        std::printf("test_nccl_comm: FAIL (communicator inactive)\n");
        return 1;
    }
    comm.barrier();

    if (cfg.tp_size > 1) {
        float in = static_cast<float>(cfg.tp_rank() + 1);
        float* d = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&d, sizeof(float)));
        CUDA_TEST_CHECK(cudaMemcpy(d, &in, sizeof(float), cudaMemcpyHostToDevice));
        comm.all_reduce_sum(d, 1);
        float got = 0.0f;
        CUDA_TEST_CHECK(cudaMemcpy(&got, d, sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d);
        float want = static_cast<float>(cfg.tp_size * (cfg.tp_size + 1) / 2);
        if (std::fabs(got - want) > 1e-5f) {
            std::printf("test_nccl_comm: FAIL rank=%d got %.6f want %.6f\n",
                        cfg.rank, got, want);
            return 1;
        }
    }

    // Custom P2P all-reduce path (GLMSERVE_CUSTOM_AR=1): decode-sized counts
    // take the one-shot IPC kernel; repeat to exercise both slot parities and
    // interleave an ineligible count to prove NCCL fallback stays ordered.
    if (cfg.tp_size > 1) {
        const int64_t counts[] = {6144, 8 * 6144, 6144 + 4, 6144, 3, 6144};
        for (int round = 0; round < 2; ++round) {
            for (int64_t n : counts) {
                std::vector<float> h(static_cast<size_t>(n));
                for (int64_t i = 0; i < n; ++i)
                    h[static_cast<size_t>(i)] =
                        static_cast<float>(cfg.tp_rank() + 1) * 0.5f +
                        static_cast<float>(i % 97) * 0.25f;
                float* d = nullptr;
                CUDA_TEST_CHECK(cudaMalloc(&d, static_cast<size_t>(n) * sizeof(float)));
                CUDA_TEST_CHECK(cudaMemcpy(d, h.data(), static_cast<size_t>(n) * sizeof(float),
                                           cudaMemcpyHostToDevice));
                comm.all_reduce_sum(d, n);
                std::vector<float> got(static_cast<size_t>(n));
                CUDA_TEST_CHECK(cudaMemcpy(got.data(), d, static_cast<size_t>(n) * sizeof(float),
                                           cudaMemcpyDeviceToHost));
                cudaFree(d);
                const float tsum = static_cast<float>(cfg.tp_size * (cfg.tp_size + 1) / 2);
                for (int64_t i = 0; i < n; ++i) {
                    const float want = tsum * 0.5f +
                                       static_cast<float>(cfg.tp_size) *
                                           static_cast<float>(i % 97) * 0.25f;
                    if (std::fabs(got[static_cast<size_t>(i)] - want) > 1e-3f) {
                        std::printf("test_nccl_comm: FAIL rank=%d round=%d n=%lld i=%lld "
                                    "got %.6f want %.6f\n",
                                    cfg.rank, round, (long long)n, (long long)i,
                                    got[static_cast<size_t>(i)], want);
                        return 1;
                    }
                }
            }
        }
    }

    // Row-wise all-reduce parity: each row of all_reduce_sum_rows must be
    // BITWISE identical to a standalone all_reduce_sum of that row (that is
    // the property the spec-decode verify pass relies on; the wide [n*elems]
    // reduce is allowed to differ in the last bits, which is the whole point).
    if (cfg.tp_size > 1) {
        const int64_t rows = 5, elems = 6144;
        std::vector<float> h(static_cast<size_t>(rows * elems));
        for (int64_t i = 0; i < rows * elems; ++i)
            h[static_cast<size_t>(i)] =
                std::sin(static_cast<float>(i % 1013) * 0.37f +
                         static_cast<float>(cfg.tp_rank())) *
                (1.0f + static_cast<float>(i % 7));
        float* d_rows = nullptr;
        float* d_one = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&d_rows, static_cast<size_t>(rows * elems) * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&d_one, static_cast<size_t>(elems) * sizeof(float)));
        CUDA_TEST_CHECK(cudaMemcpy(d_rows, h.data(),
                                   static_cast<size_t>(rows * elems) * sizeof(float),
                                   cudaMemcpyHostToDevice));
        comm.all_reduce_sum_rows(d_rows, rows, elems);
        std::vector<float> got(static_cast<size_t>(rows * elems));
        CUDA_TEST_CHECK(cudaMemcpy(got.data(), d_rows,
                                   static_cast<size_t>(rows * elems) * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int64_t r = 0; r < rows; ++r) {
            CUDA_TEST_CHECK(cudaMemcpy(d_one, h.data() + r * elems,
                                       static_cast<size_t>(elems) * sizeof(float),
                                       cudaMemcpyHostToDevice));
            comm.all_reduce_sum(d_one, elems);
            std::vector<float> ref(static_cast<size_t>(elems));
            CUDA_TEST_CHECK(cudaMemcpy(ref.data(), d_one,
                                       static_cast<size_t>(elems) * sizeof(float),
                                       cudaMemcpyDeviceToHost));
            for (int64_t i = 0; i < elems; ++i) {
                if (got[static_cast<size_t>(r * elems + i)] != ref[static_cast<size_t>(i)]) {
                    std::printf("test_nccl_comm: FAIL rank=%d rows-parity r=%lld i=%lld "
                                "rows %.9g single %.9g\n",
                                cfg.rank, (long long)r, (long long)i,
                                got[static_cast<size_t>(r * elems + i)],
                                ref[static_cast<size_t>(i)]);
                    return 1;
                }
            }
        }
        cudaFree(d_rows);
        cudaFree(d_one);
        std::printf("test_nccl_comm: rank=%d rows-parity (5x6144) exact\n", cfg.rank);
    }

    // Latency probe: decode-sized all-reduce, µs/call (custom vs NCCL is
    // selected by GLMSERVE_CUSTOM_AR; compare two runs of this test).
    if (cfg.tp_size > 1) {
        const int64_t n = 6144;
        float* d = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&d, static_cast<size_t>(n) * sizeof(float)));
        CUDA_TEST_CHECK(cudaMemset(d, 0, static_cast<size_t>(n) * sizeof(float)));
        for (int i = 0; i < 20; ++i) comm.all_reduce_sum(d, n);  // warmup
        CUDA_TEST_CHECK(cudaDeviceSynchronize());
        comm.barrier();
        const int iters = 500;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) comm.all_reduce_sum(d, n);
        CUDA_TEST_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        const double us =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::printf("test_nccl_comm: rank=%d all_reduce(6144 fp32) = %.1f us/call\n",
                    cfg.rank, us);
        cudaFree(d);
    }

    std::printf("test_nccl_comm: PASS\n");
    return 0;
#else
    std::printf("test_nccl_comm: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
