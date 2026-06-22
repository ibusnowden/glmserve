// glmserve test — NCCL pipeline send/recv smoke.
//
// Validates the PP handoff primitive used by TP=8/PP=2 serving: stage 0 sends a
// device hidden-state buffer to stage 1, which receives and compares it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>

#include "cuda_test_utils.hpp"
#include "nccl_comm.hpp"
#endif

int main() {
#ifdef GLMSERVE_CUDA
    const char* test_name = "test_nccl_pipeline";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    glmserve::DistConfig cfg = glmserve::dist_config_from_env();
    if (cfg.world_size <= 1) {
        std::printf("test_nccl_pipeline: SKIPPED (requires PP=2 multi-rank launch)\n");
        return 0;
    }
    if (cfg.pp_size != 2 || cfg.tp_size != 1 || cfg.world_size != 2) {
        std::printf("test_nccl_pipeline: SKIPPED (requires world=2 TP=1 PP=2; got world=%d TP=%d PP=%d)\n",
                    cfg.world_size, cfg.tp_size, cfg.pp_size);
        return 0;
    }

    glmserve::Communicator comm(cfg);
    if (!comm.active()) {
        std::printf("test_nccl_pipeline: FAIL (communicator inactive)\n");
        return 1;
    }

    const int count = 64;
    std::vector<float> want(count);
    for (int i = 0; i < count; ++i) {
        want[i] = 0.125f * static_cast<float>((i % 17) - 8) + 0.01f * i;
    }

    float* d = nullptr;
    CUDA_TEST_CHECK(cudaMalloc(&d, count * sizeof(float)));
    if (cfg.is_first_stage()) {
        CUDA_TEST_CHECK(cudaMemcpy(d, want.data(), count * sizeof(float),
                                   cudaMemcpyHostToDevice));
        comm.pipeline_send_next(d, count);
    } else if (cfg.is_last_stage()) {
        CUDA_TEST_CHECK(cudaMemset(d, 0, count * sizeof(float)));
        comm.pipeline_recv_prev(d, count);
        std::vector<float> got(count);
        CUDA_TEST_CHECK(cudaMemcpy(got.data(), d, count * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        for (int i = 0; i < count; ++i) {
            max_diff = std::max(max_diff, std::fabs(got[i] - want[i]));
        }
        if (max_diff > 1e-7f) {
            std::printf("test_nccl_pipeline: FAIL rank=%d max_diff=%.9g\n",
                        cfg.rank, max_diff);
            cudaFree(d);
            return 1;
        }
        std::printf("test_nccl_pipeline: PASS rank=%d max_diff=%.9g\n",
                    cfg.rank, max_diff);
    }
    cudaFree(d);

    comm.barrier();
    if (cfg.is_first_stage()) {
        std::printf("test_nccl_pipeline: PASS rank=%d sent=%d floats\n", cfg.rank, count);
    }
    return 0;
#else
    std::printf("test_nccl_pipeline: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
