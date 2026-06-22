// glmserve test — NCCL communicator smoke test.
//
// On a GPU build with a visible device, forces a one-rank NCCL communicator so
// the test proves ncclCommInitRank + barrier execute on real hardware. Multi-rank
// TP/PP tests can reuse the same Communicator with launcher-provided env vars.
#include <cstdio>
#include <cstdlib>
#include <cmath>

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

    std::printf("test_nccl_comm: PASS\n");
    return 0;
#else
    std::printf("test_nccl_comm: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
