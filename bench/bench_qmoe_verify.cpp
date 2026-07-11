// Real-dimension GLM-5.2 TP=8 quant-MoE verify microbenchmark.
//
// Usage (one process per setting because kernel policy is cached):
//   build/gpu/bench/bench_qmoe_verify 64 10
//   GLMSERVE_MMA_MOE_MIN_TS=512 build/gpu/bench/bench_qmoe_verify 64 10
//
// The E=256 routing distribution matters: a 64-token/top-8 verify pass has
// only two slots per expert on average, unlike small-E unit tests that make a
// 64-token tile look artificially dense.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>

#include "gguf_quant.hpp"
#include "kernels.cuh"
#endif

int main(int argc, char** argv) {
#ifndef GLMSERVE_CUDA
    std::puts("bench_qmoe_verify: CUDA build required");
    return 0;
#else
    using namespace glmserve;
    using namespace glmserve::cuda;
    const int n = argc > 1 ? std::max(1, std::atoi(argv[1])) : 64;
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 10;
    const int E = 256, topk = 8, hidden = 6144, moe_inter = 256;
    const uint32_t gate_type = 18, up_type = 18, down_type = 23;
    const int nts = n * topk;

    const int64_t gate_rb = (hidden / static_cast<int>(gguf_type_block_elements(gate_type))) *
                            static_cast<int64_t>(gguf_type_block_bytes(gate_type));
    const int64_t up_rb = gate_rb;
    const int64_t down_rb = (moe_inter / static_cast<int>(gguf_type_block_elements(down_type))) *
                            static_cast<int64_t>(gguf_type_block_bytes(down_type));

    int8_t *xq = nullptr, *hq = nullptr;
    float *xs = nullptr, *topk_w = nullptr, *h_act = nullptr, *hs = nullptr;
    float *dpart = nullptr, *out = nullptr, *gu_part = nullptr;
    int *topk_ids = nullptr, *dispatch = nullptr;
    uint8_t *gate_q = nullptr, *up_q = nullptr, *down_q = nullptr;
    auto alloc = [](void** p, size_t bytes) {
        const cudaError_t e = cudaMalloc(p, bytes);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc(%zu): %s\n", bytes, cudaGetErrorString(e));
            std::exit(2);
        }
        cudaMemset(*p, 0, bytes);
    };
    alloc(reinterpret_cast<void**>(&xq), static_cast<size_t>(n) * hidden);
    alloc(reinterpret_cast<void**>(&xs), static_cast<size_t>(n) * (hidden / 32) * sizeof(float));
    alloc(reinterpret_cast<void**>(&topk_ids), static_cast<size_t>(nts) * sizeof(int));
    alloc(reinterpret_cast<void**>(&topk_w), static_cast<size_t>(nts) * sizeof(float));
    alloc(reinterpret_cast<void**>(&gate_q), static_cast<size_t>(E) * moe_inter * gate_rb);
    alloc(reinterpret_cast<void**>(&up_q), static_cast<size_t>(E) * moe_inter * up_rb);
    alloc(reinterpret_cast<void**>(&down_q), static_cast<size_t>(E) * hidden * down_rb);
    alloc(reinterpret_cast<void**>(&h_act), static_cast<size_t>(nts) * moe_inter * sizeof(float));
    alloc(reinterpret_cast<void**>(&hq), static_cast<size_t>(nts) * moe_inter);
    alloc(reinterpret_cast<void**>(&hs), static_cast<size_t>(nts) * (moe_inter / 32) * sizeof(float));
    alloc(reinterpret_cast<void**>(&dpart), static_cast<size_t>(nts) * hidden * sizeof(float));
    alloc(reinterpret_cast<void**>(&out), static_cast<size_t>(n) * hidden * sizeof(float));
    alloc(reinterpret_cast<void**>(&dispatch),
          static_cast<size_t>(3 * E + 1 + nts) * sizeof(int));
    alloc(reinterpret_cast<void**>(&gu_part),
          static_cast<size_t>(2) * kMoeSplitK * kMoeSplitKMaxTs * moe_inter * sizeof(float));

    std::vector<int> ids(nts);
    std::vector<float> weights(nts, 1.0f / topk);
    // Exactly uniform at n=64: two routed slots/expert, representative of the
    // sparsity that makes the tensor-core cutoff non-obvious.
    for (int t = 0; t < n; ++t)
        for (int s = 0; s < topk; ++s)
            ids[t * topk + s] = (t * 37 + s * 17) & (E - 1);
    cudaMemcpy(topk_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(topk_w, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);

    auto launch = [&] {
        moe_expert_ffn_q_i8(gate_type, up_type, down_type, xq, xs, topk_ids, topk_w,
                            gate_q, up_q, down_q, n, topk, hidden, moe_inter, E,
                            gate_rb, up_rb, down_rb, h_act, hq, hs, dpart, out,
                            dispatch, gu_part);
    };
    launch();
    launch();
    cudaDeviceSynchronize();
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    for (int i = 0; i < reps; ++i) launch();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, beg, end);
    const char* cutoff = std::getenv("GLMSERVE_MMA_MOE_MIN_TS");
    const char* tile = std::getenv("GLMSERVE_MOE_TILE");
    std::printf("qmoe verify: n=%d nts=%d E=%d hidden=%d moe_inter=%d cutoff=%s tile=%s "
                "%.3f ms/pass %.1f tok/s-equivalent\n",
                n, nts, E, hidden, moe_inter, cutoff ? cutoff : "2048(default)",
                tile ? tile : "auto", ms / reps, n * 1000.0 / (ms / reps));

    cudaEventDestroy(beg); cudaEventDestroy(end);
    cudaFree(xq); cudaFree(xs); cudaFree(topk_ids); cudaFree(topk_w);
    cudaFree(gate_q); cudaFree(up_q); cudaFree(down_q); cudaFree(h_act);
    cudaFree(hq); cudaFree(hs); cudaFree(dpart); cudaFree(out);
    cudaFree(dispatch); cudaFree(gu_part);
    return 0;
#endif
}
