// glmserve test — DSA sparse attention wrapper.
//
// Checks both modes:
//   * ctx <= index_topk: exact dense attention
//   * ctx > index_topk: current V1 recent-window sparse baseline
#include <cmath>
#include <cstdio>
#include <algorithm>
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
    auto cpu_attn_indexed = [](const std::vector<float>& qv,
                               const std::vector<float>& kcv,
                               const std::vector<float>& vcv,
                               const std::vector<int>& idx,
                               std::vector<float>& refv,
                               int nn, int HH, int KKVH, int hhd, int ttopk,
                               float sscale) {
        for (int t = 0; t < nn; ++t)
            for (int h = 0; h < HH; ++h) {
                int count = std::min(ttopk, t + 1);
                const float* qh = qv.data() + (t * HH + h) * hhd;
                std::vector<float> sv(count);
                float mx = -1e30f;
                for (int kk = 0; kk < count; ++kk) {
                    int j = idx[t * ttopk + kk];
                    float d = 0.0f;
                    const float* kj = kcv.data() + (j * KKVH + h) * hhd;
                    for (int e = 0; e < hhd; ++e) d += qh[e] * kj[e];
                    sv[kk] = d * sscale;
                    mx = std::max(mx, sv[kk]);
                }
                float sum = 0.0f;
                for (float& v : sv) { v = std::exp(v - mx); sum += v; }
                float* o = refv.data() + (t * HH + h) * hhd;
                for (int kk = 0; kk < count; ++kk) {
                    int j = idx[t * ttopk + kk];
                    const float* vj = vcv.data() + (j * KKVH + h) * hhd;
                    float w = sv[kk] / sum;
                    for (int e = 0; e < hhd; ++e) o[e] += w * vj[e];
                }
            }
    };

    const char* test_name = "test_attention_dsa";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    using namespace glmserve::cuda;
    float *dq = nullptr, *dq1 = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *dout1 = nullptr;
    float *part_acc = nullptr, *part_m = nullptr, *part_l = nullptr;
    int* dbt = nullptr;
    int bt = 0;
    const int max_splits = 8;
    CUDA_TEST_CHECK(cudaMalloc(&dq, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dq1, H * hd * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dk, kc.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dv, vc.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dout, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dout1, H * hd * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&part_acc, H * max_splits * hd * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&part_m, H * max_splits * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&part_l, H * max_splits * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dbt, sizeof(int)));
    CUDA_TEST_CHECK(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dq1, q.data() + (n - 1) * H * hd,
                               H * hd * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dk, kc.data(), kc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dv, vc.data(), vc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dbt, &bt, sizeof(int), cudaMemcpyHostToDevice));
    attention_dsa_paged(dq, dk, dv, dbt, n, 0, H, KVH, hd, block_size, topk, scale, dout);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(q.size());
    CUDA_TEST_CHECK(cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    attention_dsa_paged(dq1, dk, dv, dbt, 1, n - 1, H, KVH, hd, block_size, topk,
                        scale, dout1, 0, part_acc, part_m, part_l, max_splits);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<float> got1(H * hd);
    CUDA_TEST_CHECK(cudaMemcpy(got1.data(), dout1, got1.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Learned DSA selector + indexed sparse attention path. Synthetic indexer
    // data is chosen so expected top-k positions are deterministic.
    const int iH = 2, iD = 8;
    std::vector<float> iq(n * iH * iD), ik(n * iD), iw(n * iH);
    for (int t = 0; t < n; ++t) {
        for (int ih = 0; ih < iH; ++ih) {
            for (int d = 0; d < iD; ++d)
                iq[(t * iH + ih) * iD + d] = 0.01f * (1 + t + ih + d);
            iw[t * iH + ih] = ih == 0 ? 1.0f : 0.5f;
        }
        for (int d = 0; d < iD; ++d) ik[t * iD + d] = 0.02f * (1 + t) * (1 + d);
    }
    std::vector<int> ref_idx(n * topk);
    std::vector<float> ref_score(n * topk, -1e30f);
    const float iscale = 1.0f / std::sqrt((float)iD);
    for (int t = 0; t < n; ++t) {
        int count = std::min(topk, t + 1);
        std::vector<float> sc(t + 1);
        for (int j = 0; j <= t; ++j) {
            float total = 0.0f;
            for (int ih = 0; ih < iH; ++ih) {
                float dot = 0.0f;
                for (int d = 0; d < iD; ++d)
                    dot += iq[(t * iH + ih) * iD + d] * ik[j * iD + d];
                total += iw[t * iH + ih] * std::max(0.0f, dot * iscale);
            }
            sc[j] = total;
        }
        std::vector<int> ids(t + 1);
        for (int j = 0; j <= t; ++j) ids[j] = j;
        std::partial_sort(ids.begin(), ids.begin() + count, ids.end(),
                          [&](int a, int b) { return sc[a] > sc[b]; });
        ids.resize(count);
        std::sort(ids.begin(), ids.end());
        for (int k = 0; k < count; ++k) ref_idx[t * topk + k] = ids[k];
    }

    float *diq = nullptr, *dikf = nullptr, *diw = nullptr, *dscore = nullptr, *dout_idx = nullptr;
    float* dscratch = nullptr;
    __half* dik = nullptr;
    int* didx = nullptr;
    CUDA_TEST_CHECK(cudaMalloc(&diq, iq.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dikf, ik.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dik, ik.size() * sizeof(__half)));
    CUDA_TEST_CHECK(cudaMalloc(&diw, iw.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&didx, ref_idx.size() * sizeof(int)));
    CUDA_TEST_CHECK(cudaMalloc(&dscore, ref_score.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dscratch, (size_t)kDsaScoreChunk * n * sizeof(float)));
    CUDA_TEST_CHECK(cudaMalloc(&dout_idx, q.size() * sizeof(float)));
    CUDA_TEST_CHECK(cudaMemcpy(diq, iq.data(), iq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_TEST_CHECK(cudaMemcpy(dikf, ik.data(), ik.size() * sizeof(float), cudaMemcpyHostToDevice));
    convert_f32_f16(dikf, (int64_t)ik.size(), dik);
    CUDA_TEST_CHECK(cudaMemcpy(diw, iw.data(), iw.size() * sizeof(float), cudaMemcpyHostToDevice));
    dsa_select_topk(diq, dik, diw, n, 0, iH, iD, topk, iscale, 1.0f, dscratch, didx, dscore);
    attention_dsa_indexed_paged(dq, dk, dv, dbt, didx, n, 0, H, KVH, hd, block_size,
                                topk, scale, dout_idx);
    CUDA_TEST_CHECK(cudaGetLastError());
    CUDA_TEST_CHECK(cudaDeviceSynchronize());
    std::vector<int> got_idx(ref_idx.size());
    std::vector<float> got_idx_attn(q.size());
    CUDA_TEST_CHECK(cudaMemcpy(got_idx.data(), didx, got_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TEST_CHECK(cudaMemcpy(got_idx_attn.data(), dout_idx, got_idx_attn.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    cudaFree(diq); cudaFree(dikf); cudaFree(dik); cudaFree(diw); cudaFree(didx);
    cudaFree(dscore); cudaFree(dscratch); cudaFree(dout_idx);

    cudaFree(dq); cudaFree(dq1); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dout1);
    cudaFree(part_acc); cudaFree(part_m); cudaFree(part_l); cudaFree(dbt);

    float md = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) md = std::max(md, std::fabs(got[i] - ref[i]));
    float md_decode = 0.0f;
    const float* ref1 = ref.data() + (n - 1) * H * hd;
    for (size_t i = 0; i < got1.size(); ++i)
        md_decode = std::max(md_decode, std::fabs(got1[i] - ref1[i]));
    bool idx_match = got_idx == ref_idx;
    std::vector<float> ref_idx_attn(q.size(), 0.0f);
    cpu_attn_indexed(q, kc, vc, ref_idx, ref_idx_attn, n, H, KVH, hd, topk, scale);
    float md_indexed = 0.0f;
    for (size_t i = 0; i < got_idx_attn.size(); ++i)
        md_indexed = std::max(md_indexed, std::fabs(got_idx_attn[i] - ref_idx_attn[i]));
    std::printf("  max abs diff = %.3e  decode split-K diff = %.3e  learned-indexed diff = %.3e  idx-match=%s\n",
                md, md_decode, md_indexed, idx_match ? "yes" : "no");
    int rc = (md <= 1e-4f && md_decode <= 1e-4f && md_indexed <= 1e-4f && idx_match) ? 0 : 1;
    std::printf("test_attention_dsa: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    std::printf("test_attention_dsa: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
