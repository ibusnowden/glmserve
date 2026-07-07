// glmserve test — GGUF quant GEMM (gemm_q + moe_expert_ffn_q) vs CPU reference.
// CPU-only build: compiles and reports "skipped". GPU build: runs the kernels.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "gguf_quant.hpp"

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "cuda_test_utils.hpp"
#endif

static int check(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    float md = 0.0f, mx = 1.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        md = std::max(md, std::fabs(a[i] - b[i]));
        mx = std::max(mx, std::fabs(b[i]));
    }
    float rel = md / mx;
    std::printf("  max abs diff = %.3e (rel %.3e, max %.3e)\n", md, rel, mx);
    return rel <= tol ? 0 : 1;
}

// Build a quant weight [out, in] from random bytes and return (data, row_bytes).
// in must be a multiple of the block size.
static std::vector<uint8_t> make_quant(uint32_t type, int out, int in, int64_t& row_bytes,
                                       std::mt19937& rng) {
    uint64_t be = glmserve::gguf_type_block_elements(type);
    uint64_t bb = glmserve::gguf_type_block_bytes(type);
    if (in % be != 0) in = (in / be + 1) * be;
    row_bytes = (in / be) * bb;
    std::vector<uint8_t> data((size_t)out * row_bytes);
    for (auto& b : data) b = (uint8_t)(rng() & 0xFF);
    return data;
}

// CPU reference: y[o] = sum_i dequant_row(type, data, in, o)[i] * x[i]  (+bias).
// Accumulates in float to match the GPU kernel's float reduction order tolerance.
static std::vector<float> cpu_gemm_q(uint32_t type, const uint8_t* data, int64_t row_bytes,
                                     const float* x, const float* bias, int n, int in, int out) {
    std::vector<float> y((size_t)n * out, 0.0f);
    for (int t = 0; t < n; ++t) {
        for (int o = 0; o < out; ++o) {
            std::vector<float> row = glmserve::gguf_dequantize_row(type, data + (size_t)o * row_bytes, in, 0);
            float acc = 0.0f;
            for (int i = 0; i < in; ++i) acc += row[i] * x[(size_t)t * in + i];
            y[(size_t)t * out + o] = acc + (bias ? bias[o] : 0.0f);
        }
    }
    return y;
}

int main() {
    std::printf("test_qgemm: GGUF quant GEMM vs CPU dequant reference\n");
    std::mt19937 rng(12345);
    const int in = 512, out = 48, n = 3;
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> x((size_t)n * in);
    for (auto& v : x) v = nd(rng);
    std::vector<float> bias(out);
    for (auto& v : bias) v = 0.05f * nd(rng);

    // Types present in the real GLM-5.2 UD-Q3_K_XL GGUF.
    const uint32_t types[] = {8, 11, 12, 13, 14, 18, 23, 1};  // Q8_0 Q3_K Q4_K Q5_K Q6_K IQ3_XXS IQ4_XS F16
    const char* names[] = {"Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ3_XXS", "IQ4_XS", "F16"};
    int rc = 0;
#ifdef GLMSERVE_CUDA
    const char* test_name = "test_qgemm";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;
    using namespace glmserve::cuda;

    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ++ti) {
        uint32_t type = types[ti];
        int64_t row_bytes;
        std::vector<uint8_t> data = make_quant(type, out, in, row_bytes, rng);
        std::printf("  [%s] dequant_row_q...", names[ti]); std::fflush(stdout);

        // Direct dequant_row_q vs CPU gguf_dequantize_prefix (one row, element-wise).
        {
            std::vector<float> cpu_row = glmserve::gguf_dequantize_prefix(type, data.data(), in);
            float* drow = nullptr;
            uint8_t* dqrow = nullptr;
            CUDA_TEST_CHECK(cudaMalloc(&drow, in * sizeof(float)));
            CUDA_TEST_CHECK(cudaMalloc(&dqrow, data.size()));
            CUDA_TEST_CHECK(cudaMemcpy(dqrow, data.data(), data.size(), cudaMemcpyHostToDevice));
            dequant_row_q(type, dqrow, drow, in, row_bytes);
            CUDA_TEST_CHECK(cudaGetLastError());
            CUDA_TEST_CHECK(cudaDeviceSynchronize());
            std::vector<float> gpu_row(in);
            CUDA_TEST_CHECK(cudaMemcpy(gpu_row.data(), drow, in * sizeof(float), cudaMemcpyDeviceToHost));
            cudaFree(drow); cudaFree(dqrow);
            float md = 0.0f, mx = 1.0f; int mdi = 0;
            for (int i = 0; i < in; ++i) {
                float d = std::fabs(cpu_row[i] - gpu_row[i]);
                if (d > md) { md = d; mdi = i; }
                mx = std::max(mx, std::fabs(cpu_row[i]));
            }
            float rel = md / mx;
            std::printf("  %s dequant_row_q: max abs diff = %.3e (rel %.3e)", names[ti], md, rel);
            if (rel > 1e-4f && type == 11)
                std::printf(" at elem %d (cpu=%.6f gpu=%.6f)", mdi, cpu_row[mdi], gpu_row[mdi]);
            std::printf("\n");
            if (rel > 1e-4f) rc = 1;
        }

        std::vector<float> ref = cpu_gemm_q(type, data.data(), row_bytes, x.data(), bias.data(), n, in, out);

        float* dx = nullptr; uint8_t* dq = nullptr; float* db = nullptr; float* dy = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&dx, x.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&dq, data.size()));
        CUDA_TEST_CHECK(cudaMalloc(&db, bias.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&dy, ref.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(dq, data.data(), data.size(), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
        gemm_q(type, dx, dq, db, dy, n, in, out, row_bytes);
        CUDA_TEST_CHECK(cudaGetLastError());
        CUDA_TEST_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(ref.size());
        CUDA_TEST_CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dx); cudaFree(dq); cudaFree(db); cudaFree(dy);
        std::printf("  %s gemm_q [out=%d in=%d n=%d]:", names[ti], out, in, n);
        rc |= check(got, ref, 1e-3f);
    }

    // ---- MoE expert FFN (merged per-expert quant tensors) -----------------
    // gate/up: IQ3_XXS [E, moe_inter, hidden]; down: IQ4_XS [E, hidden, moe_inter].
    {
        // n chosen so n*topk >= 64: exercises the expert-major dispatch path
        // (token-major is still checked below by passing dispatch = nullptr).
        const int E = 4, topk = 2, hidden = 512, moe_inter = 256, n = 48;
        uint32_t gtype = 18, utype = 18, dtype = 23;  // IQ3_XXS gate/up, IQ4_XS down
        int64_t g_rb, u_rb, d_rb;
        std::vector<uint8_t> gate_q = make_quant(gtype, E * moe_inter, hidden, g_rb, rng);
        std::vector<uint8_t> up_q   = make_quant(utype, E * moe_inter, hidden, u_rb, rng);
        std::vector<uint8_t> down_q = make_quant(dtype, E * hidden, moe_inter, d_rb, rng);
        std::vector<int> topk_ids(n * topk);
        for (auto& v : topk_ids) v = (int)(rng() % E);
        std::vector<float> topk_w(n * topk);
        for (auto& v : topk_w) v = 0.5f;
        std::vector<float> xin((size_t)n * hidden);
        for (auto& v : xin) v = nd(rng);

        // CPU reference.
        std::vector<float> ref((size_t)n * hidden, 0.0f);
        for (int t = 0; t < n; ++t) {
            for (int s = 0; s < topk; ++s) {
                int e = topk_ids[t * topk + s];
                float w = topk_w[t * topk + s];
                std::vector<float> h_act(moe_inter);
                for (int f = 0; f < moe_inter; ++f) {
                    auto grow = glmserve::gguf_dequantize_row(gtype, gate_q.data() + (size_t)(e * moe_inter + f) * g_rb, hidden, 0);
                    auto urow = glmserve::gguf_dequantize_row(utype, up_q.data() + (size_t)(e * moe_inter + f) * u_rb, hidden, 0);
                    double g = 0, u = 0;
                    for (int i = 0; i < hidden; ++i) { g += grow[i] * xin[t * hidden + i]; u += urow[i] * xin[t * hidden + i]; }
                    float gf = (float)g;
                    float sv = gf / (1.0f + std::exp(-gf));   // silu, matches the kernel
                    h_act[f] = sv * (float)u;
                }
                for (int o = 0; o < hidden; ++o) {
                    auto drow = glmserve::gguf_dequantize_row(dtype, down_q.data() + (size_t)(e * hidden + o) * d_rb, moe_inter, 0);
                    double acc = 0;
                    for (int f = 0; f < moe_inter; ++f) acc += drow[f] * h_act[f];
                    ref[t * hidden + o] += w * (float)acc;
                }
            }
        }

        float* dx = nullptr; int* dti = nullptr; float* dtw = nullptr;
        uint8_t *dg = nullptr, *du = nullptr, *dd = nullptr;
        float* dh = nullptr; float* dy = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&dx, xin.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&dti, topk_ids.size() * sizeof(int)));
        CUDA_TEST_CHECK(cudaMalloc(&dtw, topk_w.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&dg, gate_q.size()));
        CUDA_TEST_CHECK(cudaMalloc(&du, up_q.size()));
        CUDA_TEST_CHECK(cudaMalloc(&dd, down_q.size()));
        CUDA_TEST_CHECK(cudaMalloc(&dh, (size_t)n * topk * moe_inter * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&dy, ref.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMemcpy(dx, xin.data(), xin.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(dti, topk_ids.data(), topk_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(dtw, topk_w.data(), topk_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(dg, gate_q.data(), gate_q.size(), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(du, up_q.data(), up_q.size(), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(dd, down_q.data(), down_q.size(), cudaMemcpyHostToDevice));
        moe_expert_ffn_q(gtype, utype, dtype, dx, dti, dtw, dg, du, dd,
                         n, topk, hidden, moe_inter, E, g_rb, u_rb, d_rb, dh, dy);
        CUDA_TEST_CHECK(cudaGetLastError());
        CUDA_TEST_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(ref.size());
        CUDA_TEST_CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("  moe_expert_ffn_q [E=%d topk=%d hidden=%d moe_inter=%d n=%d] token-major:",
                    E, topk, hidden, moe_inter, n);
        rc |= check(got, ref, 1e-2f);

        // Expert-major path (dispatch scratch provided, n*topk >= 64).
        int* ddisp = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&ddisp, (3 * E + 1 + n * topk) * sizeof(int)));
        moe_expert_ffn_q(gtype, utype, dtype, dx, dti, dtw, dg, du, dd,
                         n, topk, hidden, moe_inter, E, g_rb, u_rb, d_rb, dh, dy, ddisp);
        CUDA_TEST_CHECK(cudaGetLastError());
        CUDA_TEST_CHECK(cudaDeviceSynchronize());
        std::vector<float> got_em(ref.size());
        CUDA_TEST_CHECK(cudaMemcpy(got_em.data(), dy, got_em.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("  moe_expert_ffn_q expert-major vs CPU ref:");
        rc |= check(got_em, ref, 1e-2f);
        std::printf("  moe_expert_ffn_q expert-major vs token-major:");
        rc |= check(got_em, got, 1e-4f);
        cudaFree(ddisp);
        cudaFree(dx); cudaFree(dti); cudaFree(dtw); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dh); cudaFree(dy);
    }

    // ---- TP shard consistency (col/row slicing + strided 2D repack) --------
    // Column-parallel: rows [r0, r0+out_l) of y_full == gemm on qdata+r0*rb.
    // Row-parallel: sum over ranks of gemm(repacked W[:, c0:c0+in_l], x[c0:])
    // == y_full (the repack is the same cudaMemcpy2D the engine upload uses).
    {
        const int tp = 4, out2 = 32, in2 = 2048, n2 = 1;
        const uint32_t type = 18;  // IQ3_XXS, the dominant expert format
        int64_t rb;
        std::vector<uint8_t> data = make_quant(type, out2, in2, rb, rng);
        std::vector<float> x2((size_t)n2 * in2);
        for (auto& v : x2) v = nd(rng);
        std::vector<float> ref = cpu_gemm_q(type, data.data(), rb, x2.data(), nullptr, n2, in2, out2);

        const uint64_t be = glmserve::gguf_type_block_elements(type);
        const uint64_t bb = glmserve::gguf_type_block_bytes(type);
        const int64_t in_l = in2 / tp;
        const int64_t rb_l = (in_l / (int64_t)be) * (int64_t)bb;
        std::vector<float> ysum((size_t)n2 * out2, 0.0f);
        float* dx = nullptr; uint8_t* dq = nullptr; float* dy = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&dx, (size_t)n2 * in_l * sizeof(float)));
        CUDA_TEST_CHECK(cudaMalloc(&dq, (size_t)out2 * rb_l));
        CUDA_TEST_CHECK(cudaMalloc(&dy, ref.size() * sizeof(float)));
        for (int r = 0; r < tp; ++r) {
            const int64_t c0 = r * in_l;
            const int64_t skip = (c0 / (int64_t)be) * (int64_t)bb;
            // strided host view -> contiguous device rows (engine upload path)
            CUDA_TEST_CHECK(cudaMemcpy2D(dq, rb_l, data.data() + skip, rb, rb_l, out2,
                                         cudaMemcpyHostToDevice));
            std::vector<float> xs((size_t)n2 * in_l);
            for (int t = 0; t < n2; ++t)
                std::copy(x2.begin() + t * in2 + c0, x2.begin() + t * in2 + c0 + in_l,
                          xs.begin() + (size_t)t * in_l);
            CUDA_TEST_CHECK(cudaMemcpy(dx, xs.data(), xs.size() * sizeof(float), cudaMemcpyHostToDevice));
            gemm_q(type, dx, dq, nullptr, dy, n2, in_l, out2, rb_l);
            CUDA_TEST_CHECK(cudaDeviceSynchronize());
            std::vector<float> part(ref.size());
            CUDA_TEST_CHECK(cudaMemcpy(part.data(), dy, part.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < ysum.size(); ++i) ysum[i] += part[i];
        }
        cudaFree(dx); cudaFree(dq); cudaFree(dy);
        std::printf("  row-parallel shard consistency [tp=%d in=%d out=%d IQ3_XXS]:", tp, in2, out2);
        rc |= check(ysum, ref, 1e-3f);
    }

    // ---- embed_gather_q at the real GLM-5.2 width (H=6144, Q8_0) -----------
    {
        const int vocab = 32, H = 6144;
        const uint32_t type = 8;  // Q8_0, the embed table format
        int64_t rb;
        std::vector<uint8_t> table = make_quant(type, vocab, H, rb, rng);
        // Clamp each block's f16 scale to a finite value (random bytes can
        // encode inf/nan scales, which poison the comparison).
        for (size_t off = 0; off + 34 <= table.size(); off += 34) {
            table[off + 1] &= 0x3F;  // clear f16 exponent high bits
        }
        // Same-width dequant_row_q first: isolates group-walk bugs from
        // token-indexing bugs.
        {
            std::vector<float> cref = glmserve::gguf_dequantize_row(type, table.data(), H, 5);
            float* drow = nullptr; uint8_t* dq2 = nullptr;
            CUDA_TEST_CHECK(cudaMalloc(&drow, H * sizeof(float)));
            CUDA_TEST_CHECK(cudaMalloc(&dq2, table.size()));
            CUDA_TEST_CHECK(cudaMemcpy(dq2, table.data(), table.size(), cudaMemcpyHostToDevice));
            dequant_row_q(type, dq2 + 5 * rb, drow, H, rb);
            CUDA_TEST_CHECK(cudaDeviceSynchronize());
            std::vector<float> grow(H);
            CUDA_TEST_CHECK(cudaMemcpy(grow.data(), drow, H * sizeof(float), cudaMemcpyDeviceToHost));
            cudaFree(drow); cudaFree(dq2);
            float md = 0; int mdi = -1;
            for (int i = 0; i < H; ++i) {
                float d = std::fabs(cref[i] - grow[i]);
                if (d > md) { md = d; mdi = i; }
            }
            std::printf("  dequant_row_q [H=%d Q8_0]: max abs diff = %.3e at %d\n", H, md, mdi);
            if (md > 1e-6f) rc = 1;
        }
        std::vector<int> toks = {3, 0, 31, 3};
        std::vector<float> ref((size_t)toks.size() * H);
        for (size_t t = 0; t < toks.size(); ++t) {
            auto row = glmserve::gguf_dequantize_row(type, table.data(), H, (uint64_t)toks[t]);
            std::copy(row.begin(), row.end(), ref.begin() + t * H);
        }
        uint8_t* dtab = nullptr; int* dtok = nullptr; float* dh = nullptr;
        CUDA_TEST_CHECK(cudaMalloc(&dtab, table.size()));
        CUDA_TEST_CHECK(cudaMalloc(&dtok, toks.size() * sizeof(int)));
        CUDA_TEST_CHECK(cudaMalloc(&dh, ref.size() * sizeof(float)));
        CUDA_TEST_CHECK(cudaMemcpy(dtab, table.data(), table.size(), cudaMemcpyHostToDevice));
        CUDA_TEST_CHECK(cudaMemcpy(dtok, toks.data(), toks.size() * sizeof(int), cudaMemcpyHostToDevice));
        embed_gather_q(type, dtab, dtok, dh, (int64_t)toks.size(), H, rb);
        CUDA_TEST_CHECK(cudaGetLastError());
        CUDA_TEST_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(ref.size());
        CUDA_TEST_CHECK(cudaMemcpy(got.data(), dh, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dtab); cudaFree(dtok); cudaFree(dh);
        std::printf("  embed_gather_q [vocab=%d H=%d Q8_0]:", vocab, H);
        rc |= check(got, ref, 1e-6f);
    }

    std::printf("test_qgemm: %s\n", rc ? "FAIL" : "PASS");
    return rc;
#else
    (void)types; (void)names; (void)check;
    std::printf("test_qgemm: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
