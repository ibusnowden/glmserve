// glmserve - GPU forward path (device-resident weights + CUDA kernels).
//
// Mirrors the CPU reference in src/model_glm52.cpp exactly, executing the full
// GLM-5.2 MLA block on the device: embed gather -> RMSNorm -> MLA Q/KV latent
// projections (cuBLAS) -> decoupled RoPE -> per-head K/V assembly -> dense paged
// attention -> o_proj -> dense MLP or MoE (router + grouped expert kernels +
// shared expert) -> residuals -> final norm -> lm_head.
//
// Both prefill and decode share one block-stack runner and a single persistent
// device state:
//   * forward_gpu_prefill() resets the cache, fills positions [0, n) in one shot.
//   * forward_gpu_decode()  appends one token's K/V and attends over [0, pos],
//     reusing persistent scratch (no per-token cudaMalloc) and the persistent
//     per-layer KV cache (O(ctx) per token instead of re-prefilling, O(ctx^2)).
// Only compiled in a GLMSERVE_CUDA build.
#include "model.hpp"
#include "common.hpp"

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace glmserve {

namespace {
inline float* dmalloc(size_t n_floats) {
    float* p = nullptr;
    if (cudaMalloc(&p, n_floats * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    return p;
}
inline float* upload(const std::vector<float>& v) {
    float* p = dmalloc(v.size());
    cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return p;
}
inline uint8_t* upload_u8(const std::vector<uint8_t>& v) {
    uint8_t* p = nullptr;
    if (cudaMalloc(&p, v.size() * sizeof(uint8_t)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    cudaMemcpy(p, v.data(), v.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    return p;
}
// Surface any pending CUDA error (async kernel launch failures included) so a
// bad GPU forward fails loudly instead of returning garbage logits to the gate.
inline void cuda_check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error after ") + what + ": " +
                                 cudaGetErrorString(e));
}
struct DLinear {
    float* w = nullptr;
    uint8_t* qweight = nullptr;
    float* scales = nullptr;
    int64_t out = 0, in = 0, group_size = 0;
    bool quantized_int4 = false;
};
inline DLinear up_lin(const Linear& L) {
    DLinear d; d.out = L.out_features; d.in = L.in_features;
    d.group_size = L.group_size;
    d.quantized_int4 = L.quantized_int4;
    if (!L.w.empty()) d.w = upload(L.w);
    if (!L.qweight.empty()) d.qweight = upload_u8(L.qweight);
    if (!L.scales.empty()) d.scales = upload(L.scales);
    return d;
}
inline void free_lin(DLinear& d) {
    if (d.w) cudaFree(d.w);
    if (d.qweight) cudaFree(d.qweight);
    if (d.scales) cudaFree(d.scales);
    d = DLinear{};
}
inline void gemm_linear(const float* x, const DLinear& d, const float* bias,
                        float* y, int64_t n, cudaStream_t s = 0) {
    if (d.quantized_int4) {
        GLM_CHECK(d.qweight && d.scales && d.group_size > 0,
                  "quantized linear missing resident qweight/scales");
        glmserve::cuda::gemm_w4a16(x, d.qweight, d.scales, bias, y,
                                   n, d.in, d.out, d.group_size, s);
    } else {
        GLM_CHECK(d.w, "fp32 linear missing resident weight");
        glmserve::cuda::gemm_fp32(x, d.w, bias, y, n, d.in, d.out, s);
    }
}
}  // namespace

struct DLayer {
    float *input_norm = nullptr, *post_norm = nullptr;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    DLinear q_a, q_b, kv_a, kv_b, o;
    DLinear index_wq_b, index_wk, index_weights;
    float* index_k_norm = nullptr;
    float* index_k_bias = nullptr;
    bool is_dense = false;
    DLinear gate, up, down;                 // dense MLP
    DLinear router;                         // MoE
    float* e_bias = nullptr;
    int64_t expert_inter = 0;
    float *exp_gate = nullptr, *exp_up = nullptr, *exp_down = nullptr;  // packed [E,...]
    uint8_t *qexp_gate = nullptr, *qexp_up = nullptr, *qexp_down = nullptr;
    float *sexp_gate = nullptr, *sexp_up = nullptr, *sexp_down = nullptr;
    int64_t exp_group_size = 0;
    bool quantized_experts = false;
    DLinear sh_gate, sh_up, sh_down;        // shared expert
};

// Per-forward activation buffers, allocated once and grown on demand. Decode
// (n=1) reuses the same arena every token, so there is no per-token cudaMalloc.
struct Scratch {
    int64_t cap = 0;     // max n this arena is sized for
    int*   d_tokens = nullptr;
    float *hidden_d = nullptr, *normed = nullptr, *qa = nullptr, *q = nullptr;
    float *kva = nullptr, *ckv = nullptr, *kpe = nullptr, *kvb = nullptr;
    float *attn = nullptr, *attn_proj = nullptr, *mlp = nullptr;
    float *gbuf = nullptr, *ubuf = nullptr, *gu = nullptr, *rlogits = nullptr, *shbuf = nullptr;
    float *index_q = nullptr, *index_k = nullptr, *index_w = nullptr, *dsa_scores = nullptr;
    int*   topk_ids = nullptr;
    int*   dsa_indices = nullptr;
    float* topk_w = nullptr;

    void free_all() {
        for (float* p : {hidden_d, normed, qa, q, kva, ckv, kpe, kvb, attn,
                         attn_proj, mlp, gbuf, ubuf, gu, rlogits, shbuf,
                         index_q, index_k, index_w, dsa_scores, topk_w})
            if (p) cudaFree(p);
        if (d_tokens) cudaFree(d_tokens);
        if (topk_ids) cudaFree(topk_ids);
        if (dsa_indices) cudaFree(dsa_indices);
        *this = Scratch{};
    }
    void ensure(int64_t n, const GLM52Config& c, int64_t local_heads,
                int64_t max_dense_inter, int64_t max_moe_inter) {
        if (n <= cap) return;
        free_all();
        const int64_t hidden = c.hidden_size, nheads = local_heads;
        const int64_t qlat = c.q_lora_rank, kvlat = c.kv_lora_rank;
        const int64_t rope = c.qk_rope_head_dim, nope = c.qk_nope_head_dim;
        const int64_t qk = c.qk_head_dim, vd = c.v_head_dim;
        const int64_t E = c.n_routed_experts, topk = c.num_experts_per_tok;
        const int64_t iH = c.index_n_heads, iD = c.index_head_dim;
        const int64_t ffmax = std::max<int64_t>(1, std::max(max_dense_inter, max_moe_inter));
        if (cudaMalloc(&d_tokens, n * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        hidden_d  = dmalloc(n * hidden);
        normed    = dmalloc(n * hidden);
        qa        = dmalloc(n * qlat);
        q         = dmalloc(n * nheads * qk);
        kva       = dmalloc(n * (kvlat + rope));
        ckv       = dmalloc(n * kvlat);
        kpe       = dmalloc(n * rope);
        kvb       = dmalloc(n * nheads * (nope + vd));
        attn      = dmalloc(n * nheads * vd);
        attn_proj = dmalloc(n * hidden);
        mlp       = dmalloc(n * hidden);
        gbuf      = dmalloc(n * ffmax);
        ubuf      = dmalloc(n * ffmax);
        gu        = dmalloc(n * ffmax);
        rlogits   = dmalloc(n * E);
        shbuf     = dmalloc(n * hidden);
        index_q   = dmalloc(n * iH * iD);
        index_k   = dmalloc(n * iD);
        index_w   = dmalloc(n * iH);
        dsa_scores = dmalloc(n * c.index_topk);
        if (cudaMalloc(&topk_ids, n * topk * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        if (cudaMalloc(&dsa_indices, n * c.index_topk * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        topk_w    = dmalloc(n * topk);
        cap = n;
    }
};

struct GpuState {
    float* embed = nullptr;          // [vocab, H]
    float* final_norm = nullptr;
    DLinear lm_head;
    std::vector<DLayer> layers;
    int64_t local_heads = 0;
    int64_t max_dense_inter = 0;
    int64_t max_moe_inter = 0;

    // Persistent KV cache: one contiguous block per layer of capacity max_ctx,
    // laid out [max_ctx, local_heads, hc]. A single-entry block_table (=0) maps the
    // sequence's one logical block to it, so the paged-attention kernel runs
    // unchanged with block_size = max_ctx.
    std::vector<float*> Kc, Vc, Ic;
    int64_t max_ctx = 0;
    int*   block_table = nullptr;    // single entry, value 0
    float* logits_d = nullptr;       // [vocab]
    Scratch sc;
    int64_t cur_len = 0;             // positions currently valid in the cache

    // Flash-decoding partials: per (head, split) accumulator + softmax stats.
    float *part_acc = nullptr, *part_m = nullptr, *part_l = nullptr;
};

// Key-range splits for flash-decoding (decode-step attention parallelism).
static constexpr int64_t kMaxSplits = 128;

static bool is_shared_dsa_layer(const GLM52Config& c, int64_t global_layer) {
    return c.use_dsa &&
           global_layer >= 0 &&
           global_layer < static_cast<int64_t>(c.indexer_types.size()) &&
           c.indexer_types[static_cast<size_t>(global_layer)] != "full";
}

bool GLM52Model::upload_to_gpu(int64_t max_ctx) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        GLM_WARN("no CUDA device available; staying on CPU path");
        return false;
    }
    Timer t;
    auto* g = new GpuState();
    g->local_heads = tp_heads_;
    GLM_CHECK(g->local_heads > 0, "GPU upload requires at least one local attention head");

    if (dist_.is_first_stage()) {
        GLM_CHECK(!embed_tokens_.empty(), "first GPU pipeline stage is missing embeddings");
        g->embed = upload(embed_tokens_);
    }
    if (dist_.is_last_stage()) {
        GLM_CHECK(!final_norm_.w.empty(), "last GPU pipeline stage is missing final norm");
        GLM_CHECK(lm_head_.valid(), "last GPU pipeline stage is missing lm_head");
        g->final_norm = upload(final_norm_.w);
        g->lm_head = up_lin(lm_head_);
    }

    const auto& c = cfg_;
    const int64_t hidden = c.hidden_size;
    g->layers.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        const Layer& L = layers_[i];
        DLayer& d = g->layers[i];
        d.input_norm = upload(L.input_norm.w);
        d.post_norm  = upload(L.post_attn_norm.w);
        d.q_a_norm   = upload(L.q_a_norm.w);
        d.kv_a_norm  = upload(L.kv_a_norm.w);
        d.q_a = up_lin(L.q_a_proj); d.q_b = up_lin(L.q_b_proj);
        d.kv_a = up_lin(L.kv_a_proj); d.kv_b = up_lin(L.kv_b_proj);
        d.o = up_lin(L.o_proj);
        if (L.indexer.valid()) {
            d.index_wq_b = up_lin(L.indexer.wq_b);
            d.index_wk = up_lin(L.indexer.wk);
            d.index_weights = up_lin(L.indexer.weights_proj);
            d.index_k_norm = upload(L.indexer.k_norm.w);
            if (!L.indexer.k_norm_bias.empty()) d.index_k_bias = upload(L.indexer.k_norm_bias);
        }
        d.is_dense = L.is_dense;
        if (L.is_dense) {
            d.gate = up_lin(L.dense_mlp.gate_proj);
            d.up   = up_lin(L.dense_mlp.up_proj);
            d.down = up_lin(L.dense_mlp.down_proj);
            g->max_dense_inter = std::max(g->max_dense_inter, d.gate.out);
        } else {
            d.router = up_lin(L.moe.router);
            if (!L.moe.e_bias.empty()) d.e_bias = upload(L.moe.e_bias);
            // pack experts contiguously: [E, moei, hidden] (gate/up), [E, hidden, moei] (down)
            int64_t E = static_cast<int64_t>(L.moe.experts.size());
            d.expert_inter = E > 0 ? L.moe.experts[0].gate_proj.out_features : 0;
            GLM_CHECK(d.expert_inter > 0, "MoE layer has no expert intermediate dimension");
            bool any_quantized_expert = false;
            bool all_quantized_experts = true;
            for (const Expert& e : L.moe.experts) {
                const bool qe = e.gate_proj.quantized_int4 ||
                                e.up_proj.quantized_int4 ||
                                e.down_proj.quantized_int4;
                const bool ae = e.gate_proj.quantized_int4 &&
                                e.up_proj.quantized_int4 &&
                                e.down_proj.quantized_int4;
                any_quantized_expert = any_quantized_expert || qe;
                all_quantized_experts = all_quantized_experts && ae;
            }
            GLM_CHECK(!any_quantized_expert || all_quantized_experts,
                      "mixed quantized/unquantized routed expert weights are not supported");
            d.quantized_experts = all_quantized_experts;
            std::vector<float> pg, pu, pd, sg, su, sd;
            std::vector<uint8_t> qg, qu, qd;
            if (d.quantized_experts) {
                d.exp_group_size = L.moe.experts[0].gate_proj.group_size;
                const int64_t gate_packed = (hidden + 1) / 2;
                const int64_t gate_groups = (hidden + d.exp_group_size - 1) / d.exp_group_size;
                const int64_t down_packed = (d.expert_inter + 1) / 2;
                const int64_t down_groups = (d.expert_inter + d.exp_group_size - 1) / d.exp_group_size;
                qg.reserve(E * d.expert_inter * gate_packed);
                qu.reserve(E * d.expert_inter * gate_packed);
                qd.reserve(E * hidden * down_packed);
                sg.reserve(E * d.expert_inter * gate_groups);
                su.reserve(E * d.expert_inter * gate_groups);
                sd.reserve(E * hidden * down_groups);
            } else {
                pg.reserve(E * d.expert_inter * hidden);
                pu.reserve(E * d.expert_inter * hidden);
                pd.reserve(E * hidden * d.expert_inter);
            }
            for (const Expert& e : L.moe.experts) {
                GLM_CHECK(e.gate_proj.out_features == d.expert_inter &&
                          e.up_proj.out_features == d.expert_inter &&
                          e.down_proj.in_features == d.expert_inter,
                          "inconsistent sharded expert dimensions");
                if (d.quantized_experts) {
                    GLM_CHECK(e.gate_proj.quantized_int4 && e.up_proj.quantized_int4 &&
                              e.down_proj.quantized_int4 &&
                              e.gate_proj.group_size == d.exp_group_size &&
                              e.up_proj.group_size == d.exp_group_size &&
                              e.down_proj.group_size == d.exp_group_size,
                              "mixed or inconsistent quantized expert weights");
                    qg.insert(qg.end(), e.gate_proj.qweight.begin(), e.gate_proj.qweight.end());
                    qu.insert(qu.end(), e.up_proj.qweight.begin(), e.up_proj.qweight.end());
                    qd.insert(qd.end(), e.down_proj.qweight.begin(), e.down_proj.qweight.end());
                    sg.insert(sg.end(), e.gate_proj.scales.begin(), e.gate_proj.scales.end());
                    su.insert(su.end(), e.up_proj.scales.begin(), e.up_proj.scales.end());
                    sd.insert(sd.end(), e.down_proj.scales.begin(), e.down_proj.scales.end());
                } else {
                    GLM_CHECK(e.gate_proj.has_f32() && e.up_proj.has_f32() && e.down_proj.has_f32(),
                              "routed expert missing fp32 resident weights");
                    pg.insert(pg.end(), e.gate_proj.w.begin(), e.gate_proj.w.end());
                    pu.insert(pu.end(), e.up_proj.w.begin(), e.up_proj.w.end());
                    pd.insert(pd.end(), e.down_proj.w.begin(), e.down_proj.w.end());
                }
            }
            if (d.quantized_experts) {
                d.qexp_gate = upload_u8(qg); d.qexp_up = upload_u8(qu); d.qexp_down = upload_u8(qd);
                d.sexp_gate = upload(sg); d.sexp_up = upload(su); d.sexp_down = upload(sd);
            } else {
                d.exp_gate = upload(pg); d.exp_up = upload(pu); d.exp_down = upload(pd);
            }
            g->max_moe_inter = std::max(g->max_moe_inter, d.expert_inter);
            if (L.moe.has_shared) {
                d.sh_gate = up_lin(L.moe.shared.gate_proj);
                d.sh_up   = up_lin(L.moe.shared.up_proj);
                d.sh_down = up_lin(L.moe.shared.down_proj);
                g->max_moe_inter = std::max(g->max_moe_inter, d.sh_gate.out);
            }
        }
    }

    // Persistent KV cache + small persistent buffers.
    const int64_t nheads = g->local_heads, hc = c.kv_cache_head_dim();
    g->max_ctx = max_ctx > 0 ? max_ctx : 4096;
    g->Kc.resize(layers_.size());
    g->Vc.resize(layers_.size());
    g->Ic.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        g->Kc[i] = dmalloc(g->max_ctx * nheads * hc);
        g->Vc[i] = dmalloc(g->max_ctx * nheads * hc);
        g->Ic[i] = dmalloc(g->max_ctx * c.index_head_dim);
    }
    cudaMalloc(&g->block_table, sizeof(int));
    int zero = 0;
    cudaMemcpy(g->block_table, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (dist_.is_last_stage())
        g->logits_d = dmalloc(c.vocab_size);

    // Flash-decoding partials: [n_heads, kMaxSplits, hc] + [n_heads, kMaxSplits].
    g->part_acc = dmalloc(nheads * kMaxSplits * hc);
    g->part_m   = dmalloc(nheads * kMaxSplits);
    g->part_l   = dmalloc(nheads * kMaxSplits);

    gpu_state_ = g;
    size_t freeb = 0, total = 0; cudaMemGetInfo(&freeb, &total);
    GLM_INFO("uploaded weights to GPU in %.1f s; KV cache ctx=%lld (%.2f GiB free / %.2f GiB total)",
             t.ms() / 1000.0, (long long)g->max_ctx, freeb / 1073741824.0, total / 1073741824.0);
    return true;
}

int64_t GLM52Model::gpu_kv_ctx() const {
    return gpu_state_ ? static_cast<GpuState*>(gpu_state_)->max_ctx : 0;
}

namespace {
// Run this stage's block stack on `n` tokens whose activations are already in
// g->sc.hidden_d. Appends each local layer's K/V to the persistent cache at
// absolute offset `start_pos` and attends over [0, start_pos+n). Leaves the
// final pre-norm hidden state in g->sc.hidden_d.
bool run_block_stack(GpuState* g, const GLM52Config& c, Communicator* comm,
                     const DistConfig& dist, int64_t n, int64_t start_pos,
                     bool have_shared_dsa_indices) {
    using namespace glmserve::cuda;
    Scratch& s = g->sc;
    const int64_t hidden = c.hidden_size, nheads = g->local_heads;
    const int64_t qlat = c.q_lora_rank, kvlat = c.kv_lora_rank;
    const int64_t nope = c.qk_nope_head_dim, rope = c.qk_rope_head_dim;
    const int64_t qk = c.qk_head_dim, vd = c.v_head_dim, hc = c.kv_cache_head_dim();
    const int64_t E = c.n_routed_experts, topk = c.num_experts_per_tok;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const bool il = c.rope_interleave;
    const bool tp = dist.tp_size > 1;
    for (size_t l = 0; l < g->layers.size(); ++l) {
        const DLayer& d = g->layers[l];
        // attention sub-block
        rmsnorm(s.hidden_d, d.input_norm, s.normed, n, hidden, eps);
        gemm_linear(s.normed, d.q_a, nullptr, s.qa, n);
        rmsnorm(s.qa, d.q_a_norm, s.qa, n, qlat, eps);
        gemm_linear(s.qa, d.q_b, nullptr, s.q, n);
        gemm_linear(s.normed, d.kv_a, nullptr, s.kva, n);
        slice_rows(s.kva, s.ckv, n, kvlat + rope, 0, kvlat);
        rmsnorm(s.ckv, d.kv_a_norm, s.ckv, n, kvlat, eps);
        gemm_linear(s.ckv, d.kv_b, nullptr, s.kvb, n);
        slice_rows(s.kva, s.kpe, n, kvlat + rope, kvlat, rope);
        rope_k(s.kpe, n, rope, start_pos, c.rope_theta, il);
        rope_q(s.q, n, nheads, qk, nope, rope, start_pos, c.rope_theta, il);

        float* Kbase = g->Kc[l] + start_pos * nheads * hc;
        float* Vbase = g->Vc[l] + start_pos * nheads * hc;
        assemble_kv(s.kvb, s.kpe, Kbase, Vbase, n, nheads, nope, rope, vd, hc);

        const int64_t ctx_after = start_pos + n;
        const bool sparse_dsa = c.use_dsa && ctx_after > c.index_topk;
        const bool learned_dsa = sparse_dsa && d.index_wq_b.out > 0;
        const bool shared_dsa = sparse_dsa && !learned_dsa && have_shared_dsa_indices;
        if (learned_dsa) {
            gemm_linear(s.qa, d.index_wq_b, nullptr, s.index_q, n);
            gemm_linear(s.normed, d.index_wk, nullptr, s.index_k, n);
            layernorm(s.index_k, d.index_k_norm, d.index_k_bias, s.index_k, n,
                      c.index_head_dim, 1e-6f);
            rope_index_q(s.index_q, n, c.index_n_heads, c.index_head_dim, rope,
                         start_pos, c.rope_theta, false);
            rope_index_q(s.index_k, n, 1, c.index_head_dim, rope,
                         start_pos, c.rope_theta, false);
            cudaMemcpy(g->Ic[l] + start_pos * c.index_head_dim, s.index_k,
                       n * c.index_head_dim * sizeof(float), cudaMemcpyDeviceToDevice);
            gemm_linear(s.normed, d.index_weights, nullptr, s.index_w, n);
            dsa_select_topk(s.index_q, g->Ic[l], s.index_w, n, start_pos,
                            c.index_n_heads, c.index_head_dim, c.index_topk,
                            1.0f / std::sqrt(static_cast<float>(c.index_head_dim)),
                            1.0f / std::sqrt(static_cast<float>(c.index_n_heads)),
                            s.dsa_indices, s.dsa_scores);
            have_shared_dsa_indices = true;
            attention_dsa_indexed_paged(s.q, g->Kc[l], g->Vc[l], g->block_table,
                                        s.dsa_indices, n, start_pos, nheads, nheads, qk,
                                        g->max_ctx, c.index_topk, scale, s.attn);
        } else if (shared_dsa) {
            attention_dsa_indexed_paged(s.q, g->Kc[l], g->Vc[l], g->block_table,
                                        s.dsa_indices, n, start_pos, nheads, nheads, qk,
                                        g->max_ctx, c.index_topk, scale, s.attn);
        } else if (sparse_dsa) {
            attention_dsa_paged(s.q, g->Kc[l], g->Vc[l], g->block_table, n, start_pos,
                                nheads, nheads, qk, g->max_ctx, c.index_topk, scale, s.attn, 0,
                                g->part_acc, g->part_m, g->part_l, kMaxSplits);
        } else if (n == 1) {
            attention_decode_paged(s.q, g->Kc[l], g->Vc[l], g->block_table, start_pos,
                                   nheads, nheads, qk, g->max_ctx, scale, s.attn,
                                   g->part_acc, g->part_m, g->part_l, kMaxSplits);
        } else {
            attention_dense_paged(s.q, g->Kc[l], g->Vc[l], g->block_table, n, start_pos,
                                  nheads, nheads, qk, g->max_ctx, scale, s.attn);
        }
        gemm_linear(s.attn, d.o, nullptr, s.attn_proj, n);
        if (tp) {
            GLM_CHECK(comm, "tensor-parallel GPU forward requires a communicator");
            comm->all_reduce_sum(s.attn_proj, n * hidden);
        }
        add_inplace(s.hidden_d, s.attn_proj, n * hidden);

        // MLP sub-block
        rmsnorm(s.hidden_d, d.post_norm, s.normed, n, hidden, eps);
        if (d.is_dense) {
            const int64_t inter = d.gate.out;
            gemm_linear(s.normed, d.gate, nullptr, s.gbuf, n);
            gemm_linear(s.normed, d.up,   nullptr, s.ubuf, n);
            silu_mul(s.gbuf, s.ubuf, s.gu, n * inter);
            gemm_linear(s.gu, d.down, nullptr, s.mlp, n);
        } else {
            const int64_t moei = d.expert_inter;
            gemm_linear(s.normed, d.router, nullptr, s.rlogits, n);
            moe_router(s.rlogits, d.e_bias, n, E, topk, c.norm_topk_prob,
                       static_cast<float>(c.routed_scaling_factor), s.topk_ids, s.topk_w);
            cudaMemset(s.mlp, 0, n * hidden * sizeof(float));
            if (d.quantized_experts) {
                moe_expert_ffn_w4a16(s.normed, s.topk_ids, s.topk_w,
                                     d.qexp_gate, d.sexp_gate, d.qexp_up, d.sexp_up,
                                     d.qexp_down, d.sexp_down, n, topk, hidden, moei, E,
                                     d.exp_group_size, s.mlp);
            } else {
                moe_expert_ffn(s.normed, s.topk_ids, s.topk_w, d.exp_gate, d.exp_up, d.exp_down,
                               n, topk, hidden, moei, E, s.mlp);
            }
            if (d.sh_gate.out > 0) {
                const int64_t sh_inter = d.sh_gate.out;
                gemm_linear(s.normed, d.sh_gate, nullptr, s.gbuf, n);
                gemm_linear(s.normed, d.sh_up,   nullptr, s.ubuf, n);
                silu_mul(s.gbuf, s.ubuf, s.gu, n * sh_inter);
                gemm_linear(s.gu, d.sh_down, nullptr, s.shbuf, n);
                add_inplace(s.mlp, s.shbuf, n * hidden);
            }
        }
        if (tp) comm->all_reduce_sum(s.mlp, n * hidden);
        add_inplace(s.hidden_d, s.mlp, n * hidden);
    }
    return have_shared_dsa_indices;
}

// Final RMSNorm + lm_head over the LAST row, copy logits to host.
std::vector<float> finish_logits(GpuState* g, const GLM52Config& c, int64_t n) {
    using namespace glmserve::cuda;
    GLM_CHECK(g->final_norm && g->logits_d && g->lm_head.out > 0,
              "finish_logits called on a non-final or incompletely uploaded GPU stage");
    rmsnorm(g->sc.hidden_d, g->final_norm, g->sc.normed, n, c.hidden_size,
            static_cast<float>(c.rms_norm_eps));
    gemm_linear(g->sc.normed + (n - 1) * c.hidden_size, g->lm_head, nullptr,
                g->logits_d, 1);
    std::vector<float> logits(c.vocab_size);
    cudaMemcpy(logits.data(), g->logits_d, c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return logits;
}
}  // namespace

std::vector<float> GLM52Model::forward_gpu_prefill(const std::vector<int>& tokens) {
    GLM_CHECK(gpu_state_ != nullptr, "forward_gpu_prefill: call upload_to_gpu() first");
    using namespace glmserve::cuda;
    auto* g = static_cast<GpuState*>(gpu_state_);
    const int64_t n = static_cast<int64_t>(tokens.size());
    GLM_CHECK(n > 0, "forward_gpu_prefill: empty prompt");
    GLM_CHECK(n <= g->max_ctx, "forward_gpu_prefill: prompt %lld > GPU KV ctx %lld",
              (long long)n, (long long)g->max_ctx);

    g->sc.ensure(n, cfg_, g->local_heads, g->max_dense_inter, g->max_moe_inter);
    if (dist_.is_first_stage()) {
        GLM_CHECK(g->embed, "forward_gpu_prefill: first GPU pipeline stage is missing embeddings");
        std::vector<int> toks(tokens.begin(), tokens.end());
        cudaMemcpy(g->sc.d_tokens, toks.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, n, cfg_.hidden_size);
    } else {
        GLM_CHECK(comm_, "forward_gpu_prefill: non-first pipeline stage requires a communicator");
        comm_->pipeline_recv_prev(g->sc.hidden_d, n * cfg_.hidden_size);
    }

    const int64_t ctx_after = n;
    const bool sparse_dsa = cfg_.use_dsa && ctx_after > cfg_.index_topk;
    const bool recv_shared_dsa = sparse_dsa && !dist_.is_first_stage() &&
                                 is_shared_dsa_layer(cfg_, layer_begin_);
    if (recv_shared_dsa) {
        GLM_CHECK(comm_, "forward_gpu_prefill: DSA mask receive requires a communicator");
        comm_->pipeline_recv_prev_int(g->sc.dsa_indices, n * cfg_.index_topk);
    }

    bool have_shared_dsa = run_block_stack(g, cfg_, comm_, dist_, n, /*start_pos=*/0,
                                           recv_shared_dsa);
    g->cur_len = n;
    if (!dist_.is_last_stage()) {
        GLM_CHECK(comm_, "forward_gpu_prefill: non-last pipeline stage requires a communicator");
        comm_->pipeline_send_next(g->sc.hidden_d, n * cfg_.hidden_size);
        int64_t next_layer = layer_begin_ + num_layers();
        if (sparse_dsa && have_shared_dsa && is_shared_dsa_layer(cfg_, next_layer))
            comm_->pipeline_send_next_int(g->sc.dsa_indices, n * cfg_.index_topk);
        cuda_check("forward_gpu_prefill");
        return {};
    }
    std::vector<float> logits = finish_logits(g, cfg_, n);
    cuda_check("forward_gpu_prefill");
    return logits;
}

std::vector<float> GLM52Model::forward_gpu_decode(int token, int64_t pos) {
    GLM_CHECK(gpu_state_ != nullptr, "forward_gpu_decode: call upload_to_gpu() first");
    using namespace glmserve::cuda;
    auto* g = static_cast<GpuState*>(gpu_state_);
    GLM_CHECK(pos < g->max_ctx, "forward_gpu_decode: position %lld >= GPU KV ctx %lld "
              "(raise --ctx)", (long long)pos, (long long)g->max_ctx);

    g->sc.ensure(1, cfg_, g->local_heads, g->max_dense_inter, g->max_moe_inter);
    if (dist_.is_first_stage()) {
        GLM_CHECK(g->embed, "forward_gpu_decode: first GPU pipeline stage is missing embeddings");
        cudaMemcpy(g->sc.d_tokens, &token, sizeof(int), cudaMemcpyHostToDevice);
        embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, 1, cfg_.hidden_size);
    } else {
        GLM_CHECK(comm_, "forward_gpu_decode: non-first pipeline stage requires a communicator");
        comm_->pipeline_recv_prev(g->sc.hidden_d, cfg_.hidden_size);
    }

    const int64_t ctx_after = pos + 1;
    const bool sparse_dsa = cfg_.use_dsa && ctx_after > cfg_.index_topk;
    const bool recv_shared_dsa = sparse_dsa && !dist_.is_first_stage() &&
                                 is_shared_dsa_layer(cfg_, layer_begin_);
    if (recv_shared_dsa) {
        GLM_CHECK(comm_, "forward_gpu_decode: DSA mask receive requires a communicator");
        comm_->pipeline_recv_prev_int(g->sc.dsa_indices, cfg_.index_topk);
    }

    bool have_shared_dsa = run_block_stack(g, cfg_, comm_, dist_, /*n=*/1, /*start_pos=*/pos,
                                           recv_shared_dsa);
    g->cur_len = pos + 1;
    if (!dist_.is_last_stage()) {
        GLM_CHECK(comm_, "forward_gpu_decode: non-last pipeline stage requires a communicator");
        comm_->pipeline_send_next(g->sc.hidden_d, cfg_.hidden_size);
        int64_t next_layer = layer_begin_ + num_layers();
        if (sparse_dsa && have_shared_dsa && is_shared_dsa_layer(cfg_, next_layer))
            comm_->pipeline_send_next_int(g->sc.dsa_indices, cfg_.index_topk);
        cuda_check("forward_gpu_decode");
        return {};
    }
    std::vector<float> logits = finish_logits(g, cfg_, 1);
    cuda_check("forward_gpu_decode");
    return logits;
}

GLM52Model::~GLM52Model() {
    if (!gpu_state_) return;
    auto* g = static_cast<GpuState*>(gpu_state_);
    cudaFree(g->embed); cudaFree(g->final_norm); free_lin(g->lm_head);
    for (auto& d : g->layers) {
        for (float* p : {d.input_norm, d.post_norm, d.q_a_norm, d.kv_a_norm,
                         d.index_k_norm, d.index_k_bias,
                         d.exp_gate, d.exp_up, d.exp_down,
                         d.sexp_gate, d.sexp_up, d.sexp_down, d.e_bias})
            if (p) cudaFree(p);
        for (uint8_t* p : {d.qexp_gate, d.qexp_up, d.qexp_down})
            if (p) cudaFree(p);
        for (DLinear* lin : {&d.q_a, &d.q_b, &d.kv_a, &d.kv_b, &d.o,
                             &d.index_wq_b, &d.index_wk, &d.index_weights,
                             &d.gate, &d.up, &d.down, &d.router,
                             &d.sh_gate, &d.sh_up, &d.sh_down})
            free_lin(*lin);
    }
    for (float* p : g->Kc) if (p) cudaFree(p);
    for (float* p : g->Vc) if (p) cudaFree(p);
    for (float* p : g->Ic) if (p) cudaFree(p);
    if (g->block_table) cudaFree(g->block_table);
    if (g->logits_d) cudaFree(g->logits_d);
    if (g->part_acc) cudaFree(g->part_acc);
    if (g->part_m) cudaFree(g->part_m);
    if (g->part_l) cudaFree(g->part_l);
    g->sc.free_all();
    delete g;
    gpu_state_ = nullptr;
}

}  // namespace glmserve
#endif  // GLMSERVE_CUDA
