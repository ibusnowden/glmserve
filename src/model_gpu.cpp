// glmserve — GPU forward path (device-resident weights + CUDA kernels).
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
// Surface any pending CUDA error (async kernel launch failures included) so a
// bad GPU forward fails loudly instead of returning garbage logits to the gate.
inline void cuda_check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error after ") + what + ": " +
                                 cudaGetErrorString(e));
}
struct DLinear { float* w = nullptr; int64_t out = 0, in = 0; };
inline DLinear up_lin(const Linear& L) {
    DLinear d; d.out = L.out_features; d.in = L.in_features;
    if (!L.w.empty()) d.w = upload(L.w);
    return d;
}
}  // namespace

struct DLayer {
    float *input_norm = nullptr, *post_norm = nullptr;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    DLinear q_a, q_b, kv_a, kv_b, o;
    bool is_dense = false;
    DLinear gate, up, down;                 // dense MLP
    DLinear router;                         // MoE
    float* e_bias = nullptr;
    float *exp_gate = nullptr, *exp_up = nullptr, *exp_down = nullptr;  // packed [E,...]
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
    int*   topk_ids = nullptr;
    float* topk_w = nullptr;

    void free_all() {
        for (float* p : {hidden_d, normed, qa, q, kva, ckv, kpe, kvb, attn,
                         attn_proj, mlp, gbuf, ubuf, gu, rlogits, shbuf, topk_w})
            if (p) cudaFree(p);
        if (d_tokens) cudaFree(d_tokens);
        if (topk_ids) cudaFree(topk_ids);
        *this = Scratch{};
    }
    void ensure(int64_t n, const GLM52Config& c) {
        if (n <= cap) return;
        free_all();
        const int64_t hidden = c.hidden_size, nheads = c.num_attention_heads;
        const int64_t qlat = c.q_lora_rank, kvlat = c.kv_lora_rank;
        const int64_t rope = c.qk_rope_head_dim, nope = c.qk_nope_head_dim;
        const int64_t qk = c.qk_head_dim, vd = c.v_head_dim;
        const int64_t inter = c.intermediate_size, moei = c.moe_intermediate_size;
        const int64_t E = c.n_routed_experts, topk = c.num_experts_per_tok;
        const int64_t ffmax = inter > moei ? inter : moei;
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
        if (cudaMalloc(&topk_ids, n * topk * sizeof(int)) != cudaSuccess)
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

    // Persistent KV cache: one contiguous block per layer of capacity max_ctx,
    // laid out [max_ctx, n_heads, hc]. A single-entry block_table (=0) maps the
    // sequence's one logical block to it, so the paged-attention kernel runs
    // unchanged with block_size = max_ctx.
    std::vector<float*> Kc, Vc;
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

bool GLM52Model::upload_to_gpu(int64_t max_ctx) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        GLM_WARN("no CUDA device available; staying on CPU path");
        return false;
    }
    Timer t;
    auto* g = new GpuState();
    g->embed = upload(embed_tokens_);
    g->final_norm = upload(final_norm_.w);
    g->lm_head = up_lin(lm_head_);

    const auto& c = cfg_;
    const int64_t hidden = c.hidden_size;
    const int64_t moei = c.moe_intermediate_size;
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
        d.is_dense = L.is_dense;
        if (L.is_dense) {
            d.gate = up_lin(L.dense_mlp.gate_proj);
            d.up   = up_lin(L.dense_mlp.up_proj);
            d.down = up_lin(L.dense_mlp.down_proj);
        } else {
            d.router = up_lin(L.moe.router);
            if (!L.moe.e_bias.empty()) d.e_bias = upload(L.moe.e_bias);
            // pack experts contiguously: [E, moei, hidden] (gate/up), [E, hidden, moei] (down)
            int64_t E = static_cast<int64_t>(L.moe.experts.size());
            std::vector<float> pg, pu, pd;
            pg.reserve(E * moei * hidden); pu.reserve(E * moei * hidden);
            pd.reserve(E * hidden * moei);
            for (const Expert& e : L.moe.experts) {
                pg.insert(pg.end(), e.gate_proj.w.begin(), e.gate_proj.w.end());
                pu.insert(pu.end(), e.up_proj.w.begin(), e.up_proj.w.end());
                pd.insert(pd.end(), e.down_proj.w.begin(), e.down_proj.w.end());
            }
            d.exp_gate = upload(pg); d.exp_up = upload(pu); d.exp_down = upload(pd);
            if (L.moe.has_shared) {
                d.sh_gate = up_lin(L.moe.shared.gate_proj);
                d.sh_up   = up_lin(L.moe.shared.up_proj);
                d.sh_down = up_lin(L.moe.shared.down_proj);
            }
        }
    }

    // Persistent KV cache + small persistent buffers.
    const int64_t nheads = c.num_attention_heads, hc = c.kv_cache_head_dim();
    g->max_ctx = max_ctx > 0 ? max_ctx : 4096;
    g->Kc.resize(layers_.size());
    g->Vc.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        g->Kc[i] = dmalloc(g->max_ctx * nheads * hc);
        g->Vc[i] = dmalloc(g->max_ctx * nheads * hc);
    }
    cudaMalloc(&g->block_table, sizeof(int));
    int zero = 0;
    cudaMemcpy(g->block_table, &zero, sizeof(int), cudaMemcpyHostToDevice);
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
// Run the full 78-layer stack on `n` tokens whose embeddings are already in
// g->sc.hidden_d. Appends each layer's K/V to the persistent cache at absolute
// offset `start_pos` and attends over [0, start_pos+n). Leaves the final
// pre-norm hidden state in g->sc.hidden_d.
void run_block_stack(GpuState* g, const GLM52Config& c, int64_t n, int64_t start_pos) {
    using namespace glmserve::cuda;
    Scratch& s = g->sc;
    const int64_t hidden = c.hidden_size, nheads = c.num_attention_heads;
    const int64_t qlat = c.q_lora_rank, kvlat = c.kv_lora_rank;
    const int64_t nope = c.qk_nope_head_dim, rope = c.qk_rope_head_dim;
    const int64_t qk = c.qk_head_dim, vd = c.v_head_dim, hc = c.kv_cache_head_dim();
    const int64_t inter = c.intermediate_size, moei = c.moe_intermediate_size;
    const int64_t E = c.n_routed_experts, topk = c.num_experts_per_tok;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const bool il = c.rope_interleave;

    for (size_t l = 0; l < g->layers.size(); ++l) {
        const DLayer& d = g->layers[l];
        // attention sub-block
        rmsnorm(s.hidden_d, d.input_norm, s.normed, n, hidden, eps);
        gemm_fp32(s.normed, d.q_a.w, nullptr, s.qa, n, hidden, qlat);
        rmsnorm(s.qa, d.q_a_norm, s.qa, n, qlat, eps);
        gemm_fp32(s.qa, d.q_b.w, nullptr, s.q, n, qlat, nheads * qk);
        gemm_fp32(s.normed, d.kv_a.w, nullptr, s.kva, n, hidden, kvlat + rope);
        slice_rows(s.kva, s.ckv, n, kvlat + rope, 0, kvlat);
        rmsnorm(s.ckv, d.kv_a_norm, s.ckv, n, kvlat, eps);
        gemm_fp32(s.ckv, d.kv_b.w, nullptr, s.kvb, n, kvlat, nheads * (nope + vd));
        slice_rows(s.kva, s.kpe, n, kvlat + rope, kvlat, rope);
        rope_k(s.kpe, n, rope, start_pos, c.rope_theta, il);
        rope_q(s.q, n, nheads, qk, nope, rope, start_pos, c.rope_theta, il);
        // append this chunk's K/V at [start_pos, start_pos+n) in the persistent cache
        float* Kbase = g->Kc[l] + start_pos * nheads * hc;
        float* Vbase = g->Vc[l] + start_pos * nheads * hc;
        assemble_kv(s.kvb, s.kpe, Kbase, Vbase, n, nheads, nope, rope, vd, hc);
        // Attend over the cache so far (block_size = max_ctx => one block).
        // For short contexts GLM DSA is exact dense because ctx <= index_topk.
        // Past that, route through the DSA wrapper. Its current implementation
        // is the V1 recent-window sparse baseline; the learned lightning-indexer
        // top-k path will replace it behind the same kernel API.
        const int64_t ctx_after = start_pos + n;
        const bool sparse_dsa = c.use_dsa && ctx_after > c.index_topk;
        if (sparse_dsa) {
            attention_dsa_paged(s.q, g->Kc[l], g->Vc[l], g->block_table, n, start_pos,
                                nheads, nheads, qk, g->max_ctx, c.index_topk, scale, s.attn);
        } else if (n == 1) {
            attention_decode_paged(s.q, g->Kc[l], g->Vc[l], g->block_table, start_pos,
                                   nheads, nheads, qk, g->max_ctx, scale, s.attn,
                                   g->part_acc, g->part_m, g->part_l, kMaxSplits);
        } else {
            attention_dense_paged(s.q, g->Kc[l], g->Vc[l], g->block_table, n, start_pos,
                                  nheads, nheads, qk, g->max_ctx, scale, s.attn);
        }
        gemm_fp32(s.attn, d.o.w, nullptr, s.attn_proj, n, nheads * vd, hidden);
        add_inplace(s.hidden_d, s.attn_proj, n * hidden);

        // MLP sub-block
        rmsnorm(s.hidden_d, d.post_norm, s.normed, n, hidden, eps);
        if (d.is_dense) {
            gemm_fp32(s.normed, d.gate.w, nullptr, s.gbuf, n, hidden, inter);
            gemm_fp32(s.normed, d.up.w,   nullptr, s.ubuf, n, hidden, inter);
            silu_mul(s.gbuf, s.ubuf, s.gu, n * inter);
            gemm_fp32(s.gu, d.down.w, nullptr, s.mlp, n, inter, hidden);
        } else {
            gemm_fp32(s.normed, d.router.w, nullptr, s.rlogits, n, hidden, E);
            moe_router(s.rlogits, d.e_bias, n, E, topk, c.norm_topk_prob,
                       static_cast<float>(c.routed_scaling_factor), s.topk_ids, s.topk_w);
            cudaMemset(s.mlp, 0, n * hidden * sizeof(float));
            moe_expert_ffn(s.normed, s.topk_ids, s.topk_w, d.exp_gate, d.exp_up, d.exp_down,
                           n, topk, hidden, moei, E, s.mlp);
            if (d.sh_gate.w) {
                gemm_fp32(s.normed, d.sh_gate.w, nullptr, s.gbuf, n, hidden, moei);
                gemm_fp32(s.normed, d.sh_up.w,   nullptr, s.ubuf, n, hidden, moei);
                silu_mul(s.gbuf, s.ubuf, s.gu, n * moei);
                gemm_fp32(s.gu, d.sh_down.w, nullptr, s.shbuf, n, moei, hidden);
                add_inplace(s.mlp, s.shbuf, n * hidden);
            }
        }
        add_inplace(s.hidden_d, s.mlp, n * hidden);
    }
}

// Final RMSNorm + lm_head over the LAST row, copy logits to host.
std::vector<float> finish_logits(GpuState* g, const GLM52Config& c, int64_t n) {
    using namespace glmserve::cuda;
    rmsnorm(g->sc.hidden_d, g->final_norm, g->sc.normed, n, c.hidden_size,
            static_cast<float>(c.rms_norm_eps));
    gemm_fp32(g->sc.normed + (n - 1) * c.hidden_size, g->lm_head.w, nullptr,
              g->logits_d, 1, c.hidden_size, c.vocab_size);
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

    g->sc.ensure(n, cfg_);
    std::vector<int> toks(tokens.begin(), tokens.end());
    cudaMemcpy(g->sc.d_tokens, toks.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, n, cfg_.hidden_size);

    run_block_stack(g, cfg_, n, /*start_pos=*/0);
    std::vector<float> logits = finish_logits(g, cfg_, n);
    cuda_check("forward_gpu_prefill");
    g->cur_len = n;
    return logits;
}

std::vector<float> GLM52Model::forward_gpu_decode(int token, int64_t pos) {
    GLM_CHECK(gpu_state_ != nullptr, "forward_gpu_decode: call upload_to_gpu() first");
    using namespace glmserve::cuda;
    auto* g = static_cast<GpuState*>(gpu_state_);
    GLM_CHECK(pos < g->max_ctx, "forward_gpu_decode: position %lld >= GPU KV ctx %lld "
              "(raise --ctx)", (long long)pos, (long long)g->max_ctx);

    g->sc.ensure(1, cfg_);
    cudaMemcpy(g->sc.d_tokens, &token, sizeof(int), cudaMemcpyHostToDevice);
    embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, 1, cfg_.hidden_size);

    run_block_stack(g, cfg_, /*n=*/1, /*start_pos=*/pos);
    std::vector<float> logits = finish_logits(g, cfg_, 1);
    cuda_check("forward_gpu_decode");
    g->cur_len = pos + 1;
    return logits;
}

GLM52Model::~GLM52Model() {
    if (!gpu_state_) return;
    auto* g = static_cast<GpuState*>(gpu_state_);
    cudaFree(g->embed); cudaFree(g->final_norm); cudaFree(g->lm_head.w);
    for (auto& d : g->layers) {
        for (float* p : {d.input_norm, d.post_norm, d.q_a_norm, d.kv_a_norm,
                         d.q_a.w, d.q_b.w, d.kv_a.w, d.kv_b.w, d.o.w,
                         d.gate.w, d.up.w, d.down.w, d.router.w, d.e_bias,
                         d.exp_gate, d.exp_up, d.exp_down,
                         d.sh_gate.w, d.sh_up.w, d.sh_down.w})
            if (p) cudaFree(p);
    }
    for (float* p : g->Kc) if (p) cudaFree(p);
    for (float* p : g->Vc) if (p) cudaFree(p);
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
