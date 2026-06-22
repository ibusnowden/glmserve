#include "model.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>

namespace glmserve {

// ---------------------------------------------------------------------------
// Small CPU math helpers (the float32 reference path). These define the exact
// numerical conventions the engine commits to; tools/make_tiny_checkpoint.py
// mirrors them so logits can be compared bit-for-precision.
// ---------------------------------------------------------------------------

static unsigned hw_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1u : std::min(n, 32u);
}

// Run fn(i) for i in [0, n) across a few threads.
template <typename F>
static void parallel_for(int64_t n, F&& fn) {
    unsigned nt = std::min<unsigned>(hw_threads(), static_cast<unsigned>(std::max<int64_t>(1, n)));
    if (nt <= 1 || n < 256) { for (int64_t i = 0; i < n; ++i) fn(i); return; }
    std::vector<std::thread> pool;
    int64_t chunk = (n + nt - 1) / nt;
    for (unsigned t = 0; t < nt; ++t) {
        int64_t b = t * chunk, e = std::min<int64_t>(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&fn, b, e] { for (int64_t i = b; i < e; ++i) fn(i); });
    }
    for (auto& th : pool) th.join();
}

// y[n,out] = x[n,in] @ W[out,in]^T (+ bias[out])
static void linear_forward(const Linear& L, const float* x, float* y, int64_t n_tokens) {
    const int64_t in = L.in_features, out = L.out_features;
    const float* w = L.w.data();
    const bool bias = L.has_bias();
    parallel_for(n_tokens, [&](int64_t t) {
        const float* xt = x + t * in;
        float* yt = y + t * out;
        for (int64_t o = 0; o < out; ++o) {
            const float* wo = w + o * in;
            float acc = 0.0f;
            for (int64_t i = 0; i < in; ++i) acc += xt[i] * wo[i];
            yt[o] = bias ? acc + L.b[o] : acc;
        }
    });
}

// RMSNorm over the last `dim` elements of each of n rows, in place into out.
static void rmsnorm_rows(const float* x, const float* w, float* out,
                         int64_t n, int64_t dim, float eps) {
    parallel_for(n, [&](int64_t r) {
        const float* xr = x + r * dim;
        float* outr = out + r * dim;
        double ss = 0.0;
        for (int64_t i = 0; i < dim; ++i) ss += static_cast<double>(xr[i]) * xr[i];
        float inv = static_cast<float>(1.0 / std::sqrt(ss / dim + eps));
        for (int64_t i = 0; i < dim; ++i) outr[i] = xr[i] * inv * w[i];
    });
}

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// RoPE on a single dim-length vector at absolute position `pos`.
// interleave=true  -> GPT-J interleaved pairs (2i, 2i+1)  [GLM-5.2 / DeepSeek]
// interleave=false -> NeoX rotate-half pairs (i, i+dim/2)
static void rope_inplace(float* v, int64_t dim, int64_t pos, double theta, bool interleave) {
    int64_t half = dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        double freq = 1.0 / std::pow(theta, (2.0 * i) / dim);
        double ang = pos * freq;
        float c = static_cast<float>(std::cos(ang));
        float s = static_cast<float>(std::sin(ang));
        int64_t a = interleave ? 2 * i : i;
        int64_t b = interleave ? 2 * i + 1 : i + half;
        float x0 = v[a], x1 = v[b];
        v[a] = x0 * c - x1 * s;
        v[b] = x1 * c + x0 * s;
    }
}

// RMSNorm a single dim-length vector in place: x * rsqrt(mean(x^2)+eps) * w.
static void rmsnorm_vec(float* v, const float* w, int64_t dim, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < dim; ++i) ss += static_cast<double>(v[i]) * v[i];
    float inv = static_cast<float>(1.0 / std::sqrt(ss / dim + eps));
    for (int64_t i = 0; i < dim; ++i) v[i] = v[i] * inv * w[i];
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

// Load a [out,in] linear from the store, dequantizing to f32. Tries `${base}.weight`
// (and `${base}.qweight` for packed int4) plus an optional `${base}.bias`.
static Linear load_linear(const SafeTensors& st, const std::string& base,
                          bool required = true) {
    Linear L;
    std::string wname = base + ".weight";
    if (!st.has(wname) && st.has(base + ".qweight")) wname = base + ".qweight";
    if (!st.has(wname)) {
        GLM_CHECK(!required, "missing weight: %s", (base + ".weight").c_str());
        return L;
    }
    Tensor w = st.get(wname);
    GLM_CHECK(w.ndim() == 2, "%s expected 2-D, got %s", wname.c_str(), w.shape_str().c_str());
    L.out_features = w.dim(0);
    L.in_features  = w.dim(1);
    int64_t count  = L.out_features * L.in_features;
    L.w.resize(count);

    DType dt = w.dtype();
    if (dt == DType::kI8 && st.has(base + ".weight_scale")) {
        // per-output-channel or per-group symmetric int8
        Tensor sc = st.get(base + ".weight_scale");
        std::vector<float> scales = sc.dequant_to_f32();
        int64_t groups = static_cast<int64_t>(scales.size());
        int64_t gsize = std::max<int64_t>(1, count / std::max<int64_t>(1, groups));
        dequant_int8_to_f32(w.as<int8_t>(), scales.data(), count, gsize, L.w.data());
    } else if (dt == DType::kU8 && st.has(base + ".scales")) {
        // packed int4 (two nibbles/byte) + per-group scales
        Tensor sc = st.get(base + ".scales");
        std::vector<float> scales = sc.dequant_to_f32();
        int64_t groups = static_cast<int64_t>(scales.size());
        int64_t gsize = std::max<int64_t>(1, count / std::max<int64_t>(1, groups));
        dequant_int4_to_f32(w.as<uint8_t>(), scales.data(), count, gsize, L.w.data());
    } else {
        dequant_to_f32(dt, w.data(), L.w.data(), count);
    }

    if (st.has(base + ".bias")) {
        Tensor b = st.get(base + ".bias");
        L.b = b.dequant_to_f32();
    }
    return L;
}

static RMSNormW load_norm(const SafeTensors& st, const std::string& name,
                          bool required = true) {
    RMSNormW n;
    if (!st.has(name)) {
        GLM_CHECK(!required, "missing norm: %s", name.c_str());
        return n;
    }
    Tensor t = st.get(name);
    n.w = t.dequant_to_f32();
    n.dim = static_cast<int64_t>(n.w.size());
    return n;
}

static DSAIndexer load_dsa_indexer(const SafeTensors& st, const std::string& base) {
    DSAIndexer ix;
    if (!st.has(base + ".wq_b.weight") && !st.has(base + ".wk.weight") &&
        !st.has(base + ".weights_proj.weight")) {
        return ix;
    }
    ix.wq_b = load_linear(st, base + ".wq_b");
    ix.wk = load_linear(st, base + ".wk");
    ix.weights_proj = load_linear(st, base + ".weights_proj");
    ix.k_norm = load_norm(st, base + ".k_norm.weight");
    if (st.has(base + ".k_norm.bias")) ix.k_norm_bias = st.get(base + ".k_norm.bias").dequant_to_f32();
    GLM_CHECK(ix.valid(), "incomplete DSA indexer weights under %s", base.c_str());
    return ix;
}

// First present among candidates, else "".
static std::string first_present(const SafeTensors& st,
                                 const std::vector<std::string>& cands) {
    for (auto& c : cands) {
        if (st.has(c + ".weight") || st.has(c + ".qweight")) return c;
    }
    return "";
}

// ---------------------------------------------------------------------------
// Tensor-parallel weight slicing (Megatron-style). A column-parallel linear
// keeps a contiguous slice of the OUTPUT rows (out_features); a row-parallel
// linear keeps a contiguous slice of the INPUT columns (in_features), and its
// partial output is summed across the TP group with all_reduce. Both assume the
// split dimension divides evenly across tp_size, which holds for GLM-5.2's head
// and FFN dims. q_b/kv_b are column-sharded by head and o_proj row-sharded by
// head (heads are contiguous), so a rank owns the same head slice end to end.
// ---------------------------------------------------------------------------
static void shard_linear_columns(Linear& L, int tp_rank, int tp_size) {
    if (tp_size <= 1 || !L.valid()) return;
    const int64_t out = L.out_features, in = L.in_features;
    GLM_CHECK(out % tp_size == 0, "column-parallel out_features %lld not divisible by tp %d",
              (long long)out, tp_size);
    const int64_t out_l = out / tp_size;
    const int64_t r0 = static_cast<int64_t>(tp_rank) * out_l;
    std::vector<float> w(static_cast<size_t>(out_l * in));
    std::memcpy(w.data(), L.w.data() + r0 * in, static_cast<size_t>(out_l * in) * sizeof(float));
    L.w.swap(w);
    if (L.has_bias()) {
        std::vector<float> b(L.b.begin() + r0, L.b.begin() + r0 + out_l);
        L.b.swap(b);
    }
    L.out_features = out_l;
}

static void shard_linear_rows(Linear& L, int tp_rank, int tp_size) {
    if (tp_size <= 1 || !L.valid()) return;
    const int64_t out = L.out_features, in = L.in_features;
    GLM_CHECK(in % tp_size == 0, "row-parallel in_features %lld not divisible by tp %d",
              (long long)in, tp_size);
    const int64_t in_l = in / tp_size;
    const int64_t c0 = static_cast<int64_t>(tp_rank) * in_l;
    std::vector<float> w(static_cast<size_t>(out * in_l));
    for (int64_t o = 0; o < out; ++o)
        std::memcpy(w.data() + o * in_l, L.w.data() + o * in + c0,
                    static_cast<size_t>(in_l) * sizeof(float));
    L.w.swap(w);
    // The bias adds to the post-all-reduce sum, so keep it on rank 0 only (else
    // it would be counted tp_size times).
    if (L.has_bias() && tp_rank != 0) std::fill(L.b.begin(), L.b.end(), 0.0f);
    L.in_features = in_l;
}

// Tensor-parallel shard of one transformer block's linears, in place. Attention
// is split across heads (q_b/kv_b column, o_proj row); MLP/MoE FFNs are split
// across the intermediate dim (gate/up column, down row). q_a/kv_a, the router,
// norms, embeddings and lm_head stay replicated.
static void shard_layer_tp(Layer& L, int tp_rank, int tp_size) {
    if (tp_size <= 1) return;
    shard_linear_columns(L.q_b_proj, tp_rank, tp_size);
    shard_linear_columns(L.kv_b_proj, tp_rank, tp_size);
    shard_linear_rows(L.o_proj, tp_rank, tp_size);
    if (L.is_dense) {
        shard_linear_columns(L.dense_mlp.gate_proj, tp_rank, tp_size);
        shard_linear_columns(L.dense_mlp.up_proj, tp_rank, tp_size);
        shard_linear_rows(L.dense_mlp.down_proj, tp_rank, tp_size);
    } else {
        for (Expert& E : L.moe.experts) {
            shard_linear_columns(E.gate_proj, tp_rank, tp_size);
            shard_linear_columns(E.up_proj, tp_rank, tp_size);
            shard_linear_rows(E.down_proj, tp_rank, tp_size);
        }
        if (L.moe.has_shared) {
            shard_linear_columns(L.moe.shared.gate_proj, tp_rank, tp_size);
            shard_linear_columns(L.moe.shared.up_proj, tp_rank, tp_size);
            shard_linear_rows(L.moe.shared.down_proj, tp_rank, tp_size);
        }
    }
}

void GLM52Model::set_distributed(Communicator* comm) {
    comm_ = comm;
    dist_ = comm ? comm->config() : DistConfig{};
}

void GLM52Model::load(const SafeTensors& st, int64_t max_layers) {
    Timer timer;
    const auto& c = cfg_;

    // Pipeline partition: the full stack (after any max_layers truncation) is
    // split into pp_size contiguous stages; this rank keeps [begin,end) only.
    int64_t total_layers = (max_layers > 0) ? std::min(max_layers, c.num_hidden_layers)
                                            : c.num_hidden_layers;
    LayerRange range = partition_layers(total_layers, dist_.pp_stage(), dist_.pp_size);
    layer_begin_ = range.begin;
    const bool first = dist_.is_first_stage();
    const bool last  = dist_.is_last_stage();
    const bool tied  = !st.has("lm_head.weight");

    // Tensor-parallel: this rank owns num_attention_heads / tp_size heads.
    const int tp = std::max(1, dist_.tp_size);
    const int tp_rank = dist_.tp_rank();
    GLM_CHECK(c.num_attention_heads % tp == 0,
              "num_attention_heads %lld not divisible by tp_size %d",
              (long long)c.num_attention_heads, tp);
    tp_heads_ = c.num_attention_heads / tp;

    // Embeddings: needed to embed the input ids on the first stage, and to
    // source a tied lm_head on the last stage. Skipped on interior stages.
    if (first || (last && tied)) {
        std::string emb = st.has("model.embed_tokens.weight") ? "model.embed_tokens.weight"
                          : "embed_tokens.weight";
        GLM_CHECK(st.has(emb), "missing embedding weight (model.embed_tokens.weight)");
        Tensor e = st.get(emb);
        embed_tokens_ = e.dequant_to_f32();
        GLM_CHECK(e.dim(0) == c.vocab_size && e.dim(1) == c.hidden_size,
                  "embedding shape %s != [vocab=%lld, hidden=%lld]",
                  e.shape_str().c_str(), (long long)c.vocab_size, (long long)c.hidden_size);
    }

    // Final norm + lm_head produce logits, so they live only on the last stage.
    if (last) {
        final_norm_ = load_norm(st, st.has("model.norm.weight") ? "model.norm.weight" : "norm.weight");
        if (!tied) {
            lm_head_ = load_linear(st, "lm_head");
            tied_embeddings_ = false;
        } else {
            // tied to embeddings: lm_head == embed_tokens
            lm_head_.w = embed_tokens_;
            lm_head_.out_features = c.vocab_size;
            lm_head_.in_features  = c.hidden_size;
            tied_embeddings_ = true;
            GLM_INFO("lm_head tied to embeddings");
        }
    }

    int64_t n_local = range.end - range.begin;
    layers_.resize(n_local);
    int64_t loaded_indexers = 0;

    for (int64_t li = 0; li < n_local; ++li) {
        int64_t i = range.begin + li;           // global layer index
        Layer& L = layers_[li];
        std::string p = "model.layers." + std::to_string(i) + ".";

        L.input_norm     = load_norm(st, p + "input_layernorm.weight");
        L.post_attn_norm = load_norm(st, p + "post_attention_layernorm.weight");

        // MLA: Q via q_a_proj -> q_a_layernorm -> q_b_proj; KV via
        // kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj. The DSA
        // indexer.* weights are present but unused on the V0 dense path
        // (index_topk=2048 >> typical prompt -> attention is exact dense).
        L.q_a_proj  = load_linear(st, p + "self_attn.q_a_proj");
        L.q_a_norm  = load_norm(st, p + "self_attn.q_a_layernorm.weight");
        L.q_b_proj  = load_linear(st, p + "self_attn.q_b_proj");
        L.kv_a_proj = load_linear(st, p + "self_attn.kv_a_proj_with_mqa");
        L.kv_a_norm = load_norm(st, p + "self_attn.kv_a_layernorm.weight");
        L.kv_b_proj = load_linear(st, p + "self_attn.kv_b_proj");
        L.o_proj    = load_linear(st, p + "self_attn.o_proj");
        L.indexer   = load_dsa_indexer(st, p + "self_attn.indexer");
        if (L.indexer.valid()) ++loaded_indexers;

        L.is_dense = c.is_dense_layer(i);
        if (L.is_dense) {
            L.dense_mlp.gate_proj = load_linear(st, p + "mlp.gate_proj");
            L.dense_mlp.up_proj   = load_linear(st, p + "mlp.up_proj");
            L.dense_mlp.down_proj = load_linear(st, p + "mlp.down_proj");
        } else {
            // router gate
            std::string gate = st.has(p + "mlp.gate.weight") ? p + "mlp.gate"
                                                             : p + "mlp.router";
            L.moe.router = load_linear(st, gate);
            // sigmoid score-correction bias (aux-loss-free routing)
            for (const std::string& bn : {p + "mlp.gate.e_score_correction_bias",
                                          p + "mlp.e_score_correction_bias"}) {
                if (st.has(bn)) { L.moe.e_bias = st.get(bn).dequant_to_f32(); break; }
            }
            // experts
            L.moe.experts.resize(c.n_routed_experts);
            for (int64_t e = 0; e < c.n_routed_experts; ++e) {
                std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
                L.moe.experts[e].gate_proj = load_linear(st, ep + "gate_proj");
                L.moe.experts[e].up_proj   = load_linear(st, ep + "up_proj");
                L.moe.experts[e].down_proj = load_linear(st, ep + "down_proj");
            }
            // shared expert
            std::string sh = first_present(st, {p + "mlp.shared_experts.gate_proj",
                                                p + "mlp.shared_expert.gate_proj"});
            if (!sh.empty()) {
                std::string base = sh.substr(0, sh.size() - std::string(".gate_proj").size());
                L.moe.shared.gate_proj = load_linear(st, base + ".gate_proj");
                L.moe.shared.up_proj   = load_linear(st, base + ".up_proj");
                L.moe.shared.down_proj = load_linear(st, base + ".down_proj");
                L.moe.has_shared = true;
            }
        }
        // Tensor-parallel: keep only this rank's head / FFN slice of the block.
        shard_layer_tp(L, tp_rank, tp);

        if ((li + 1) % 8 == 0 || li + 1 == n_local)
            GLM_DEBUG("loaded layer %lld/%lld (global %lld)",
                      (long long)(li + 1), (long long)n_local, (long long)i);
    }

    if (dist_.pp_size > 1) {
        GLM_INFO("model loaded (PP stage %d/%d): global layers [%lld,%lld) = %lld local, %.1f s",
                 dist_.pp_stage(), dist_.pp_size, (long long)range.begin, (long long)range.end,
                 (long long)n_local, timer.ms() / 1000.0);
    } else {
        GLM_INFO("model loaded: %lld layers in %.1f s", (long long)n_local, timer.ms() / 1000.0);
    }
    if (loaded_indexers > 0) {
        GLM_INFO("loaded DSA indexer weights for %lld/%lld transformer layers",
                 (long long)loaded_indexers, (long long)n_local);
    }
}

void GLM52Model::embed(int token_id, float* out) const {
    GLM_CHECK(token_id >= 0 && token_id < cfg_.vocab_size, "token id %d out of range", token_id);
    std::memcpy(out, embed_tokens_.data() + static_cast<int64_t>(token_id) * cfg_.hidden_size,
                cfg_.hidden_size * sizeof(float));
}

// ---------------------------------------------------------------------------
// MLA attention (DeepSeek-style latent attention + decoupled RoPE).
//
// Q : hidden -> q_a (q_lora_rank) -> RMSNorm -> q_b -> [H, qk_head_dim];
//     each head splits into nope(qk_nope) | pe(qk_rope), RoPE on the pe part.
// KV: hidden -> [c_kv (kv_lora_rank) | k_pe (qk_rope)]; c_kv -> RMSNorm ->
//     kv_b -> [H, qk_nope + v_head]; k_pe RoPE'd once and shared across heads.
// We materialize per-head K = [k_nope | k_pe] (qk_head_dim) and V (v_head_dim)
// into the paged cache (naive/unabsorbed form), then run standard causal
// attention with head_dim = qk_head_dim == v_head_dim. The latent-cache
// optimization (store the 576-wide latent instead) is a future memory win.
// DSA indexer is a no-op here: index_topk(2048) >> typical prompt -> exact dense.
// ---------------------------------------------------------------------------
void GLM52Model::attention(const Layer& L, int64_t layer_idx, const float* normed,
                           float* attn_out, int64_t n_tokens, int64_t start_pos,
                           SequenceKV& kv) {
    const auto& c = cfg_;
    const int64_t H = tp_heads_;   // attention heads owned by this TP rank
    const int64_t nope = c.qk_nope_head_dim, rope = c.qk_rope_head_dim;
    const int64_t qk = c.qk_head_dim, vd = c.v_head_dim;
    const int64_t hc = c.kv_cache_head_dim();   // per-head cache slot width
    const int64_t kvlat = c.kv_lora_rank;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const bool il = c.rope_interleave;

    // Q latent -> q_b
    std::vector<float> qa(n_tokens * c.q_lora_rank);
    linear_forward(L.q_a_proj, normed, qa.data(), n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t)
        rmsnorm_vec(qa.data() + t * c.q_lora_rank, L.q_a_norm.w.data(), c.q_lora_rank, eps);
    std::vector<float> q(n_tokens * H * qk);
    linear_forward(L.q_b_proj, qa.data(), q.data(), n_tokens);

    // KV latent: [c_kv | k_pe]
    const int64_t kva_dim = kvlat + rope;
    std::vector<float> kva(n_tokens * kva_dim);
    linear_forward(L.kv_a_proj, normed, kva.data(), n_tokens);
    // normalize the c_kv slice, then up-project to per-head [k_nope | v]
    std::vector<float> ckv(n_tokens * kvlat);
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::memcpy(ckv.data() + t * kvlat, kva.data() + t * kva_dim, kvlat * sizeof(float));
        rmsnorm_vec(ckv.data() + t * kvlat, L.kv_a_norm.w.data(), kvlat, eps);
    }
    std::vector<float> kvb(n_tokens * H * (nope + vd));
    linear_forward(L.kv_b_proj, ckv.data(), kvb.data(), n_tokens);

    // Per token: RoPE the q pe-parts and the shared k_pe, then write per-head
    // K=[k_nope|k_pe] and V into the paged cache.
    for (int64_t t = 0; t < n_tokens; ++t) {
        int64_t pos = start_pos + t;
        // shared key rope part
        float* k_pe = kva.data() + t * kva_dim + kvlat;          // [rope]
        rope_inplace(k_pe, rope, pos, c.rope_theta, il);

        float* qt = q.data() + t * H * qk;
        const float* kvt = kvb.data() + t * H * (nope + vd);
        float* kslot = kv.k_slot(layer_idx, pos);
        float* vslot = kv.v_slot(layer_idx, pos);
        for (int64_t h = 0; h < H; ++h) {
            // rope the q pe sub-vector (offset nope, length rope)
            rope_inplace(qt + h * qk + nope, rope, pos, c.rope_theta, il);
            // K[h] = [k_nope(nope) | k_pe(rope)]
            float* kh = kslot + h * hc;
            std::memcpy(kh, kvt + h * (nope + vd), nope * sizeof(float));
            std::memcpy(kh + nope, k_pe, rope * sizeof(float));
            // V[h] = v (v_head_dim)
            std::memcpy(vslot + h * hc, kvt + h * (nope + vd) + nope, vd * sizeof(float));
        }
    }

    // Standard causal attention with head_dim = qk (K) / vd (V) inside an hc slot.
    parallel_for(n_tokens * H, [&](int64_t idx) {
        int64_t t = idx / H, h = idx % H;
        int64_t pos = start_pos + t;
        const float* qh = q.data() + (t * H + h) * qk;
        std::vector<float> scores(pos + 1);
        float maxv = -1e30f;
        for (int64_t j = 0; j <= pos; ++j) {
            const float* kj = kv.k_slot(layer_idx, j) + h * hc;
            float dot = 0.0f;
            for (int64_t d = 0; d < qk; ++d) dot += qh[d] * kj[d];
            dot *= scale;
            scores[j] = dot;
            maxv = std::max(maxv, dot);
        }
        float sum = 0.0f;
        for (int64_t j = 0; j <= pos; ++j) { scores[j] = std::exp(scores[j] - maxv); sum += scores[j]; }
        float inv = 1.0f / sum;
        float* out = attn_out + (t * H + h) * vd;   // attn_out is [n, H, v_head_dim]
        for (int64_t d = 0; d < vd; ++d) out[d] = 0.0f;
        for (int64_t j = 0; j <= pos; ++j) {
            const float* vj = kv.v_slot(layer_idx, j) + h * hc;
            float wgt = scores[j] * inv;
            for (int64_t d = 0; d < vd; ++d) out[d] += wgt * vj[d];
        }
    });
}

void GLM52Model::dense_mlp(const DenseMLP& m, const float* x, float* out,
                           int64_t n_tokens) const {
    int64_t ffn = m.gate_proj.out_features;
    std::vector<float> g(n_tokens * ffn), u(n_tokens * ffn);
    linear_forward(m.gate_proj, x, g.data(), n_tokens);
    linear_forward(m.up_proj,   x, u.data(), n_tokens);
    for (int64_t i = 0; i < n_tokens * ffn; ++i) g[i] = silu(g[i]) * u[i];
    linear_forward(m.down_proj, g.data(), out, n_tokens);
}

// MoE: sigmoid scoring, top-k (aux-loss-free bias for selection), optional norm,
// routed_scaling_factor, plus the shared expert. Reference token-by-token loop.
void GLM52Model::moe_mlp(const MoEMLP& m, const float* x, float* out,
                         int64_t n_tokens) const {
    const auto& c = cfg_;
    const int64_t hidden = c.hidden_size;
    const int64_t n_exp = c.n_routed_experts;
    const int64_t topk = c.num_experts_per_tok;

    // router logits -> sigmoid scores
    std::vector<float> logits(n_tokens * n_exp);
    linear_forward(m.router, x, logits.data(), n_tokens);

    std::fill(out, out + n_tokens * hidden, 0.0f);

    parallel_for(n_tokens, [&](int64_t t) {
        const float* xt = x + t * hidden;
        const float* lt = logits.data() + t * n_exp;

        std::vector<float> score(n_exp), choose(n_exp);
        for (int64_t e = 0; e < n_exp; ++e) {
            score[e]  = sigmoidf(lt[e]);
            choose[e] = score[e] + (m.e_bias.empty() ? 0.0f : m.e_bias[e]);
        }
        // top-k by choose[]
        std::vector<int> idx(n_exp);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int a, int b) { return choose[a] > choose[b]; });

        float wsum = 0.0f;
        std::vector<float> w(topk);
        for (int64_t kk = 0; kk < topk; ++kk) { w[kk] = score[idx[kk]]; wsum += w[kk]; }
        if (c.norm_topk_prob && wsum > 0.0f)
            for (int64_t kk = 0; kk < topk; ++kk) w[kk] /= wsum;
        for (int64_t kk = 0; kk < topk; ++kk) w[kk] *= static_cast<float>(c.routed_scaling_factor);

        float* ot = out + t * hidden;
        std::vector<float> g, u, tmp(hidden);
        for (int64_t kk = 0; kk < topk; ++kk) {
            const Expert& E = m.experts[idx[kk]];
            int64_t ffn = E.gate_proj.out_features;
            g.assign(ffn, 0.0f); u.assign(ffn, 0.0f);
            linear_forward(E.gate_proj, xt, g.data(), 1);
            linear_forward(E.up_proj,   xt, u.data(), 1);
            for (int64_t i = 0; i < ffn; ++i) g[i] = silu(g[i]) * u[i];
            linear_forward(E.down_proj, g.data(), tmp.data(), 1);
            for (int64_t i = 0; i < hidden; ++i) ot[i] += w[kk] * tmp[i];
        }
        // shared expert (always-on, weight 1.0)
        if (m.has_shared) {
            int64_t ffn = m.shared.gate_proj.out_features;
            g.assign(ffn, 0.0f); u.assign(ffn, 0.0f);
            linear_forward(m.shared.gate_proj, xt, g.data(), 1);
            linear_forward(m.shared.up_proj,   xt, u.data(), 1);
            for (int64_t i = 0; i < ffn; ++i) g[i] = silu(g[i]) * u[i];
            linear_forward(m.shared.down_proj, g.data(), tmp.data(), 1);
            for (int64_t i = 0; i < hidden; ++i) ot[i] += tmp[i];
        }
    });
}

void GLM52Model::run_layer(int64_t layer_idx, float* hidden, int64_t n_tokens,
                           int64_t start_pos, SequenceKV& kv) {
    const Layer& L = layers_[layer_idx];
    const int64_t H = cfg_.hidden_size;
    const float eps = static_cast<float>(cfg_.rms_norm_eps);
    const bool tp = dist_.tp_size > 1;   // tensor-parallel: o_proj / down are row-sharded
    const int64_t local_attn_out = tp_heads_ * cfg_.v_head_dim;

    std::vector<float> normed(n_tokens * H);
    std::vector<float> attn(n_tokens * local_attn_out);
    std::vector<float> attn_proj(n_tokens * H);

    // --- attention sub-block ---
    // Under TP each rank attends its head slice and o_proj (row-parallel) yields
    // a partial residual contribution; all_reduce sums them across the TP group.
    rmsnorm_rows(hidden, L.input_norm.w.data(), normed.data(), n_tokens, H, eps);
    attention(L, layer_idx, normed.data(), attn.data(), n_tokens, start_pos, kv);
    linear_forward(L.o_proj, attn.data(), attn_proj.data(), n_tokens);
    if (tp) {
        GLM_CHECK(comm_, "tensor-parallel forward requires a communicator");
        comm_->all_reduce_sum(attn_proj.data(), n_tokens * H);
    }
    for (int64_t i = 0; i < n_tokens * H; ++i) hidden[i] += attn_proj[i];

    // --- MLP sub-block --- (gate/up column-parallel, down row-parallel -> reduce)
    rmsnorm_rows(hidden, L.post_attn_norm.w.data(), normed.data(), n_tokens, H, eps);
    std::vector<float> mlp_out(n_tokens * H);
    if (L.is_dense) dense_mlp(L.dense_mlp, normed.data(), mlp_out.data(), n_tokens);
    else            moe_mlp(L.moe, normed.data(), mlp_out.data(), n_tokens);
    if (tp) comm_->all_reduce_sum(mlp_out.data(), n_tokens * H);
    for (int64_t i = 0; i < n_tokens * H; ++i) hidden[i] += mlp_out[i];
}

std::vector<float> GLM52Model::forward(const std::vector<int>& tokens, int64_t start_pos,
                                       SequenceKV& kv, std::vector<float>* all_logits) {
    const int64_t n = static_cast<int64_t>(tokens.size());
    const int64_t H = cfg_.hidden_size;
    GLM_CHECK(n > 0, "forward called with no tokens");

    kv.reserve(n);

    // Pipeline input: the first stage embeds the token ids; any later stage
    // receives the upstream stage's hidden state over the PP link.
    std::vector<float> hidden(n * H);
    if (dist_.is_first_stage()) {
        for (int64_t t = 0; t < n; ++t) embed(tokens[t], hidden.data() + t * H);
    } else {
        GLM_CHECK(comm_, "non-first pipeline stage requires a communicator");
        comm_->pipeline_recv_prev(hidden.data(), n * H);
    }

    // This stage's slice of the block stack (layers are local-indexed, and so is
    // the per-stage KV cache, so run_layer indexes both with the same l).
    for (int64_t l = 0; l < num_layers(); ++l)
        run_layer(l, hidden.data(), n, start_pos, kv);

    kv.length = start_pos + n;

    // Pipeline output: a non-final stage forwards the hidden state downstream and
    // produces no logits; only the last stage runs final-norm + lm_head.
    if (!dist_.is_last_stage()) {
        GLM_CHECK(comm_, "non-last pipeline stage requires a communicator");
        comm_->pipeline_send_next(hidden.data(), n * H);
        if (all_logits) all_logits->clear();
        return {};
    }

    // final norm
    std::vector<float> normed(n * H);
    rmsnorm_rows(hidden.data(), final_norm_.w.data(), normed.data(), n, H,
                 static_cast<float>(cfg_.rms_norm_eps));

    // logits for the last token (and optionally all positions)
    std::vector<float> last_logits(cfg_.vocab_size);
    linear_forward(lm_head_, normed.data() + (n - 1) * H, last_logits.data(), 1);

    if (all_logits) {
        all_logits->resize(n * cfg_.vocab_size);
        linear_forward(lm_head_, normed.data(), all_logits->data(), n);
    }
    return last_logits;
}

// CPU-only build: GPU entry points are stubs (the real ones live in model_gpu.cpp).
#ifndef GLMSERVE_CUDA
GLM52Model::~GLM52Model() {}
bool GLM52Model::upload_to_gpu(int64_t) {
    GLM_WARN("upload_to_gpu: built without CUDA (rebuild with GPU=1)");
    return false;
}
int64_t GLM52Model::gpu_kv_ctx() const { return 0; }
std::vector<float> GLM52Model::forward_gpu_prefill(const std::vector<int>&) {
    GLM_CHECK(false, "forward_gpu_prefill: built without CUDA (rebuild with GPU=1)");
    return {};
}
std::vector<float> GLM52Model::forward_gpu_decode(int, int64_t) {
    GLM_CHECK(false, "forward_gpu_decode: built without CUDA (rebuild with GPU=1)");
    return {};
}
#endif

}  // namespace glmserve
