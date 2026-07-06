#include "model.hpp"
#include "common.hpp"
#include "model_gguf.hpp"
#include "gguf_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
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
    GLM_CHECK(L.has_f32(), "linear_forward called on quantized-only weight [%lld,%lld]",
              (long long)L.out_features, (long long)L.in_features);
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

// Standard LayerNorm over rows, used by the GLM DSA indexer key path.
static void layernorm_rows(const float* x, const float* w, const float* b, float* out,
                           int64_t n, int64_t dim, float eps) {
    parallel_for(n, [&](int64_t r) {
        const float* xr = x + r * dim;
        float* outr = out + r * dim;
        double mean = 0.0;
        for (int64_t i = 0; i < dim; ++i) mean += xr[i];
        mean /= dim;
        double var = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double d = static_cast<double>(xr[i]) - mean;
            var += d * d;
        }
        float inv = static_cast<float>(1.0 / std::sqrt(var / dim + eps));
        for (int64_t i = 0; i < dim; ++i) {
            float bias = b ? b[i] : 0.0f;
            outr[i] = (static_cast<float>(xr[i] - mean) * inv) * w[i] + bias;
        }
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

static void check_dsa_indexer_shapes(const DSAIndexer& ix, const GLM52Config& c) {
    GLM_CHECK(ix.wq_b.in_features == c.q_lora_rank,
              "DSA indexer wq_b input %lld != q_lora_rank %lld",
              (long long)ix.wq_b.in_features, (long long)c.q_lora_rank);
    GLM_CHECK(ix.wq_b.out_features == c.index_n_heads * c.index_head_dim,
              "DSA indexer wq_b output %lld != index_n_heads*index_head_dim %lld",
              (long long)ix.wq_b.out_features,
              (long long)(c.index_n_heads * c.index_head_dim));
    GLM_CHECK(ix.wk.in_features == c.hidden_size && ix.wk.out_features == c.index_head_dim,
              "DSA indexer wk shape [%lld,%lld] != [%lld,%lld]",
              (long long)ix.wk.out_features, (long long)ix.wk.in_features,
              (long long)c.index_head_dim, (long long)c.hidden_size);
    GLM_CHECK(ix.weights_proj.in_features == c.hidden_size &&
                  ix.weights_proj.out_features == c.index_n_heads,
              "DSA indexer weights_proj shape [%lld,%lld] != [%lld,%lld]",
              (long long)ix.weights_proj.out_features,
              (long long)ix.weights_proj.in_features,
              (long long)c.index_n_heads, (long long)c.hidden_size);
    GLM_CHECK(ix.k_norm.dim == c.index_head_dim,
              "DSA indexer k_norm dim %lld != index_head_dim %lld",
              (long long)ix.k_norm.dim, (long long)c.index_head_dim);
    GLM_CHECK(ix.k_norm_bias.empty() ||
                  static_cast<int64_t>(ix.k_norm_bias.size()) == c.index_head_dim,
              "DSA indexer k_norm bias dim %zu != index_head_dim %lld",
              ix.k_norm_bias.size(), (long long)c.index_head_dim);
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
    DType dt = w.dtype();
    if (wname.size() >= 8 && wname.compare(wname.size() - 8, 8, ".qweight") == 0) {
        GLM_CHECK(dt == DType::kU8, "%s expected U8 packed int4, got %s",
                  wname.c_str(), dtype_name(dt));
        GLM_CHECK(st.has(base + ".scales"), "missing int4 scales: %s.scales", base.c_str());
        Tensor sc = st.get(base + ".scales");
        GLM_CHECK(sc.ndim() == 2 && sc.dim(0) == w.dim(0),
                  "%s.scales expected [out,groups], got %s", base.c_str(), sc.shape_str().c_str());
        L.out_features = w.dim(0);
        L.in_features = w.dim(1) * 2;  // one byte stores two input-column weights
        L.scales = sc.dequant_to_f32();
        const int64_t groups_per_row = sc.dim(1);
        GLM_CHECK(groups_per_row > 0 && L.in_features % groups_per_row == 0,
                  "cannot infer int4 group size for %s: in=%lld groups=%lld",
                  base.c_str(), (long long)L.in_features, (long long)groups_per_row);
        L.group_size = L.in_features / groups_per_row;
        L.quantized_int4 = true;
        L.qweight.resize(static_cast<size_t>(w.numel()));
        std::memcpy(L.qweight.data(), w.data(), L.qweight.size());

        const char* keep_quant = std::getenv("GLMSERVE_QUANT_ONLY");
        if (!keep_quant || !*keep_quant || keep_quant[0] == '0') {
            int64_t count = L.out_features * L.in_features;
            L.w.resize(static_cast<size_t>(count));
            dequant_int4_to_f32(L.qweight.data(), L.scales.data(), count, L.group_size, L.w.data());
        }
    } else {
        L.out_features = w.dim(0);
        L.in_features  = w.dim(1);
        int64_t count  = L.out_features * L.in_features;
        L.w.resize(count);

        if (dt == DType::kI8 && st.has(base + ".weight_scale")) {
        // per-output-channel or per-group symmetric int8
            Tensor sc = st.get(base + ".weight_scale");
            std::vector<float> scales = sc.dequant_to_f32();
            int64_t groups = static_cast<int64_t>(scales.size());
            int64_t gsize = std::max<int64_t>(1, count / std::max<int64_t>(1, groups));
            dequant_int8_to_f32(w.as<int8_t>(), scales.data(), count, gsize, L.w.data());
        } else {
            dequant_to_f32(dt, w.data(), L.w.data(), count);
        }
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
    if (!st.has(base + ".wq_b.weight") && !st.has(base + ".wq_b.qweight") &&
        !st.has(base + ".wk.weight") && !st.has(base + ".wk.qweight") &&
        !st.has(base + ".weights_proj.weight") && !st.has(base + ".weights_proj.qweight")) {
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

static void load_layer_body(const SafeTensors& st, const GLM52Config& c,
                            int64_t layer_id, Layer& L) {
    std::string p = "model.layers." + std::to_string(layer_id) + ".";

    L.input_norm     = load_norm(st, p + "input_layernorm.weight");
    L.post_attn_norm = load_norm(st, p + "post_attention_layernorm.weight");

    L.q_a_proj  = load_linear(st, p + "self_attn.q_a_proj");
    L.q_a_norm  = load_norm(st, p + "self_attn.q_a_layernorm.weight");
    L.q_b_proj  = load_linear(st, p + "self_attn.q_b_proj");
    L.kv_a_proj = load_linear(st, p + "self_attn.kv_a_proj_with_mqa");
    L.kv_a_norm = load_norm(st, p + "self_attn.kv_a_layernorm.weight");
    L.kv_b_proj = load_linear(st, p + "self_attn.kv_b_proj");
    L.o_proj    = load_linear(st, p + "self_attn.o_proj");
    L.indexer   = load_dsa_indexer(st, p + "self_attn.indexer");

    L.is_dense = c.is_dense_layer(layer_id);
    if (L.is_dense) {
        L.dense_mlp.gate_proj = load_linear(st, p + "mlp.gate_proj");
        L.dense_mlp.up_proj   = load_linear(st, p + "mlp.up_proj");
        L.dense_mlp.down_proj = load_linear(st, p + "mlp.down_proj");
    } else {
        std::string gate = (st.has(p + "mlp.gate.weight") || st.has(p + "mlp.gate.qweight"))
                         ? p + "mlp.gate" : p + "mlp.router";
        L.moe.router = load_linear(st, gate);
        for (const std::string& bn : {p + "mlp.gate.e_score_correction_bias",
                                      p + "mlp.e_score_correction_bias"}) {
            if (st.has(bn)) { L.moe.e_bias = st.get(bn).dequant_to_f32(); break; }
        }
        L.moe.experts.resize(c.n_routed_experts);
        for (int64_t e = 0; e < c.n_routed_experts; ++e) {
            std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            L.moe.experts[e].gate_proj = load_linear(st, ep + "gate_proj");
            L.moe.experts[e].up_proj   = load_linear(st, ep + "up_proj");
            L.moe.experts[e].down_proj = load_linear(st, ep + "down_proj");
        }
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
    if (L.has_f32()) {
        std::vector<float> w(static_cast<size_t>(out_l * in));
        std::memcpy(w.data(), L.w.data() + r0 * in,
                    static_cast<size_t>(out_l * in) * sizeof(float));
        L.w.swap(w);
    }
    if (L.has_bias()) {
        std::vector<float> b(L.b.begin() + r0, L.b.begin() + r0 + out_l);
        L.b.swap(b);
    }
    if (L.quantized_int4) {
        const int64_t packed_in = (in + 1) / 2;
        const int64_t groups_per_row = static_cast<int64_t>(L.scales.size()) / out;
        std::vector<uint8_t> q(static_cast<size_t>(out_l * packed_in));
        std::vector<float> sc(static_cast<size_t>(out_l * groups_per_row));
        std::memcpy(q.data(), L.qweight.data() + r0 * packed_in,
                    static_cast<size_t>(out_l * packed_in));
        std::memcpy(sc.data(), L.scales.data() + r0 * groups_per_row,
                    static_cast<size_t>(out_l * groups_per_row) * sizeof(float));
        L.qweight.swap(q);
        L.scales.swap(sc);
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
    if (L.has_f32()) {
        std::vector<float> w(static_cast<size_t>(out * in_l));
        for (int64_t o = 0; o < out; ++o)
            std::memcpy(w.data() + o * in_l, L.w.data() + o * in + c0,
                        static_cast<size_t>(in_l) * sizeof(float));
        L.w.swap(w);
    }
    // The bias adds to the post-all-reduce sum, so keep it on rank 0 only (else
    // it would be counted tp_size times).
    if (L.has_bias() && tp_rank != 0) std::fill(L.b.begin(), L.b.end(), 0.0f);
    if (L.quantized_int4) {
        GLM_CHECK(c0 % L.group_size == 0 && in_l % L.group_size == 0,
                  "row-parallel int4 shard must align to group_size=%lld (c0=%lld in_l=%lld)",
                  (long long)L.group_size, (long long)c0, (long long)in_l);
        const int64_t old_packed_in = (in + 1) / 2;
        const int64_t new_packed_in = (in_l + 1) / 2;
        const int64_t old_groups = static_cast<int64_t>(L.scales.size()) / out;
        const int64_t new_groups = in_l / L.group_size;
        const int64_t g0 = c0 / L.group_size;
        std::vector<uint8_t> q(static_cast<size_t>(out * new_packed_in), 0);
        std::vector<float> sc(static_cast<size_t>(out * new_groups));
        auto get_q = [](const uint8_t* row, int64_t col) {
            uint8_t byte = row[col >> 1];
            return static_cast<uint8_t>((col & 1) ? (byte >> 4) : (byte & 0x0F));
        };
        auto set_q = [](uint8_t* row, int64_t col, uint8_t v) {
            uint8_t& byte = row[col >> 1];
            if (col & 1) byte = static_cast<uint8_t>((byte & 0x0F) | (v << 4));
            else         byte = static_cast<uint8_t>((byte & 0xF0) | (v & 0x0F));
        };
        for (int64_t o = 0; o < out; ++o) {
            const uint8_t* src = L.qweight.data() + o * old_packed_in;
            uint8_t* dst = q.data() + o * new_packed_in;
            for (int64_t i = 0; i < in_l; ++i) set_q(dst, i, get_q(src, c0 + i));
            std::memcpy(sc.data() + o * new_groups,
                        L.scales.data() + o * old_groups + g0,
                        static_cast<size_t>(new_groups) * sizeof(float));
        }
        L.qweight.swap(q);
        L.scales.swap(sc);
    }
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
    const bool tied  = !st.has("lm_head.weight") && !st.has("lm_head.qweight");

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

        load_layer_body(st, c, i, L);
        if (L.indexer.valid()) ++loaded_indexers;
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

    mtp_blocks_.clear();
    if (last && c.num_nextn_predict_layers > 0) {
        for (int64_t m = 0; m < c.num_nextn_predict_layers; ++m) {
            int64_t i = c.num_hidden_layers + m;
            std::string p = "model.layers." + std::to_string(i) + ".";
            if (!st.has(p + "eh_proj.weight") && !st.has(p + "eh_proj.qweight")) continue;
            MTPBlock mtp;
            mtp.eh_proj = load_linear(st, p + "eh_proj");
            mtp.enorm = load_norm(st, p + "enorm.weight");
            mtp.hnorm = load_norm(st, p + "hnorm.weight");
            mtp.shared_head_norm = load_norm(st, p + "shared_head.norm.weight");
            load_layer_body(st, c, i, mtp.layer);
            shard_layer_tp(mtp.layer, tp_rank, tp);
            GLM_CHECK(mtp.eh_proj.in_features == 2 * c.hidden_size &&
                          mtp.eh_proj.out_features == c.hidden_size,
                      "MTP eh_proj shape [%lld,%lld] != [%lld,%lld]",
                      (long long)mtp.eh_proj.out_features,
                      (long long)mtp.eh_proj.in_features,
                      (long long)c.hidden_size, (long long)(2 * c.hidden_size));
            mtp_blocks_.push_back(std::move(mtp));
        }
        if (!mtp_blocks_.empty()) {
            GLM_INFO("loaded %zu MTP predictor block(s)", mtp_blocks_.size());
        }
    }
}

// Populate a Linear from a 2-D GGUF linear view (in=shape[0], out=product(rest)).
// The weight stays in-place as a mmap'd quant payload (qtype/qdata/row_bytes);
// the f32 `w` is empty so has_q() is true and the GPU forward dequantizes on the fly.
// F32/F16 tensors (indexer proj, MoE router) are small, so dequantize them to fp32
// `w` and run them through the cuBLAS fp32 path instead of the per-element qgemm.
static Linear gguf_lin(const GGUFLinearView& v) {
    Linear L;
    GLM_CHECK(v.valid(), "gguf_lin: invalid linear view");
    L.in_features = static_cast<int64_t>(v.in_features);
    L.out_features = static_cast<int64_t>(v.out_features);
    const uint32_t type = v.weight->tensor->type;
    if (type == 0 /*F32*/ || type == 1 /*F16*/) {
        L.w = gguf_dequantize_prefix(type, v.weight->data, v.weight->tensor->n_elements);
    } else {
        L.qtype = type;
        L.qdata = v.weight->data;
        L.row_bytes = static_cast<int64_t>(v.row_bytes);
    }
    return L;
}

// Dequantize a 1-D GGUF norm/bias tensor to f32.
static RMSNormW gguf_norm_w(const GGUFWeightView* v) {
    RMSNormW r;
    GLM_CHECK(v && v->data, "gguf_norm_w: missing norm tensor");
    r.dim = static_cast<int64_t>(v->tensor->n_elements);
    r.w = gguf_dequantize_prefix(v->tensor->type, v->data, v->tensor->n_elements);
    return r;
}

// Dequantize a 1-D GGUF bias tensor to f32.
static std::vector<float> gguf_bias(const GGUFWeightView* v) {
    GLM_CHECK(v && v->data, "gguf_bias: missing bias tensor");
    return gguf_dequantize_prefix(v->tensor->type, v->data, v->tensor->n_elements);
}

// Merge the split GGUF kv_b projections (3-D per-head k_b/v_b) into a single
// fp32 kv_b_proj [H*(nope+vd), kv_lora] matching the safetensors merged layout
// the GPU forward expects. k_b: ne={qk_nope, kv_lora, heads}; v_b: ne={kv_lora, v_head, heads}.
// kv_b_proj row (h*(nope+vd)+i) for i<nope <- k_b[h][i][j]; row (h*(nope+vd)+nope+j) <- v_b[h][j][i].
static Linear gguf_merge_kv_b(const GGUFWeightView* k_b, const GGUFWeightView* v_b,
                              int64_t heads, int64_t nope, int64_t vd, int64_t kv_lora) {
    Linear L;
    GLM_CHECK(k_b && v_b, "gguf_merge_kv_b: missing k_b/v_b");
    const int64_t out = heads * (nope + vd);
    L.in_features = kv_lora;
    L.out_features = out;
    L.w.resize(static_cast<size_t>(out * kv_lora), 0.0f);
    std::vector<float> kf = gguf_dequantize_prefix(k_b->tensor->type, k_b->data, k_b->tensor->n_elements);
    std::vector<float> vf = gguf_dequantize_prefix(v_b->tensor->type, v_b->data, v_b->tensor->n_elements);
    // k_b: ne={nope, kv_lora, heads} -> kf[h*kv_lora*nope + j*nope + i] = W_k[h][j][i]
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t i = 0; i < nope; ++i)
            for (int64_t j = 0; j < kv_lora; ++j)
                L.w[static_cast<size_t>((h * (nope + vd) + i) * kv_lora + j)] =
                    kf[static_cast<size_t>(h * kv_lora * nope + j * nope + i)];
    // v_b: ne={kv_lora, v_head, heads} -> vf[h*v_head*kv_lora + j*kv_lora + i] = W_v[h][j][i]
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t j = 0; j < vd; ++j)
            for (int64_t i = 0; i < kv_lora; ++i)
                L.w[static_cast<size_t>((h * (nope + vd) + nope + j) * kv_lora + i)] =
                    vf[static_cast<size_t>(h * vd * kv_lora + j * kv_lora + i)];
    return L;
}

// Slice a quant Linear's output rows [r0, r0+out_l) (column-parallel shard).
// qdata rows are contiguous (row_bytes each), so just advance the pointer.
static void shard_linear_columns_q(Linear& L, int64_t r0, int64_t out_l) {
    if (r0 == 0 && out_l == L.out_features) return;
    L.qdata = L.qdata + r0 * L.row_bytes;
    L.out_features = out_l;
    if (!L.b.empty()) L.b.erase(L.b.begin(), L.b.begin() + static_cast<ptrdiff_t>(r0));
}

// Slice a quant Linear's input columns [c0, c0+in_l) (row-parallel shard).
// Quant blocks span the input dimension; requires in_l % block_elems == 0.
static void shard_linear_rows_q(Linear& L, int64_t c0, int64_t in_l) {
    if (c0 == 0 && in_l == L.in_features) return;
    const uint64_t be = gguf_type_block_elements(L.qtype);
    const uint64_t bb = gguf_type_block_bytes(L.qtype);
    GLM_CHECK(c0 % be == 0 && in_l % be == 0,
              "row-parallel GGUF shard must align to block elems=%llu (c0=%lld in_l=%lld)",
              (unsigned long long)be, (long long)c0, (long long)in_l);
    const int64_t new_row_bytes = (in_l / static_cast<int64_t>(be)) * static_cast<int64_t>(bb);
    const int64_t skip = (c0 / static_cast<int64_t>(be)) * static_cast<int64_t>(bb);
    // Zero-copy: keep the mmap'd payload in place as a strided view; the GPU
    // upload repacks it into a contiguous device buffer with one cudaMemcpy2D
    // (no ~GBs of host-side slice buffers, no 4 KiB random page faults).
    L.qstride = L.qsrc_stride();
    L.qdata = L.qdata + skip;
    L.row_bytes = new_row_bytes;
    L.in_features = in_l;
}

// Tensor-parallel shard of one GGUF-loaded block's quant/f32 linears.
static void shard_layer_tp_gguf(Layer& L, int tp_rank, int tp_size) {
    if (tp_size <= 1) return;
    const int64_t H = L.q_b_proj.out_features / tp_size * tp_size;  // (unused, kept for parity)
    (void)H;
    auto col = [&](Linear& lin) {
        if (lin.has_q()) {
            int64_t out = lin.out_features, out_l = out / tp_size;
            shard_linear_columns_q(lin, static_cast<int64_t>(tp_rank) * out_l, out_l);
        } else if (lin.has_f32()) {
            shard_linear_columns(lin, tp_rank, tp_size);
        }
    };
    auto row = [&](Linear& lin) {
        if (lin.has_q()) {
            int64_t in = lin.in_features, in_l = in / tp_size;
            shard_linear_rows_q(lin, static_cast<int64_t>(tp_rank) * in_l, in_l);
        } else if (lin.has_f32()) {
            shard_linear_rows(lin, tp_rank, tp_size);
        }
    };
    col(L.q_b_proj);
    col(L.kv_b_proj);   // fp32 merged; shard_linear_columns handles .w
    row(L.o_proj);
    if (L.is_dense) {
        col(L.dense_mlp.gate_proj); col(L.dense_mlp.up_proj); row(L.dense_mlp.down_proj);
    } else {
        for (Expert& E : L.moe.experts) {
            col(E.gate_proj); col(E.up_proj); row(E.down_proj);
        }
        if (L.moe.has_shared) {
            col(L.moe.shared.gate_proj); col(L.moe.shared.up_proj); row(L.moe.shared.down_proj);
        }
    }
}

void GLM52Model::load_gguf(const std::string& gguf_path, int64_t max_layers,
                           bool touch_payloads) {
    Timer timer;
    const auto& c = cfg_;
    const int tp = std::max(1, dist_.tp_size);

    GLM_CHECK(c.num_attention_heads % tp == 0,
              "num_attention_heads %lld not divisible by tp_size %d",
              (long long)c.num_attention_heads, tp);
    tp_heads_ = c.num_attention_heads / tp;

    int64_t total_layers = (max_layers > 0) ? std::min(max_layers, c.num_hidden_layers)
                                            : c.num_hidden_layers;
    LayerRange range = partition_layers(total_layers, dist_.pp_stage(), dist_.pp_size);
    layer_begin_ = range.begin;
    layers_.clear();
    mtp_blocks_.clear();

    gguf_weights_ = std::make_unique<GLM52GGUFWeights>();
    gguf_weights_->load(gguf_path, cfg_, max_layers, touch_payloads);

    // Warm the page cache for the loaded payload ranges before any per-tensor
    // work or the GPU upload touches the mmap: 4 KiB random faults on a network
    // filesystem are ~100x slower than these large sequential reads. Ranks
    // stripe the chunks (page cache is node-shared); the barrier keeps any rank
    // from racing ahead into cold pages.
    if (std::getenv("GLMSERVE_NO_PREFAULT") == nullptr) {
        Timer pf;
        const uint64_t bytes = gguf_weights_->prefault(dist_.rank, dist_.world_size);
        GLM_INFO("prefault (rank %d/%d): read %.2f GiB into page cache in %.1f s",
                 dist_.rank + 1, dist_.world_size, bytes / 1073741824.0, pf.ms() / 1000.0);
        if (comm_) comm_->barrier();
    }

    const GGUFGLM52Layout& layout = gguf_weights_->layout();
    const int64_t H = c.hidden_size;
    const int64_t heads = c.num_attention_heads;
    const int64_t nope = c.qk_nope_head_dim, rope = c.qk_rope_head_dim;
    const int64_t vd = c.v_head_dim, kv_lora = c.kv_lora_rank;

    const bool first = dist_.is_first_stage();
    const bool last = dist_.is_last_stage();
    const int tp_rank = dist_.tp_rank();

    // Embeddings (first stage) and final norm + lm_head (last stage).
    if (first) {
        GGUFLinearView ev = gguf_weights_->linear("model.embed_tokens.weight");
        GLM_CHECK(ev.valid(), "GGUF missing model.embed_tokens.weight");
        embed_lin_ = gguf_lin(ev);
        // Also dequantize a small fp32 copy for the CPU embed() path / tests.
        if (ev.weight->tensor->type == 0 || ev.weight->tensor->type == 1) {
            embed_tokens_ = gguf_dequantize_prefix(ev.weight->tensor->type, ev.weight->data,
                                                   ev.weight->tensor->n_elements);
        }
    }
    if (last) {
        const GGUFWeightView* nv = gguf_weights_->role("model.norm.weight");
        GLM_CHECK(nv, "GGUF missing model.norm.weight");
        final_norm_ = gguf_norm_w(nv);
        GGUFLinearView lv = gguf_weights_->linear("lm_head.weight");
        GLM_CHECK(lv.valid(), "GGUF missing lm_head.weight");
        lm_head_ = gguf_lin(lv);
        // Column-parallel lm_head shard under TP: each rank computes logits for
        // vocab rows [rank*V/tp, (rank+1)*V/tp); the GPU forward zero-fills the
        // full logits buffer, writes its slice, and all-reduces. Saves ~0.7 GiB
        // of replicated Q6_K weight per rank and cuts the per-token lm_head
        // read 8x at the cost of one extra small all-reduce.
        if (lm_head_.has_q() && tp > 1 && cfg_.vocab_size % tp == 0) {
            const int64_t v_l = cfg_.vocab_size / tp;
            lm_head_shard_off_ = static_cast<int64_t>(tp_rank) * v_l;
            shard_linear_columns_q(lm_head_, lm_head_shard_off_, v_l);
        }
    }

    layers_.resize(range.end - range.begin);
    for (int64_t li = 0; li < static_cast<int64_t>(layers_.size()); ++li) {
        const int64_t gi = layer_begin_ + li;
        Layer& L = layers_[li];
        const std::string hf = "model.layers." + std::to_string(gi) + ".";
        L.is_dense = c.is_dense_layer(gi);
        L.input_norm = gguf_norm_w(gguf_weights_->role(hf + "input_layernorm.weight"));
        L.post_attn_norm = gguf_norm_w(gguf_weights_->role(hf + "post_attention_layernorm.weight"));
        L.q_a_proj = gguf_lin(gguf_weights_->linear(hf + "self_attn.q_a_proj.weight"));
        L.q_a_norm = gguf_norm_w(gguf_weights_->role(hf + "self_attn.q_a_layernorm.weight"));
        L.q_b_proj = gguf_lin(gguf_weights_->linear(hf + "self_attn.q_b_proj.weight"));
        L.kv_a_proj = gguf_lin(gguf_weights_->linear(hf + "self_attn.kv_a_proj_with_mqa.weight"));
        L.kv_a_norm = gguf_norm_w(gguf_weights_->role(hf + "self_attn.kv_a_layernorm.weight"));
        L.kv_b_proj = gguf_merge_kv_b(gguf_weights_->role(hf + "self_attn.kv_b_proj.weight[k]"),
                                      gguf_weights_->role(hf + "self_attn.kv_b_proj.weight[v]"),
                                      heads, nope, vd, kv_lora);
        L.o_proj = gguf_lin(gguf_weights_->linear(hf + "self_attn.o_proj.weight"));
        if (gguf_weights_->role(hf + "self_attn.indexer.wq_b.weight")) {
            L.indexer.wq_b = gguf_lin(gguf_weights_->linear(hf + "self_attn.indexer.wq_b.weight"));
            L.indexer.wk = gguf_lin(gguf_weights_->linear(hf + "self_attn.indexer.wk.weight"));
            L.indexer.weights_proj = gguf_lin(gguf_weights_->linear(hf + "self_attn.indexer.weights_proj.weight"));
            L.indexer.k_norm = gguf_norm_w(gguf_weights_->role(hf + "self_attn.indexer.k_norm.weight"));
            const GGUFWeightView* kb = gguf_weights_->role(hf + "self_attn.indexer.k_norm.bias");
            if (kb) L.indexer.k_norm_bias = gguf_bias(kb);
        }
        if (L.is_dense) {
            L.dense_mlp.gate_proj = gguf_lin(gguf_weights_->linear(hf + "mlp.gate_proj.weight"));
            L.dense_mlp.up_proj = gguf_lin(gguf_weights_->linear(hf + "mlp.up_proj.weight"));
            L.dense_mlp.down_proj = gguf_lin(gguf_weights_->linear(hf + "mlp.down_proj.weight"));
        } else {
            L.moe.router = gguf_lin(gguf_weights_->linear(hf + "mlp.gate.weight"));
            const GGUFWeightView* eb = gguf_weights_->role(hf + "mlp.gate.e_score_correction_bias");
            if (eb) L.moe.e_bias = gguf_bias(eb);
            const int64_t E = c.n_routed_experts, moei = c.moe_intermediate_size;
            GGUFLinearView gv = gguf_weights_->linear(hf + "mlp.experts.*.gate_proj.weight");
            GGUFLinearView uv = gguf_weights_->linear(hf + "mlp.experts.*.up_proj.weight");
            GGUFLinearView dv = gguf_weights_->linear(hf + "mlp.experts.*.down_proj.weight");
            GLM_CHECK(gv.valid() && uv.valid() && dv.valid(), "GGUF missing expert tensors for layer %lld", (long long)gi);
            GLM_CHECK(gv.out_features == moei * E && gv.in_features == H, "GGUF expert gate shape mismatch");
            L.moe.experts.resize(static_cast<size_t>(E));
            for (int64_t e = 0; e < E; ++e) {
                Expert& E0 = L.moe.experts[static_cast<size_t>(e)];
                E0.gate_proj = gguf_lin(gv); E0.gate_proj.qdata += e * moei * gv.row_bytes; E0.gate_proj.out_features = moei;
                E0.up_proj = gguf_lin(uv);   E0.up_proj.qdata   += e * moei * uv.row_bytes; E0.up_proj.out_features = moei;
                E0.down_proj = gguf_lin(dv); E0.down_proj.qdata += e * H * dv.row_bytes;  E0.down_proj.out_features = H;
            }
            const GGUFWeightView* sg = gguf_weights_->role(hf + "mlp.shared_experts.gate_proj.weight");
            if (sg) {
                L.moe.shared.gate_proj = gguf_lin(gguf_weights_->linear(hf + "mlp.shared_experts.gate_proj.weight"));
                L.moe.shared.up_proj = gguf_lin(gguf_weights_->linear(hf + "mlp.shared_experts.up_proj.weight"));
                L.moe.shared.down_proj = gguf_lin(gguf_weights_->linear(hf + "mlp.shared_experts.down_proj.weight"));
                L.moe.has_shared = true;
            }
        }
        shard_layer_tp_gguf(L, tp_rank, tp);
    }

    // MTP/NextN block (layer 78 = num_hidden_layers). The GGUF stores its
    // attention/MLP as blk.78.* (trunk-style) and its nextn extras as
    // blk.78.nextn.*. Only load on the last pipeline stage; skip if absent.
    if (last && c.num_nextn_predict_layers > 0 &&
        gguf_weights_->role("model.layers.78.eh_proj.weight")) {
        MTPBlock m;
        const std::string p = "model.layers.78.";
        m.eh_proj = gguf_lin(gguf_weights_->linear(p + "eh_proj.weight"));
        m.enorm = gguf_norm_w(gguf_weights_->role(p + "enorm.weight"));
        m.hnorm = gguf_norm_w(gguf_weights_->role(p + "hnorm.weight"));
        m.shared_head_norm = gguf_norm_w(gguf_weights_->role(p + "shared_head.norm.weight"));
        Layer L;
        L.is_dense = c.is_dense_layer(78);
        L.input_norm = gguf_norm_w(gguf_weights_->role(p + "input_layernorm.weight"));
        L.post_attn_norm = gguf_norm_w(gguf_weights_->role(p + "post_attention_layernorm.weight"));
        L.q_a_proj = gguf_lin(gguf_weights_->linear(p + "self_attn.q_a_proj.weight"));
        L.q_a_norm = gguf_norm_w(gguf_weights_->role(p + "self_attn.q_a_layernorm.weight"));
        L.q_b_proj = gguf_lin(gguf_weights_->linear(p + "self_attn.q_b_proj.weight"));
        L.kv_a_proj = gguf_lin(gguf_weights_->linear(p + "self_attn.kv_a_proj_with_mqa.weight"));
        L.kv_a_norm = gguf_norm_w(gguf_weights_->role(p + "self_attn.kv_a_layernorm.weight"));
        L.kv_b_proj = gguf_merge_kv_b(gguf_weights_->role(p + "self_attn.kv_b_proj.weight[k]"),
                                      gguf_weights_->role(p + "self_attn.kv_b_proj.weight[v]"),
                                      heads, nope, vd, kv_lora);
        L.o_proj = gguf_lin(gguf_weights_->linear(p + "self_attn.o_proj.weight"));
        if (L.is_dense) {
            L.dense_mlp.gate_proj = gguf_lin(gguf_weights_->linear(p + "mlp.gate_proj.weight"));
            L.dense_mlp.up_proj = gguf_lin(gguf_weights_->linear(p + "mlp.up_proj.weight"));
            L.dense_mlp.down_proj = gguf_lin(gguf_weights_->linear(p + "mlp.down_proj.weight"));
        } else {
            L.moe.router = gguf_lin(gguf_weights_->linear(p + "mlp.gate.weight"));
            const GGUFWeightView* eb = gguf_weights_->role(p + "mlp.gate.e_score_correction_bias");
            if (eb) L.moe.e_bias = gguf_bias(eb);
            const int64_t E = c.n_routed_experts, moei = c.moe_intermediate_size;
            GGUFLinearView gv = gguf_weights_->linear(p + "mlp.experts.*.gate_proj.weight");
            GGUFLinearView uv = gguf_weights_->linear(p + "mlp.experts.*.up_proj.weight");
            GGUFLinearView dv = gguf_weights_->linear(p + "mlp.experts.*.down_proj.weight");
            if (gv.valid() && uv.valid() && dv.valid()) {
                L.moe.experts.resize(static_cast<size_t>(E));
                for (int64_t e = 0; e < E; ++e) {
                    Expert& E0 = L.moe.experts[static_cast<size_t>(e)];
                    E0.gate_proj = gguf_lin(gv); E0.gate_proj.qdata += e * moei * gv.row_bytes; E0.gate_proj.out_features = moei;
                    E0.up_proj = gguf_lin(uv);   E0.up_proj.qdata   += e * moei * uv.row_bytes; E0.up_proj.out_features = moei;
                    E0.down_proj = gguf_lin(dv); E0.down_proj.qdata += e * H * dv.row_bytes;  E0.down_proj.out_features = H;
                }
                const GGUFWeightView* sg = gguf_weights_->role(p + "mlp.shared_experts.gate_proj.weight");
                if (sg) {
                    L.moe.shared.gate_proj = gguf_lin(gguf_weights_->linear(p + "mlp.shared_experts.gate_proj.weight"));
                    L.moe.shared.up_proj = gguf_lin(gguf_weights_->linear(p + "mlp.shared_experts.up_proj.weight"));
                    L.moe.shared.down_proj = gguf_lin(gguf_weights_->linear(p + "mlp.shared_experts.down_proj.weight"));
                    L.moe.has_shared = true;
                }
            }
        }
        shard_layer_tp_gguf(L, tp_rank, tp);
        m.layer = std::move(L);
        mtp_blocks_.push_back(std::move(m));
    }

    GLM_INFO("GGUF quant weights loaded: views=%zu/%zu shards=%zu tensors=%zu mapped=%.2f GiB "
             "quantized=%zu tensors %.2f GiB in %.1f s",
             gguf_weights_->views().size(), layout.modules.size(),
             gguf_weights_->gguf().shards().size(), gguf_weights_->gguf().tensors().size(),
             gguf_weights_->mapped_payload_bytes() / (1024.0 * 1024.0 * 1024.0),
             layout.quantized_tensors,
             layout.quantized_tensor_bytes / (1024.0 * 1024.0 * 1024.0),
             timer.ms() / 1000.0);
    if (touch_payloads) {
        GLM_INFO("GGUF payload touch checksum=%016llx",
                 (unsigned long long)gguf_weights_->payload_checksum());
    }
}

bool GLM52Model::gguf_ready() const {
    return gguf_weights_ && gguf_weights_->ready();
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
// DSA indexer: for ctx <= index_topk the sparse mask is exact dense. Past that,
// full-indexer layers run the learned top-k selector and refresh the shared
// IndexShare mask; shared layers reuse that mask against their own K/V cache.
// ---------------------------------------------------------------------------
void GLM52Model::attention(const Layer& L, int64_t layer_idx, const float* normed,
                           float* attn_out, int64_t n_tokens, int64_t start_pos,
                           SequenceKV& kv,
                           std::vector<std::vector<int64_t>>* shared_dsa_indices) {
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

    const int64_t ctx_after = start_pos + n_tokens;
    const bool sparse_dsa = c.use_dsa && ctx_after > c.index_topk;
    const bool learned_dsa = sparse_dsa && L.indexer.valid();
    const bool shared_dsa = sparse_dsa && !learned_dsa && shared_dsa_indices &&
                            static_cast<int64_t>(shared_dsa_indices->size()) == n_tokens;
    std::vector<float> index_q;
    std::vector<float> index_w;
    std::vector<std::vector<int64_t>> selected_keys;
    if (learned_dsa) {
        GLM_CHECK(kv.cache && kv.cache->indexer_dim() == c.index_head_dim,
                  "learned DSA requires KVCache indexer_dim=%lld, got %lld",
                  (long long)c.index_head_dim,
                  (long long)(kv.cache ? kv.cache->indexer_dim() : 0));
        check_dsa_indexer_shapes(L.indexer, c);
        index_q.resize(n_tokens * c.index_n_heads * c.index_head_dim);
        linear_forward(L.indexer.wq_b, qa.data(), index_q.data(), n_tokens);
        index_w.resize(n_tokens * c.index_n_heads);
        linear_forward(L.indexer.weights_proj, normed, index_w.data(), n_tokens);
        float inv_h = 1.0f / std::sqrt(static_cast<float>(c.index_n_heads));
        for (float& v : index_w) v *= inv_h;
    }

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

    std::vector<float> index_k;
    if (learned_dsa) {
        index_k.resize(n_tokens * c.index_head_dim);
        linear_forward(L.indexer.wk, normed, index_k.data(), n_tokens);
        layernorm_rows(index_k.data(), L.indexer.k_norm.w.data(),
                       L.indexer.k_norm_bias.empty() ? nullptr : L.indexer.k_norm_bias.data(),
                       index_k.data(), n_tokens, c.index_head_dim, 1e-6f);
    }

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
        if (learned_dsa) {
            float* iq = index_q.data() + t * c.index_n_heads * c.index_head_dim;
            for (int64_t ih = 0; ih < c.index_n_heads; ++ih)
                rope_inplace(iq + ih * c.index_head_dim, rope, pos, c.rope_theta, false);

            float* ik = index_k.data() + t * c.index_head_dim;
            rope_inplace(ik, rope, pos, c.rope_theta, false);
            std::memcpy(kv.index_slot(layer_idx, pos), ik,
                        c.index_head_dim * sizeof(float));
        }
    }

    if (learned_dsa) {
        selected_keys.resize(static_cast<size_t>(n_tokens));
        const float index_scale = 1.0f / std::sqrt(static_cast<float>(c.index_head_dim));
        parallel_for(n_tokens, [&](int64_t t) {
            int64_t pos = start_pos + t;
            int64_t topk = std::min<int64_t>(c.index_topk, pos + 1);
            std::vector<float> scores(pos + 1, 0.0f);
            const float* qt = index_q.data() + t * c.index_n_heads * c.index_head_dim;
            const float* wt = index_w.data() + t * c.index_n_heads;
            for (int64_t j = 0; j <= pos; ++j) {
                const float* kj = kv.index_slot(layer_idx, j);
                float total = 0.0f;
                for (int64_t ih = 0; ih < c.index_n_heads; ++ih) {
                    const float* qh = qt + ih * c.index_head_dim;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < c.index_head_dim; ++d) dot += qh[d] * kj[d];
                    total += wt[ih] * std::max(0.0f, dot * index_scale);
                }
                scores[j] = total;
            }

            std::vector<int64_t> ids(pos + 1);
            std::iota(ids.begin(), ids.end(), 0);
            std::partial_sort(ids.begin(), ids.begin() + topk, ids.end(),
                              [&](int64_t a, int64_t b) { return scores[a] > scores[b]; });
            ids.resize(static_cast<size_t>(topk));
            std::sort(ids.begin(), ids.end());
            selected_keys[static_cast<size_t>(t)] = std::move(ids);
        });
    }

    if (learned_dsa && shared_dsa_indices) *shared_dsa_indices = selected_keys;

    const std::vector<std::vector<int64_t>>* sparse_keys = nullptr;
    if (learned_dsa) sparse_keys = &selected_keys;
    else if (shared_dsa) sparse_keys = shared_dsa_indices;

    // Standard causal attention with head_dim = qk (K) / vd (V) inside an hc slot.
    // In DSA mode, the key loop is restricted to learned or IndexShare positions.
    parallel_for(n_tokens * H, [&](int64_t idx) {
        int64_t t = idx / H, h = idx % H;
        int64_t pos = start_pos + t;
        const float* qh = q.data() + (t * H + h) * qk;
        const std::vector<int64_t>* keys =
            sparse_keys ? &(*sparse_keys)[static_cast<size_t>(t)] : nullptr;
        int64_t n_keys = sparse_keys ? static_cast<int64_t>(keys->size()) : pos + 1;
        std::vector<float> scores(n_keys);
        float maxv = -1e30f;
        for (int64_t kk = 0; kk < n_keys; ++kk) {
            int64_t j = sparse_keys ? (*keys)[static_cast<size_t>(kk)] : kk;
            const float* kj = kv.k_slot(layer_idx, j) + h * hc;
            float dot = 0.0f;
            for (int64_t d = 0; d < qk; ++d) dot += qh[d] * kj[d];
            dot *= scale;
            scores[kk] = dot;
            maxv = std::max(maxv, dot);
        }
        float sum = 0.0f;
        for (int64_t kk = 0; kk < n_keys; ++kk) {
            scores[kk] = std::exp(scores[kk] - maxv);
            sum += scores[kk];
        }
        float inv = 1.0f / sum;
        float* out = attn_out + (t * H + h) * vd;   // attn_out is [n, H, v_head_dim]
        for (int64_t d = 0; d < vd; ++d) out[d] = 0.0f;
        for (int64_t kk = 0; kk < n_keys; ++kk) {
            int64_t j = sparse_keys ? (*keys)[static_cast<size_t>(kk)] : kk;
            const float* vj = kv.v_slot(layer_idx, j) + h * hc;
            float wgt = scores[kk] * inv;
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

void GLM52Model::run_layer_block(const Layer& L, int64_t kv_layer_idx, float* hidden,
                                 int64_t n_tokens, int64_t start_pos, SequenceKV& kv,
                                 std::vector<std::vector<int64_t>>* shared_dsa_indices) {
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
    attention(L, kv_layer_idx, normed.data(), attn.data(), n_tokens, start_pos, kv,
              shared_dsa_indices);
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

void GLM52Model::run_layer(int64_t layer_idx, float* hidden, int64_t n_tokens,
                           int64_t start_pos, SequenceKV& kv,
                           std::vector<std::vector<int64_t>>* shared_dsa_indices) {
    run_layer_block(layers_[layer_idx], layer_idx, hidden, n_tokens, start_pos, kv,
                    shared_dsa_indices);
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
    std::vector<std::vector<int64_t>> shared_dsa_indices;
    const int64_t ctx_after = start_pos + n;
    const bool sparse_dsa = cfg_.use_dsa && ctx_after > cfg_.index_topk;
    auto shared_dsa_layer = [&](int64_t global_layer) {
        return cfg_.use_dsa &&
               global_layer >= 0 &&
               global_layer < static_cast<int64_t>(cfg_.indexer_types.size()) &&
               cfg_.indexer_types[static_cast<size_t>(global_layer)] != "full";
    };
    if (sparse_dsa && !dist_.is_first_stage() && shared_dsa_layer(layer_begin_)) {
        GLM_CHECK(comm_, "DSA mask receive requires a communicator");
        std::vector<int> flat(static_cast<size_t>(n * cfg_.index_topk), -1);
        comm_->pipeline_recv_prev_int(flat.data(), n * cfg_.index_topk);
        shared_dsa_indices.resize(static_cast<size_t>(n));
        for (int64_t t = 0; t < n; ++t) {
            auto& row = shared_dsa_indices[static_cast<size_t>(t)];
            for (int64_t k = 0; k < cfg_.index_topk; ++k) {
                int v = flat[static_cast<size_t>(t * cfg_.index_topk + k)];
                if (v >= 0) row.push_back(v);
            }
        }
    }
    for (int64_t l = 0; l < num_layers(); ++l)
        run_layer(l, hidden.data(), n, start_pos, kv, &shared_dsa_indices);

    kv.length = start_pos + n;

    // Pipeline output: a non-final stage forwards the hidden state downstream and
    // produces no logits; only the last stage runs final-norm + lm_head.
    if (!dist_.is_last_stage()) {
        GLM_CHECK(comm_, "non-last pipeline stage requires a communicator");
        comm_->pipeline_send_next(hidden.data(), n * H);
        int64_t next_layer = layer_begin_ + num_layers();
        if (sparse_dsa && shared_dsa_layer(next_layer) &&
            static_cast<int64_t>(shared_dsa_indices.size()) == n) {
            std::vector<int> flat(static_cast<size_t>(n * cfg_.index_topk), -1);
            for (int64_t t = 0; t < n; ++t) {
                const auto& row = shared_dsa_indices[static_cast<size_t>(t)];
                int64_t m = std::min<int64_t>(cfg_.index_topk, static_cast<int64_t>(row.size()));
                for (int64_t k = 0; k < m; ++k)
                    flat[static_cast<size_t>(t * cfg_.index_topk + k)] = static_cast<int>(row[static_cast<size_t>(k)]);
            }
            comm_->pipeline_send_next_int(flat.data(), n * cfg_.index_topk);
        }
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

std::vector<float> GLM52Model::mtp_draft_logits(const std::vector<int>& context_tokens,
                                                const std::vector<int>& draft_tokens) {
    GLM_CHECK(!context_tokens.empty(), "mtp requires nonempty context tokens");
    GLM_CHECK(!draft_tokens.empty(), "mtp requires nonempty draft tokens");
    GLM_CHECK(dist_.world_size == 1 && dist_.tp_size == 1 && dist_.pp_size == 1,
              "mtp_draft_logits currently supports the single-rank CPU path");
    GLM_CHECK(mtp_ready(), "MTP weights are not loaded");

    const int64_t H = cfg_.hidden_size;
    const int64_t block = 16;
    const int64_t base_blocks = (static_cast<int64_t>(context_tokens.size()) + block - 1) / block + 4;
    KVCache base_cache(num_layers(), local_kv_heads(), cfg_.kv_cache_head_dim(), block,
                       base_blocks, cfg_.use_dsa ? cfg_.index_head_dim : 0);
    SequenceKV base_kv = base_cache.make_sequence(9001);

    std::vector<float> hidden(context_tokens.size() * static_cast<size_t>(H));
    for (size_t t = 0; t < context_tokens.size(); ++t)
        embed(context_tokens[t], hidden.data() + static_cast<int64_t>(t) * H);
    base_kv.reserve(static_cast<int64_t>(context_tokens.size()));
    std::vector<std::vector<int64_t>> shared_dsa_indices;
    for (int64_t l = 0; l < num_layers(); ++l)
        run_layer(l, hidden.data(), static_cast<int64_t>(context_tokens.size()), 0,
                  base_kv, &shared_dsa_indices);

    std::vector<float> prev(H);
    std::memcpy(prev.data(), hidden.data() + (context_tokens.size() - 1) * H,
                H * sizeof(float));
    base_kv.release();

    const MTPBlock& mtp = mtp_blocks_[0];
    const int64_t steps = static_cast<int64_t>(draft_tokens.size());
    KVCache mtp_cache(1, local_kv_heads(), cfg_.kv_cache_head_dim(), block,
                      (steps + block - 1) / block + 2,
                      cfg_.use_dsa ? cfg_.index_head_dim : 0);
    SequenceKV mtp_kv = mtp_cache.make_sequence(9002);
    std::vector<float> logits(static_cast<size_t>(steps) * cfg_.vocab_size);
    std::vector<std::vector<int64_t>> mtp_shared_dsa_indices;

    for (int64_t s = 0; s < steps; ++s) {
        std::vector<float> emb(H), enormed(H), hnormed(H), fused_in(2 * H), cur(H);
        embed(draft_tokens[static_cast<size_t>(s)], emb.data());
        std::memcpy(enormed.data(), emb.data(), H * sizeof(float));
        std::memcpy(hnormed.data(), prev.data(), H * sizeof(float));
        rmsnorm_vec(enormed.data(), mtp.enorm.w.data(), H, static_cast<float>(cfg_.rms_norm_eps));
        rmsnorm_vec(hnormed.data(), mtp.hnorm.w.data(), H, static_cast<float>(cfg_.rms_norm_eps));
        std::memcpy(fused_in.data(), enormed.data(), H * sizeof(float));
        std::memcpy(fused_in.data() + H, hnormed.data(), H * sizeof(float));
        linear_forward(mtp.eh_proj, fused_in.data(), cur.data(), 1);

        mtp_kv.reserve(1);
        run_layer_block(mtp.layer, 0, cur.data(), 1, s, mtp_kv, &mtp_shared_dsa_indices);
        mtp_kv.length = s + 1;

        std::vector<float> head(H);
        std::memcpy(head.data(), cur.data(), H * sizeof(float));
        rmsnorm_vec(head.data(), mtp.shared_head_norm.w.data(), H,
                    static_cast<float>(cfg_.rms_norm_eps));
        linear_forward(lm_head_, head.data(), logits.data() + s * cfg_.vocab_size, 1);
        prev.swap(cur);
    }

    mtp_kv.release();
    return logits;
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
std::vector<float> GLM52Model::mtp_draft_logits_gpu(const std::vector<int>&,
                                                    const std::vector<int>&) {
    GLM_CHECK(false, "mtp_draft_logits_gpu: built without CUDA (rebuild with GPU=1)");
    return {};
}
#endif

}  // namespace glmserve
