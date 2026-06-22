#include "config.hpp"
#include "common.hpp"
#include "json.hpp"

#include <fstream>
#include <sstream>

namespace glmserve {

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GLM_CHECK(f.good(), "cannot open config file: %s", path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLM52Config load_config(const std::string& path) {
    // Accept either a directory (append config.json) or a direct file path.
    std::string cfg_path = path;
    if (cfg_path.size() < 5 || cfg_path.substr(cfg_path.size() - 5) != ".json") {
        if (!cfg_path.empty() && cfg_path.back() != '/') cfg_path += '/';
        cfg_path += "config.json";
    }

    GLM52Config c;
    std::string text = read_file(cfg_path);
    auto root = json::parse(text);
    GLM_CHECK(root && root->is_object(), "config.json is not a JSON object");

    auto& r = *root;

    // Some HF configs nest scalars; we read flat keys with sensible fallbacks.
    c.vocab_size          = r.get_int("vocab_size", c.vocab_size);
    c.hidden_size         = r.get_int("hidden_size", c.hidden_size);
    c.num_hidden_layers   = r.get_int("num_hidden_layers", c.num_hidden_layers);
    c.num_attention_heads = r.get_int("num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = r.get_int("num_key_value_heads", c.num_attention_heads);
    c.intermediate_size   = r.get_int("intermediate_size", c.intermediate_size);
    c.max_position_embeddings =
        r.get_int("max_position_embeddings", c.max_position_embeddings);

    // head_dim: explicit if present, else hidden/heads.
    if (r.has("head_dim"))
        c.head_dim = r.get_int("head_dim", c.head_dim);
    else
        c.head_dim = c.hidden_size / c.num_attention_heads;

    // --- MLA (DeepSeek-style latent attention) ---
    c.q_lora_rank      = r.get_int("q_lora_rank", c.q_lora_rank);
    c.kv_lora_rank     = r.get_int("kv_lora_rank", c.kv_lora_rank);
    c.qk_nope_head_dim = r.get_int("qk_nope_head_dim", c.qk_nope_head_dim);
    c.qk_rope_head_dim = r.get_int("qk_rope_head_dim", c.qk_rope_head_dim);
    c.qk_head_dim      = r.get_int("qk_head_dim", c.qk_nope_head_dim + c.qk_rope_head_dim);
    c.v_head_dim       = r.get_int("v_head_dim", c.v_head_dim);
    c.attention_bias   = r.get_bool("attention_bias", c.attention_bias);
    c.rope_interleave  = r.get_bool("rope_interleave", c.rope_interleave);

    // MoE
    c.n_routed_experts      = r.get_int("n_routed_experts", c.n_routed_experts);
    c.num_experts_per_tok   = r.get_int("num_experts_per_tok", c.num_experts_per_tok);
    c.n_shared_experts      = r.get_int("n_shared_experts", c.n_shared_experts);
    c.moe_intermediate_size = r.get_int("moe_intermediate_size", c.moe_intermediate_size);
    c.first_k_dense_replace = r.get_int("first_k_dense_replace", c.first_k_dense_replace);
    c.n_group               = r.get_int("n_group", c.n_group);
    c.topk_group            = r.get_int("topk_group", c.topk_group);
    c.routed_scaling_factor = r.get_double("routed_scaling_factor", c.routed_scaling_factor);
    c.norm_topk_prob        = r.get_bool("norm_topk_prob", c.norm_topk_prob);
    c.scoring_func          = r.get_string("scoring_func", c.scoring_func);

    // Norm / rope. rope_theta may be flat or nested under rope_parameters.
    c.rms_norm_eps = r.get_double("rms_norm_eps", c.rms_norm_eps);
    if (r.has("rope_parameters") && r.at("rope_parameters")->is_object())
        c.rope_theta = r.at("rope_parameters")->get_double("rope_theta", c.rope_theta);
    else
        c.rope_theta = r.get_double("rope_theta", c.rope_theta);

    // DSA — GLM-5.2 carries these under an index_* or dsa block; read flat keys.
    c.use_dsa        = r.get_bool("use_dsa", r.has("index_topk"));
    c.index_n_heads  = r.get_int("index_n_heads", c.index_n_heads);
    c.index_head_dim = r.get_int("index_head_dim", c.index_head_dim);
    c.index_topk     = r.get_int("index_topk", c.index_topk);
    c.index_topk_freq = r.get_int("index_topk_freq", c.index_topk_freq);
    c.index_skip_topk_offset =
        r.get_int("index_skip_topk_offset", c.index_skip_topk_offset);
    c.indexer_rope_interleave =
        r.get_bool("indexer_rope_interleave", c.indexer_rope_interleave);
    if (r.has("indexer_types") && r.at("indexer_types")->is_array()) {
        c.indexer_types.clear();
        for (auto& v : r.at("indexer_types")->arr)
            c.indexer_types.push_back(v->as_string());
    }

    // MTP
    c.num_nextn_predict_layers =
        r.get_int("num_nextn_predict_layers", c.num_nextn_predict_layers);

    // Bookkeeping
    c.model_type   = r.get_string("model_type", c.model_type);
    c.torch_dtype  = r.get_string("torch_dtype", r.get_string("dtype", c.torch_dtype));
    c.bos_token_id = r.get_int("bos_token_id", c.bos_token_id);
    c.pad_token_id = r.get_int("pad_token_id", c.pad_token_id);
    // eos_token_id may be a scalar or a list (GLM-5.2 has 3 stop ids).
    if (r.has("eos_token_id")) {
        auto e = r.at("eos_token_id");
        if (e->is_array()) {
            c.eos_token_ids.clear();
            for (auto& v : e->arr) c.eos_token_ids.push_back(v->as_int());
            if (!c.eos_token_ids.empty()) c.eos_token_id = c.eos_token_ids[0];
        } else {
            c.eos_token_id = e->as_int();
            c.eos_token_ids = {c.eos_token_id};
        }
    }
    if (r.has("architectures") && r.at("architectures")->is_array() &&
        !r.at("architectures")->arr.empty()) {
        c.architecture = r.at("architectures")->arr[0]->as_string(c.architecture);
    }

    // Validation
    GLM_CHECK(c.hidden_size > 0 && c.num_hidden_layers > 0,
              "config has nonpositive hidden_size/num_hidden_layers");
    GLM_CHECK(c.num_attention_heads % c.num_key_value_heads == 0,
              "num_attention_heads (%lld) not divisible by kv heads (%lld)",
              (long long)c.num_attention_heads, (long long)c.num_key_value_heads);
    GLM_CHECK(c.num_experts_per_tok <= c.n_routed_experts,
              "top-k (%lld) exceeds routed experts (%lld)",
              (long long)c.num_experts_per_tok, (long long)c.n_routed_experts);

    return c;
}

void GLM52Config::summarize() const {
    GLM_INFO("GLM-5.2 config: %s (%s)", architecture.c_str(), model_type.c_str());
    GLM_INFO("  layers=%lld  hidden=%lld  vocab=%lld  ctx=%lld",
             (long long)num_hidden_layers, (long long)hidden_size,
             (long long)vocab_size, (long long)max_position_embeddings);
    GLM_INFO("  MLA: heads=%lld  q_lora=%lld  kv_lora=%lld  qk(nope+rope)=%lld+%lld  v_head=%lld",
             (long long)num_attention_heads, (long long)q_lora_rank, (long long)kv_lora_rank,
             (long long)qk_nope_head_dim, (long long)qk_rope_head_dim, (long long)v_head_dim);
    GLM_INFO("  MoE: experts=%lld  top-k=%lld  shared=%lld  moe_ffn=%lld  dense_first=%lld  scale=%.3f (%s)",
             (long long)n_routed_experts, (long long)num_experts_per_tok,
             (long long)n_shared_experts, (long long)moe_intermediate_size,
             (long long)first_k_dense_replace, routed_scaling_factor,
             scoring_func.c_str());
    GLM_INFO("  DSA: %s  index_topk=%lld  index_heads=%lldx%lld  freq=%lld  full_indexers=%zu",
             use_dsa ? "on" : "off", (long long)index_topk,
             (long long)index_n_heads, (long long)index_head_dim,
             (long long)index_topk_freq,
             std::count(indexer_types.begin(), indexer_types.end(), "full"));
}

}  // namespace glmserve
