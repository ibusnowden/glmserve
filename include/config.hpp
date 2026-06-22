// glmserve — GLM-5.2 model configuration (parsed from HF config.json).
//
// Architecture facts (from the GLM-5.2 model card / config.json):
//   architecture        GlmMoeDsaForCausalLM   (model_type: glm_moe_dsa)
//   753B params, 78 layers, hidden 6144
//   256 routed experts, 8 per token, 1 shared expert
//   first 3 MLP layers are dense, the rest are sparse MoE
//   64 attention heads, 64 KV heads
//   sigmoid expert scoring, routed_scaling_factor 2.5
//   DSA (DeepSeek-style sparse attention) with a lightning indexer
//   context window 1,048,576
//   MTP speculative path: 5 draft tokens (vLLM recipe)
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace glmserve {

struct GLM52Config {
    // --- core transformer dims -------------------------------------------------
    int64_t vocab_size            = 154880;
    int64_t hidden_size           = 6144;
    int64_t num_hidden_layers     = 78;
    int64_t num_attention_heads   = 64;
    int64_t num_key_value_heads   = 64;
    int64_t head_dim              = 192;   // config head_dim (== qk_nope_head_dim)
    int64_t intermediate_size     = 12288; // dense MLP (first_k_dense_replace layers)
    int64_t max_position_embeddings = 1048576;

    // --- MLA (DeepSeek-style latent attention) ---------------------------------
    // Q is projected hidden->q_lora_rank (RMSNorm)->n_heads*qk_head_dim.
    // KV is projected hidden->(kv_lora_rank + qk_rope_head_dim); the latent part
    // is RMSNorm'd then up-projected to n_heads*(qk_nope_head_dim + v_head_dim).
    // Each head's key = [nope(qk_nope) | rope(qk_rope)] = qk_head_dim; the rope
    // key is shared across heads (decoupled RoPE).
    int64_t q_lora_rank      = 2048;
    int64_t kv_lora_rank     = 512;
    int64_t qk_nope_head_dim = 192;
    int64_t qk_rope_head_dim = 64;
    int64_t qk_head_dim      = 256;   // qk_nope_head_dim + qk_rope_head_dim
    int64_t v_head_dim       = 256;
    bool    attention_bias   = false;
    bool    rope_interleave  = true;  // interleaved (GPT-J) RoPE, not NeoX half

    // --- MoE -------------------------------------------------------------------
    int64_t n_routed_experts      = 256;
    int64_t num_experts_per_tok   = 8;     // top-k
    int64_t n_shared_experts      = 1;
    int64_t moe_intermediate_size = 2048;  // per-expert MLP width
    int64_t first_k_dense_replace = 3;     // first N layers use dense MLP
    int64_t n_group               = 1;     // expert groups (group-limited routing)
    int64_t topk_group            = 1;
    double  routed_scaling_factor = 2.5;
    bool    norm_topk_prob        = true;
    std::string scoring_func      = "sigmoid";

    // --- normalization / rope --------------------------------------------------
    double  rms_norm_eps          = 1e-5;
    double  rope_theta            = 8000000.0;

    // --- DSA (sparse attention) -----------------------------------------------
    // The "lightning indexer" picks index_topk keys per query; for short prompts
    // (<= index_topk) attention is effectively dense and the indexer is a no-op.
    bool    use_dsa               = true;
    int64_t index_n_heads         = 64;
    int64_t index_head_dim        = 128;
    int64_t index_topk            = 2048;  // keys attended per query in sparse mode
    int64_t index_topk_freq       = 4;     // shared indexer every N layers after dense prefix
    int64_t index_skip_topk_offset = 3;    // skip offset used by GLM sparse index selection
    bool    indexer_rope_interleave = true;
    std::vector<std::string> indexer_types; // "full"/"shared" per transformer layer

    // --- MTP speculative decoding ---------------------------------------------
    int64_t num_nextn_predict_layers = 1;  // MTP heads present in weights
    int64_t mtp_draft_tokens         = 5;  // speculative depth (vLLM recipe)

    // --- bookkeeping -----------------------------------------------------------
    std::string model_type   = "glm_moe_dsa";
    std::string architecture = "GlmMoeDsaForCausalLM";
    std::string torch_dtype  = "bfloat16";
    int64_t     bos_token_id = 1;
    int64_t     eos_token_id = 154820;       // first of eos_token_ids
    std::vector<int64_t> eos_token_ids{154820, 154827, 154829};
    int64_t     pad_token_id = 154820;

    // Returns true for layers that use the dense MLP (the first_k_dense_replace).
    bool is_dense_layer(int64_t layer) const { return layer < first_k_dense_replace; }
    bool has_full_indexer_layer(int64_t layer) const {
        return layer >= 0 && layer < static_cast<int64_t>(indexer_types.size()) &&
               indexer_types[static_cast<size_t>(layer)] == "full";
    }

    // attention output width (n_heads * v_head_dim), input to o_proj
    int64_t attn_out_dim() const { return num_attention_heads * v_head_dim; }
    // Per-head KV slot width used by the (naive/unabsorbed) paged cache. K and V
    // are both materialized per head; GLM-5.2 has qk_head_dim == v_head_dim.
    int64_t kv_cache_head_dim() const { return std::max(qk_head_dim, v_head_dim); }

    void summarize() const;  // logs a one-screen summary
};

// Loads and validates config.json from the given model directory (or file path).
// Unknown keys are ignored; missing keys fall back to the defaults above.
GLM52Config load_config(const std::string& path);

}  // namespace glmserve
