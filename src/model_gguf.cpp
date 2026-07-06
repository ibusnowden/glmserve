#include "model_gguf.hpp"
#include "common.hpp"
#include "gguf_quant.hpp"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace glmserve {
namespace {

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static int64_t parse_layer_id(const std::string& role) {
    const std::string prefix = "model.layers.";
    if (!starts_with(role, prefix)) return -1;
    size_t pos = prefix.size();
    size_t end = role.find('.', pos);
    if (end == std::string::npos || end == pos) return -1;
    return std::stoll(role.substr(pos, end - pos));
}

static void check_config_matches_gguf(const GLM52Config& cfg, const GGUFModel& gguf) {
    GLM_CHECK(cfg.hidden_size == 6144, "GGUF GLM-5.2 hidden size requires 6144, config has %lld",
              (long long)cfg.hidden_size);
    GLM_CHECK(cfg.vocab_size == 154880, "GGUF GLM-5.2 vocab requires 154880, config has %lld",
              (long long)cfg.vocab_size);
    GLM_CHECK(cfg.num_attention_heads == 64, "GGUF GLM-5.2 heads require 64, config has %lld",
              (long long)cfg.num_attention_heads);
    GLM_CHECK(cfg.q_lora_rank == 2048 && cfg.kv_lora_rank == 512,
              "GGUF GLM-5.2 MLA ranks require q=2048 kv=512, config has q=%lld kv=%lld",
              (long long)cfg.q_lora_rank, (long long)cfg.kv_lora_rank);
    GLM_CHECK(cfg.n_routed_experts == 256, "GGUF GLM-5.2 experts require 256, config has %lld",
              (long long)cfg.n_routed_experts);
    GLM_CHECK(cfg.first_k_dense_replace == 3, "GGUF GLM-5.2 dense prefix requires 3, config has %lld",
              (long long)cfg.first_k_dense_replace);
    GLM_CHECK(cfg.index_n_heads == 32 && cfg.index_head_dim == 128,
              "GGUF GLM-5.2 DSA indexer requires 32x128, config has %lldx%lld",
              (long long)cfg.index_n_heads, (long long)cfg.index_head_dim);

    uint64_t gguf_blocks = gguf.metadata_u64("glm-dsa.block_count");
    GLM_CHECK(gguf_blocks >= static_cast<uint64_t>(cfg.num_hidden_layers),
              "GGUF block_count %llu is smaller than config num_hidden_layers %lld",
              (unsigned long long)gguf_blocks, (long long)cfg.num_hidden_layers);
}

}  // namespace

void GLM52GGUFWeights::load(const std::string& path, const GLM52Config& cfg,
                            int64_t max_layers, bool touch_payloads) {
    ready_ = false;
    views_.clear();
    role_index_.clear();
    layout_ = GGUFGLM52Layout{};
    payload_checksum_ = 0;
    mapped_payload_bytes_ = 0;

    gguf_.load(path);
    gguf_.validate_glm52();
    check_config_matches_gguf(cfg, gguf_);
    layout_ = gguf_.build_glm52_layout();
    gguf_.map_payloads();

    int64_t max_transformer_layer = cfg.num_hidden_layers - 1;
    if (max_layers > 0) {
        max_transformer_layer = std::min<int64_t>(max_transformer_layer, max_layers - 1);
    }

    views_.reserve(layout_.modules.size());
    std::set<std::string> seen_roles;
    for (const GGUFModuleView& m : layout_.modules) {
        int64_t layer = parse_layer_id(m.role);
        if (layer >= 0 && layer < cfg.num_hidden_layers && layer > max_transformer_layer) {
            continue;
        }

        GLM_CHECK(m.tensor, "GGUF layout role %s has no tensor", m.role.c_str());
        GGUFWeightView view;
        view.role = m.role;
        view.name = m.name;
        view.tensor = m.tensor;
        view.data = gguf_.tensor_data(*m.tensor);
        GLM_CHECK(view.data, "GGUF tensor %s did not resolve to mapped data", m.name.c_str());

        size_t ix = views_.size();
        views_.push_back(std::move(view));
        if (seen_roles.insert(views_.back().role).second) {
            role_index_[views_.back().role] = ix;
        }
    }

    mapped_payload_bytes_ = layout_.tensor_bytes;
    if (touch_payloads) payload_checksum_ = gguf_.touch_payloads();
    ready_ = true;
}

const GGUFWeightView* GLM52GGUFWeights::role(const std::string& role) const {
    auto it = role_index_.find(role);
    return it == role_index_.end() ? nullptr : &views_[it->second];
}

uint64_t GLM52GGUFWeights::prefault(int rank, int world) const {
    if (world < 1) world = 1;
    // Collect the absolute file ranges of every loaded view, merged per shard.
    const auto& shards = gguf_.shards();
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges(shards.size());
    for (const GGUFWeightView& v : views_) {
        if (!v.tensor || v.tensor->n_bytes == 0) continue;
        const size_t si = v.tensor->shard_index;
        const uint64_t begin = shards[si].data_offset + v.tensor->offset;
        ranges[si].emplace_back(begin, begin + v.tensor->n_bytes);
    }
    const uint64_t CHUNK = 32ull << 20;  // 32 MiB read stripes
    std::vector<uint8_t> buf(static_cast<size_t>(CHUNK));
    uint64_t total = 0, chunk_idx = 0;
    for (size_t si = 0; si < shards.size(); ++si) {
        auto& rs = ranges[si];
        if (rs.empty()) continue;
        std::sort(rs.begin(), rs.end());
        std::vector<std::pair<uint64_t, uint64_t>> merged;
        for (const auto& r : rs) {
            if (!merged.empty() && r.first <= merged.back().second)
                merged.back().second = std::max(merged.back().second, r.second);
            else
                merged.push_back(r);
        }
        const int fd = ::open(shards[si].path.c_str(), O_RDONLY);
        if (fd < 0) continue;
        for (const auto& r : merged) {
            for (uint64_t off = r.first; off < r.second; off += CHUNK, ++chunk_idx) {
                if (static_cast<int>(chunk_idx % static_cast<uint64_t>(world)) != rank) continue;
                const uint64_t want = std::min<uint64_t>(CHUNK, r.second - off);
                uint64_t got = 0;
                while (got < want) {
                    const ssize_t n = ::pread(fd, buf.data() + got,
                                              static_cast<size_t>(want - got),
                                              static_cast<off_t>(off + got));
                    if (n <= 0) break;
                    got += static_cast<uint64_t>(n);
                }
                total += got;
            }
        }
        ::close(fd);
    }
    return total;
}

GGUFLinearView GLM52GGUFWeights::linear(const std::string& role) const {
    const GGUFWeightView* v = this->role(role);
    if (!v) return {};
    GLM_CHECK(v->tensor->shape.size() >= 2,
              "GGUF role %s is not a linear tensor with at least 2 dims: %s",
              role.c_str(), v->name.c_str());
    GGUFLinearView lin;
    lin.weight = v;
    lin.in_features = v->tensor->shape[0];
    lin.out_features = 1;
    for (size_t i = 1; i < v->tensor->shape.size(); ++i) {
        lin.out_features *= v->tensor->shape[i];
    }
    const uint64_t block_elems = gguf_type_block_elements(v->tensor->type);
    const uint64_t block_bytes = gguf_type_block_bytes(v->tensor->type);
    GLM_CHECK(lin.in_features % block_elems == 0,
              "GGUF linear %s in_features=%llu not divisible by block=%llu",
              v->name.c_str(), (unsigned long long)lin.in_features,
              (unsigned long long)block_elems);
    lin.row_bytes = (lin.in_features / block_elems) * block_bytes;
    GLM_CHECK(lin.row_bytes * lin.out_features == v->tensor->n_bytes,
              "GGUF linear %s row layout mismatch: row_bytes=%llu out=%llu tensor_bytes=%llu",
              v->name.c_str(), (unsigned long long)lin.row_bytes,
              (unsigned long long)lin.out_features,
              (unsigned long long)v->tensor->n_bytes);
    return lin;
}

// Build a GLM52Config from the GGUF metadata (glm-dsa.* keys). The GGUF is the
// only source of truth when serving straight from the 3-bit checkpoint (there
// is no config.json on disk), so every architecture field the forward needs is
// read back here. Light parse: opens shards for metadata + tensor tables only.
GLM52Config load_glm52_config_gguf(const std::string& gguf_path) {
    GGUFModel gguf;
    gguf.load(gguf_path);
    std::string arch = gguf.metadata_string("general.architecture");
    GLM_CHECK(arch == "glm-dsa", "load_glm52_config_gguf: general.architecture=%s != glm-dsa",
              arch.c_str());

    GLM52Config c;  // defaults already match GLM-5.2; override from GGUF metadata.
    c.model_type = "glm_moe_dsa";
    c.architecture = "GlmMoeDsaForCausalLM";
    c.torch_dtype = "bfloat16";
    c.vocab_size            = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.vocab_size", 154880));
    c.hidden_size           = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.embedding_length", 6144));
    uint64_t block_count    = gguf.metadata_u64("glm-dsa.block_count", 79);
    c.num_nextn_predict_layers = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.nextn_predict_layers", 1));
    c.num_hidden_layers     = static_cast<int64_t>(block_count) - c.num_nextn_predict_layers;
    GLM_CHECK(c.num_hidden_layers > 0, "GGUF block_count=%llu - nextn=%lld <= 0",
              (unsigned long long)block_count, (long long)c.num_nextn_predict_layers);
    c.num_attention_heads   = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.head_count", 64));
    c.num_key_value_heads   = c.num_attention_heads;  // MLA: shared latent KV
    c.intermediate_size     = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.feed_forward_length", 12288));
    c.max_position_embeddings = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.context_length", 1048576));
    c.q_lora_rank      = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.q_lora_rank", 2048));
    c.kv_lora_rank     = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.kv_lora_rank", 512));
    c.qk_head_dim      = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.key_length_mla", 256));
    c.qk_rope_head_dim = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.rope.dimension_count", 64));
    c.qk_nope_head_dim = c.qk_head_dim - c.qk_rope_head_dim;
    c.v_head_dim       = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.value_length_mla", 256));
    c.n_routed_experts      = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.expert_count", 256));
    c.num_experts_per_tok   = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.expert_used_count", 8));
    c.n_shared_experts      = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.expert_shared_count", 1));
    c.moe_intermediate_size = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.expert_feed_forward_length", 2048));
    c.first_k_dense_replace = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.leading_dense_block_count", 3));
    c.routed_scaling_factor = gguf.metadata_string("glm-dsa.expert_weights_scale", "2.5").empty()
                               ? 2.5 : std::stod(gguf.metadata_string("glm-dsa.expert_weights_scale", "2.5"));
    c.index_n_heads   = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.indexer.head_count", 32));
    c.index_head_dim  = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.indexer.key_length", 128));
    c.index_topk      = static_cast<int64_t>(gguf.metadata_u64("glm-dsa.attention.indexer.top_k", 2048));
    c.rope_theta      = std::stod(gguf.metadata_string("glm-dsa.rope.freq_base", "8000000"));
    c.rms_norm_eps    = std::stod(gguf.metadata_string("glm-dsa.attention.layer_norm_rms_epsilon", "1e-5"));
    // GLM-5.2 has no per-layer indexer type list in the GGUF; every DSA layer is
    // "shared" (learned top-k reused across the block group), so indexer_types
    // stays empty and has_full_indexer_layer() is uniformly false.
    c.indexer_types.clear();
    c.use_dsa = c.index_topk > 0;
    c.bos_token_id = 1;
    c.eos_token_id = 154820;
    c.eos_token_ids = {154820, 154827, 154829};
    c.pad_token_id = 154820;
    return c;
}

}  // namespace glmserve
