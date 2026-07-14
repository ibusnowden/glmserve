// glmserve — GLM-5.2 model: weights, layer structures, and forward interface.
//
// The model holds float32 weights for the CPU reference path (dequantized on
// load). The forward pass implements one GLM-5.2 block end to end:
//   RMSNorm -> Q/K/V proj (+ optional qk-norm) -> RoPE -> attention (dense or
//   DSA-sparse) -> O proj -> residual -> post-attn RMSNorm ->
//   {dense MLP | MoE (sigmoid top-k + shared expert)} -> residual.
//
// ARCHITECTURAL DECISIONS:
//   - Dual execution paths: CPU reference (float32) + GPU CUDA (device-resident)
//   - Paged KV cache: fixed-size blocks, logical->physical mapping
//   - Tensor parallelism: Megatron-style column/row parallelism with all-reduce
//   - Pipeline parallelism: contiguous layer stages with hidden state pipelining
//   - MoE: 256 routed experts + 1 shared, 8 per token (sigmoid top-k)
//   - DSA: lightning indexer for sparse attention (top-2048 keys per query)
//   - MTP: Multi-Token Prediction for speculative decoding
//
// KV is stored in a paged KV cache (see kv_cache.hpp). The forward appends the
// given tokens at start_pos and returns logits for the final position.
//
// See docs/architecture.md for complete architectural overview.
#pragma once

#include "config.hpp"
#include "safetensors.hpp"
#include "model_gguf.hpp"
#include "kv_cache.hpp"
#include "nccl_comm.hpp"

#include <memory>
#include <string>
#include <vector>

namespace glmserve {

// A dense linear weight, stored row-major as [out_features, in_features]
// (HF convention: y = x @ W^T + b). Held in float32 for the reference path.
struct Linear {
    std::vector<float> w;   // out*in
    std::vector<float> b;   // out (optional, empty if none)
    std::vector<uint8_t> qweight;  // packed int4: [out, ceil(in/2)]
    std::vector<float> scales;     // [out, ceil(in/group_size)]
    int64_t out_features = 0;
    int64_t in_features  = 0;
    int64_t group_size   = 0;
    bool quantized_int4  = false;
    // GGUF block-quantized weight (Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS/F16).
    // When set, the GPU forward dequantizes on the fly (gemm_q); the f32 `w` is
    // empty. `data` is a host pointer into the mmap'd GGUF payload (not owned).
    uint32_t qtype = 0;             // GGML type id (0 = no GGUF quant)
    const uint8_t* qdata = nullptr; // [out, in] row-major, in contiguous
    int64_t row_bytes = 0;          // bytes per (logical) output row
    // Host source stride between rows (0 = contiguous == row_bytes). A
    // row-parallel TP shard keeps the mmap'd payload in place and views a
    // row_bytes-wide slice of each qstride-wide source row; the GPU upload
    // repacks it into a contiguous device buffer via cudaMemcpy2D.
    int64_t qstride = 0;
    std::vector<uint8_t> qbuf;      // owns sliced quant bytes (row-parallel shard)
    int64_t qsrc_stride() const { return qstride > 0 ? qstride : row_bytes; }
    bool has_q() const { return qtype != 0 && (qdata != nullptr || !qbuf.empty()); }
    bool has_bias() const { return !b.empty(); }
    bool valid()    const { return !w.empty() || !qweight.empty() || has_q(); }
    bool has_f32()  const { return !w.empty(); }
};

struct RMSNormW {
    std::vector<float> w;   // [dim]
    int64_t dim = 0;
};

struct DenseMLP {
    Linear gate_proj;   // [ffn, hidden]
    Linear up_proj;     // [ffn, hidden]
    Linear down_proj;   // [hidden, ffn]
};

struct Expert {
    Linear gate_proj;   // [moe_ffn, hidden]
    Linear up_proj;     // [moe_ffn, hidden]
    Linear down_proj;   // [hidden, moe_ffn]
};

struct MoEMLP {
    Linear router;                 // [n_experts, hidden] (gate)
    std::vector<float> e_bias;     // [n_experts] sigmoid score-correction bias (optional)
    std::vector<Expert> experts;   // n_routed_experts
    DenseMLP shared;               // shared expert (n_shared_experts merged)
    bool has_shared = false;
};

struct DSAIndexer {
    Linear   wq_b;          // query latent -> index heads
    Linear   wk;            // key latent -> index heads
    Linear   weights_proj;  // index score projection
    RMSNormW k_norm;
    std::vector<float> k_norm_bias;
    bool valid() const {
        return wq_b.valid() && wk.valid() && weights_proj.valid() && !k_norm.w.empty();
    }
};

struct Layer {
    RMSNormW input_norm;
    RMSNormW post_attn_norm;

    // MLA attention (DeepSeek-style latent projections)
    Linear   q_a_proj;             // hidden -> q_lora_rank
    RMSNormW q_a_norm;             // RMSNorm over q_lora_rank
    Linear   q_b_proj;             // q_lora_rank -> n_heads * qk_head_dim
    Linear   kv_a_proj;            // hidden -> kv_lora_rank + qk_rope_head_dim
    RMSNormW kv_a_norm;            // RMSNorm over kv_lora_rank
    Linear   kv_b_proj;            // kv_lora_rank -> n_heads * (qk_nope_head_dim + v_head_dim)
    Linear   o_proj;               // n_heads * v_head_dim -> hidden
    DSAIndexer indexer;             // optional GLM DSA lightning-indexer weights

    bool is_dense = false;
    DenseMLP dense_mlp;            // valid when is_dense
    MoEMLP   moe;                  // valid when !is_dense
};

struct MTPBlock {
    Linear eh_proj;                 // concat(enorm(embed), hnorm(hidden)) -> hidden
    RMSNormW enorm;
    RMSNormW hnorm;
    RMSNormW shared_head_norm;
    Layer layer;
    bool valid() const {
        return eh_proj.valid() && !enorm.w.empty() && !hnorm.w.empty() &&
               !shared_head_norm.w.empty();
    }
};

class GLM52Model {
public:
    explicit GLM52Model(GLM52Config cfg) : cfg_(std::move(cfg)) {}

    const GLM52Config& config() const { return cfg_; }

    // Attach a NCCL communicator so the model runs as one pipeline stage of a
    // TP=tp/PP=pp deployment: load() then keeps only this stage's layer range
    // (and the embedding / final-norm+lm_head only on the first / last stage),
    // and forward() hands the hidden state to the next stage instead of
    // producing logits. With no communicator (default) the model is a single
    // full stage and behaves exactly as before. Call before load().
    void set_distributed(Communicator* comm);
    const DistConfig& dist() const { return dist_; }
    // Global [begin,end) layer range this stage owns (== [0,num_layers) when not
    // pipeline-sharded). Local layer index i maps to global index begin()+i.
    LayerRange owned_layers() const { return {layer_begin_, layer_begin_ + num_layers()}; }

    // Load weights from a safetensors store (CPU reference: dequant to f32). When
    // a communicator is attached, only this pipeline stage's layer range is
    // loaded. max_layers > 0 truncates the stack (useful for one-layer
    // correctness tests) and is applied before the pipeline partition.
    void load(const SafeTensors& st, int64_t max_layers = -1);

    // Load mmap-backed GLM-5.2 GGUF quantized weight views. This validates the
    // full GGUF->glmserve tensor map and keeps the real payloads resident, but
    // generation still uses the safetensors/W4 path until GGML quant kernels are
    // wired into forward().
    void load_gguf(const std::string& gguf_path, int64_t max_layers = -1,
                   bool touch_payloads = false);
    bool gguf_ready() const;
    const GLM52GGUFWeights* gguf_weights() const { return gguf_weights_.get(); }

    // Number of layers held *on this stage* (the full count when not sharded).
    int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }

    // Attention heads materialized into the KV cache on *this* TP rank: the full
    // count when not tensor-parallel, num_attention_heads / tp_size under TP
    // (each rank owns a contiguous slice of heads). Used to size the KV cache.
    int64_t local_kv_heads() const { return tp_heads_; }

    // Forward a contiguous chunk of `tokens` starting at absolute position
    // `start_pos`, writing K/V into `kv`. Returns logits [vocab] for the LAST
    // token (sufficient for autoregressive greedy/sampled decode). If
    // all_logits != nullptr, it is filled with logits for every position
    // ([n_tokens * vocab], row-major) — used by tests and MTP verification.
    std::vector<float> forward(const std::vector<int>& tokens, int64_t start_pos,
                               SequenceKV& kv,
                               std::vector<float>* all_logits = nullptr);

    // CPU MTP verification path. Runs the base model over context_tokens, then
    // runs the first MTP block over draft_tokens and returns [draft,vocab]
    // logits row-major. This is the correctness surface used before wiring
    // speculative acceptance into serving.
    std::vector<float> mtp_draft_logits(const std::vector<int>& context_tokens,
                                        const std::vector<int>& draft_tokens);
    bool mtp_ready() const { return !mtp_blocks_.empty() && mtp_blocks_[0].valid(); }

    // Embedding lookup for a single token (exposed for tests).
    void embed(int token_id, float* out) const;

    int64_t hidden() const { return cfg_.hidden_size; }
    int64_t vocab()  const { return cfg_.vocab_size; }

    // --- GPU path (only meaningful in a GLMSERVE_CUDA build) ---
    // Upload f32 weights to the current CUDA device and allocate the persistent
    // device KV cache for up to `max_ctx` positions (0 => derive a default).
    // Returns false on a CPU build.
    bool upload_to_gpu(int64_t max_ctx = 0);
    bool gpu_ready() const { return gpu_state_ != nullptr; }
    // Positions the device KV cache can hold (0 if the GPU path is inactive).
    int64_t gpu_kv_ctx() const;
    // One-shot GPU prefill over raw token ids; returns last-position logits
    // [vocab]. Resets the device KV cache and fills positions [0, n), leaving the
    // sequence length at n so forward_gpu_decode() can continue from there.
    // Mirrors the CPU forward() exactly (used to validate the kernels on hardware).
    std::vector<float> forward_gpu_prefill(const std::vector<int>& tokens);
    // Suffix continuation of a resident sequence (multi-turn prefix reuse):
    // positions [0, start_pos) are already in the device cache from a previous
    // prefill/decode of the same token stream; only tokens[start_pos..) run,
    // as suffix chunks, and the last position's logits come back. start_pos
    // == 0 or a PP deployment falls back to the full one-shot prefill.
    std::vector<float> forward_gpu_prefill_from(const std::vector<int>& tokens,
                                                int64_t start_pos);
    // Positions currently valid in the device KV cache (0 when inactive).
    int64_t gpu_cur_len() const;
    // Incremental single-token decode at absolute position `pos`: appends this
    // token's K/V to the device cache and attends over [0, pos], reusing the
    // persistent scratch (no per-token allocation). Returns logits [vocab].
    std::vector<float> forward_gpu_decode(int token, int64_t pos);
    // GPU mirror of mtp_draft_logits() (single-rank reference path). Runs the
    // trunk over the context on-device (clobbers the persistent KV cache, like
    // forward_gpu_prefill), then the device-resident MTP block over the draft
    // tokens with its own single-layer draft cache. Returns [draft, vocab].
    std::vector<float> mtp_draft_logits_gpu(const std::vector<int>& context_tokens,
                                            const std::vector<int>& draft_tokens);

    // --- GPU speculative decode (MTP draft + trunk verify), TP-aware ---
    // The MTP block keeps a persistent cache over ABSOLUTE positions: after a
    // trunk pass over positions [p, p+n) the caller absorbs the committed
    // prefix (mtp_gpu_absorb), then drafts ahead of the trunk (mtp_gpu_draft),
    // verifies the chunk in one suffix pass (forward_gpu_chunk) and rewinds
    // the rejected tail (gpu_rewind). All calls run in TP lockstep; logits are
    // all-reduced, so every rank takes identical accept/reject decisions.
    bool mtp_gpu_ready() const;
    // Suffix trunk pass over `tokens` at absolute start_pos. Fills all_logits
    // with [n, vocab] (n <= kVerifyMax = 64) and leaves the chunk's pre-norm
    // hiddens in scratch for mtp_gpu_absorb.
    void forward_gpu_chunk(const std::vector<int>& tokens, int64_t start_pos,
                           std::vector<float>* all_logits);
    // Greedy twin: only the per-row argmax token ids leave the device (skips
    // the [n, vocab] fill + all-reduce + D2H — greedy spec decode needs no
    // full logits). Ranks combine shard winners with one tiny all-reduce.
    void forward_gpu_chunk_greedy(const std::vector<int>& tokens, int64_t start_pos,
                                  std::vector<int>* row_argmax);
    // Logical rewind of the trunk KV to `len` positions (caches are
    // append-only; subsequent passes overwrite the rejected tail).
    void gpu_rewind(int64_t len);
    // Absolute position of scratch row 0 after the last trunk pass — the
    // lowest position mtp_gpu_absorb can read (a chunked long prefill leaves
    // only its final chunk's hiddens in scratch).
    int64_t gpu_chunk_row0() const;
    // Absorb trunk positions [pos0, pos0 + next_tokens.size()) into the MTP
    // cache: position p pairs the last trunk pass's hidden row (p - pos0) with
    // the committed token at p+1 (= next_tokens[p - pos0]). Also stashes the
    // hidden at row next_tokens.size() as the draft seed state. Must run
    // directly after the trunk pass whose rows it reads.
    // set_seed=false skips refreshing the draft seed (gm.prev) — used by the
    // interleaved per-chunk absorbs during a long prefill, where "one past the
    // chunk" is not a valid scratch row; the final absorb sets the real seed.
    void mtp_gpu_absorb(const std::vector<int>& next_tokens, int64_t pos0,
                        bool set_seed = true);
    // Interleave MTP absorbs into chunked prefill: after each non-final chunk
    // the trunk hiddens are still in scratch, so the MTP cache can cover the
    // WHOLE prompt instead of just the last chunk (empty early rows starve the
    // NextN drafts at long context: 1.08 tok/group accepted at 8K depth).
    void set_mtp_prefill_absorb(bool on) { mtp_prefill_absorb_ = on; }
    // Greedy-draft k tokens ahead: seeded with `next_token` (the committed
    // token one past the absorbed prefix) and the stashed hidden state.
    // Extends the MTP cache by one committed position; speculative draft
    // entries beyond it are rewound internally. conf_tau > 0 truncates the
    // chain at the first draft whose top-1 probability under the MTP head
    // falls below tau (that token is dropped too — every returned draft is
    // confident), so a hard position degrades to a plain 1-row verify instead
    // of paying a full-width group. The gate value comes out of the same
    // all-reduce as the greedy winner, so TP ranks decide identically.
    std::vector<int> mtp_gpu_draft(int next_token, int k, float conf_tau = 0.0f);
    ~GLM52Model();

private:
    // One transformer block over a [n_tokens, hidden] activation buffer.
    void run_layer(int64_t layer_idx, float* hidden, int64_t n_tokens,
                   int64_t start_pos, SequenceKV& kv,
                   std::vector<std::vector<int64_t>>* shared_dsa_indices);
    void run_layer_block(const Layer& L, int64_t kv_layer_idx, float* hidden,
                         int64_t n_tokens, int64_t start_pos, SequenceKV& kv,
                         std::vector<std::vector<int64_t>>* shared_dsa_indices);
    void attention(const Layer& L, int64_t layer_idx, const float* normed,
                   float* attn_out, int64_t n_tokens, int64_t start_pos,
                   SequenceKV& kv, std::vector<std::vector<int64_t>>* shared_dsa_indices);
    void dense_mlp(const DenseMLP& m, const float* x, float* out, int64_t n_tokens) const;
    void moe_mlp(const MoEMLP& m, const float* x, float* out, int64_t n_tokens) const;

    GLM52Config cfg_;
    bool mtp_prefill_absorb_ = false;   // see set_mtp_prefill_absorb()
    RMSNormW final_norm_;
    std::vector<float> embed_tokens_;   // [vocab, hidden]
    Linear embed_lin_;                  // GGUF quant embedding [vocab, hidden] (has_q)
    Linear lm_head_;                    // [vocab, hidden] (may alias embeddings)
    int64_t lm_head_shard_off_ = 0;     // first vocab row of this rank's TP shard
    bool tied_embeddings_ = false;
    std::vector<Layer> layers_;          // this stage's layers (local-indexed)
    std::vector<MTPBlock> mtp_blocks_;   // optional MTP/next-token predictor blocks
    std::unique_ptr<GLM52GGUFWeights> gguf_weights_; // mmap-backed quant weight views
    void* gpu_state_ = nullptr;   // opaque GpuState* (model_gpu.cpp), null on CPU build

    // --- distributed state (single full rank by default) ---
    Communicator* comm_ = nullptr;   // null => not distributed
    DistConfig dist_{};              // rank/world/tp/pp; defaults to a single rank
    int64_t layer_begin_ = 0;        // global index of this stage's first layer
    int64_t tp_heads_ = 0;           // attention heads owned by this TP rank
};

// GLMSERVE_PROF=1 stage profiler (GPU build): the forward records per-stage
// cudaEvent marks; this prints the accumulated per-stage GPU-timeline table
// under `tag` (if `print`) and resets the accumulator. No-op otherwise.
void gpu_prof_report(const char* tag, bool print);

}  // namespace glmserve
