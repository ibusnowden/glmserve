// glmserve — inference Engine + OpenAI-compatible serving glue.
//
// Engine owns the model, tokenizer, paged KV cache, sampler, and scheduler. It
// exposes a single generate() entry point (prefill -> autoregressive decode with
// streaming callback) and registers the HTTP routes:
//     POST /v1/chat/completions   (stream + non-stream)
//     POST /v1/completions
//     GET  /v1/models
//     GET  /health
#pragma once

#include "config.hpp"
#include "http_server.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "nccl_comm.hpp"
#include "safetensors.hpp"
#include "sampler.hpp"
#include "scheduler.hpp"
#include "tokenizer.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace glmserve {

struct EngineOptions {
    std::string model_path;                 // dir with config.json + safetensors
    std::string served_model_name = "glm-5.2-local";
    int64_t max_model_len = 65536;          // context budget
    int64_t block_size    = 16;             // KV block size (tokens)
    int64_t kv_blocks     = 0;              // 0 => derive from max_model_len
    int64_t max_layers    = -1;             // truncate stack (testing)
    int     max_concurrent = 1;             // V0 batch=1
    bool    randomize_uninitialized = false;// fill missing weights w/ noise (toy)
    bool    use_gpu = false;                 // run the forward on the CUDA path (Phase 3)
};

struct Completion {
    std::string text;
    std::vector<int> tokens;                // generated token ids (in order)
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string finish_reason = "stop";     // "stop" | "length" | "cancel"
    bool mtp_used = false;
    int mtp_groups = 0;
    int mtp_accepted = 0;
    int mtp_rejected = 0;
};

class Engine {
public:
    explicit Engine(EngineOptions opts);

    const GLM52Config& config() const { return model_->config(); }
    const std::string& model_name() const { return opts_.served_model_name; }

    // Distributed (TP/PP) state. world_size>1 means this Engine is one rank of a
    // multi-GPU deployment; is_root() is rank 0 (the only rank that prints /
    // serves HTTP). barrier() syncs all ranks (no-op when single-rank).
    bool distributed() const { return dist_.world_size > 1; }
    bool is_root() const { return dist_.rank == 0; }
    const DistConfig& dist() const { return dist_; }
    void barrier();

    // Run a full generation. If on_delta is set, it is called with each new
    // UTF-8-safe text fragment as decoding proceeds.
    Completion generate(const std::vector<ChatMessage>& msgs, const SamplingParams& p,
                        const std::function<bool(const std::string&)>& on_delta);

    // Same but for a raw prompt (no chat template).
    Completion generate_text(const std::string& prompt, const SamplingParams& p,
                             const std::function<bool(const std::string&)>& on_delta);

    // Raw token-id prompt (no chat template, no tokenizer) — used by `dump` and
    // the correctness tests where exact token ids matter.
    Completion generate_tokens(const std::vector<int>& prompt_ids, const SamplingParams& p,
                               const std::function<bool(const std::string&)>& on_delta);

    // Single prefill over raw token ids; returns last-position logits [vocab].
    // If all_logits != nullptr it receives logits for every position.
    std::vector<float> prefill_logits(const std::vector<int>& prompt_ids,
                                      std::vector<float>* all_logits = nullptr);

    // Throughput probe: time a prefill of `prompt_len` synthetic tokens then a
    // greedy decode of `gen_len` tokens, on whichever backend is active.
    struct BenchResult {
        int    prompt_len = 0, gen_len = 0;
        double prefill_ms = 0, decode_ms = 0;
        bool   gpu = false;
        double prefill_tps() const { return prefill_ms > 0 ? prompt_len / (prefill_ms / 1e3) : 0; }
        double decode_tps()  const { return decode_ms  > 0 ? gen_len    / (decode_ms  / 1e3) : 0; }
        double decode_ms_per_tok() const { return gen_len ? decode_ms / gen_len : 0; }
    };
    BenchResult profile(int prompt_len, int gen_len);

    // Correctness probe for the incremental GPU decode: greedily generate `steps`
    // tokens two ways — the incremental KV path (forward_gpu_decode) vs the
    // known-correct re-prefill path (forward_gpu_prefill on the growing sequence)
    // — and compare the token streams + final logits. GPU-only; on the CPU
    // backend the two paths are identical so it trivially passes.
    struct DecodeCheck {
        int  steps = 0;
        bool tokens_match = true;
        int  mismatch_at = -1;         // first differing step, or -1
        double max_logit_diff = 0;     // over the final logits vector
        bool gpu = false;
        std::vector<int> inc_tokens, ref_tokens;
    };
    DecodeCheck check_decode(const std::vector<int>& prompt, int steps);

    struct MTPCheck {
        int steps = 0;
        int draft_k = 0;
        int groups = 0;
        int accepted = 0;
        int rejected = 0;
        std::vector<int> proposed_tokens;
        std::vector<int> target_tokens;
        std::vector<int> output_tokens;
    };
    MTPCheck check_mtp_speculative(const std::vector<int>& prompt, int steps, int draft_k);

    Tokenizer& tokenizer() { return *tokenizer_; }

    void register_routes(HttpServer& server);

private:
    Completion run(const std::vector<int>& prompt_tokens, const SamplingParams& p,
                   const std::function<bool(const std::string&)>& on_delta);

    void handle_chat(const HttpRequest& req, HttpResponder& res);
    void handle_completions(const HttpRequest& req, HttpResponder& res);
    void handle_models(const HttpRequest& req, HttpResponder& res);

    EngineOptions opts_;
    GLM52Config cfg_;
    std::unique_ptr<GLM52Model> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<KVCache> kv_;
    std::unique_ptr<Scheduler> sched_;
    std::mutex gen_mu_;   // V0: serialize forward passes over the shared model

    // Distributed (TP/PP) runtime. When world_size > 1 the Engine creates an
    // NCCL Communicator (which also pins this rank's GPU via cudaSetDevice) and
    // attaches it to the model before load, so load_gguf/load shard the weights
    // across the TP group and upload_to_gpu lands on the local device. The GPU
    // forward then all-reduces row-parallel outputs and pipelines hidden states
    // across PP stages. Under TP=8/PP=1 every rank is first+last stage, so each
    // rank holds a 1/8 weight shard + replicated embed/lm_head and runs the full
    // forward in lockstep; rank 0 samples/serves.
    std::unique_ptr<Communicator> comm_;
    DistConfig dist_{};

    // GPU path (Phase 3): weights are uploaded to the device lazily on first use.
    bool gpu_active_ = false;   // upload succeeded and the GPU forward is live
    bool gpu_tried_  = false;   // upload attempted (so a failure isn't retried)
    void ensure_gpu();          // upload weights once; falls back to CPU on failure
};

}  // namespace glmserve
