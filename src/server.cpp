#include "server.hpp"
#include "common.hpp"
#include "json.hpp"
#include "gguf.hpp"
#include "model_gguf.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>

namespace glmserve {

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

// Largest length <= L that ends on a UTF-8 character boundary.
static size_t utf8_trunc(const std::string& s, size_t L) {
    if (L >= s.size()) return s.size();
    while (L > 0 && (static_cast<unsigned char>(s[L]) & 0xC0) == 0x80) --L;
    return L;
}

static int64_t now_unix() { return static_cast<int64_t>(std::time(nullptr)); }

// Verify-chunk cap for GPU speculative decode (matches model_gpu.cpp's
// kVerifyMax per-row logits buffer).
static constexpr int kSpecGroupMax = 8;

static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::atoi(v);
}

static double env_double(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::atof(v);
}

static bool gpu_spec_adaptive_enabled() {
    const char* v = std::getenv("GLMSERVE_SPEC_ADAPTIVE");
    return !(v && *v && *v == '0');
}

static bool gpu_spec_should_fallback(int groups, int accepted) {
    if (!gpu_spec_adaptive_enabled()) return false;
    const int probe = std::max(1, env_int("GLMSERVE_SPEC_PROBE_GROUPS", 16));
    if (groups < probe) return false;
    const double min_accept = env_double("GLMSERVE_SPEC_MIN_ACCEPT", 2.0);
    return static_cast<double>(accepted) / groups < min_accept;
}

// N-gram lookup drafter (llama.cpp lookup-decoding style): the trailing
// (m-1)-gram of the context plus the committed seed token is searched backward
// through the context; on a hit the tokens that followed the previous
// occurrence become the draft. Pure CPU and deterministic from (ctx, seed), so
// every TP rank drafts the identical chunk without a collective. A wrong draft
// costs only rejected verify rows — the longest-prefix accept keeps the output
// stream equal to plain greedy decode regardless of draft origin.
// GLMSERVE_SPEC_NGRAM=0 disables; GLMSERVE_SPEC_NGRAM_MIN sets the minimum
// match length m (default 3: seed + 2 committed tokens).
static std::vector<int> ngram_draft(const std::vector<int>& ctx, int seed, int max_k) {
    static const bool enabled = [] {
        const char* v = std::getenv("GLMSERVE_SPEC_NGRAM");
        return !(v && *v && *v == '0');
    }();
    if (!enabled || max_k <= 0) return {};
    const int m_min = std::max(2, env_int("GLMSERVE_SPEC_NGRAM_MIN", 3));
    const int64_t n = static_cast<int64_t>(ctx.size());
    for (int m = std::max(m_min, 4); m >= m_min; --m) {
        if (n < m) continue;
        // pattern = ctx[n-m+1 .. n-1] + seed; a match ending at ctx[j] drafts
        // ctx[j+1 ..]. Scan backward for the most recent match whose
        // continuation fills max_k — inside a token run the latest match sits
        // at the tail with a 1-token continuation, so latest-match-only would
        // never draft more than one token.
        int64_t best_j = -1;
        int best_len = 0;
        for (int64_t j = n - 2; j >= m - 1; --j) {
            if (ctx[j] != seed) continue;
            bool hit = true;
            for (int t = 1; t < m; ++t)
                if (ctx[j - t] != ctx[n - t]) { hit = false; break; }
            if (!hit) continue;
            const int len = static_cast<int>(std::min<int64_t>(n - (j + 1), max_k));
            if (len > best_len) { best_len = len; best_j = j; }
            if (best_len >= max_k) break;
        }
        if (best_j >= 0)
            return std::vector<int>(ctx.begin() + best_j + 1,
                                    ctx.begin() + best_j + 1 + best_len);
    }
    return {};
}

static std::string gen_id(const char* prefix) {
    static std::atomic<uint64_t> ctr{0};
    return std::string(prefix) + std::to_string(now_unix()) + "-" +
           std::to_string(ctr.fetch_add(1));
}

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------
Engine::Engine(EngineOptions opts) : opts_(std::move(opts)) {
    const bool is_gguf = gguf_path_like(opts_.model_path);
    if (is_gguf) {
        // GGUF checkpoints carry the full architecture under glm-dsa.* metadata;
        // there is no config.json on disk.
        cfg_ = load_glm52_config_gguf(opts_.model_path);
    } else {
        cfg_ = load_config(opts_.model_path);
    }
    cfg_.summarize();

    // Distributed (TP/PP) runtime: when world_size > 1, create the NCCL
    // communicator (which pins this rank's GPU via cudaSetDevice) and attach it
    // to the model before load so load_gguf/load shard the weights across the TP
    // group and upload_to_gpu lands on the local device. Under TP=8/PP=1 every
    // rank is first+last stage and runs the full forward in lockstep.
    dist_ = dist_config_from_env();
    if (dist_.world_size > 1) {
        comm_ = std::make_unique<Communicator>(dist_);
        GLM_CHECK(comm_->active(), "distributed Engine requested (world_size=%d) but NCCL communicator is inactive",
                  dist_.world_size);
        model_ = std::make_unique<GLM52Model>(cfg_);
        model_->set_distributed(comm_.get());
    } else {
        model_ = std::make_unique<GLM52Model>(cfg_);
    }

    if (is_gguf) {
        // Native GGUF path: mmap the quant payloads and dequantize on the fly.
        model_->load_gguf(opts_.model_path, opts_.max_layers, false);
        GLM_CHECK(model_->gguf_ready(), "GGUF load did not leave model in ready state");
    } else {
        SafeTensors st;
        st.load(opts_.model_path);
        if (opts_.use_gpu && std::getenv("GLMSERVE_QUANT_ONLY") == nullptr)
            setenv("GLMSERVE_QUANT_ONLY", "1", 0);
        model_->load(st, opts_.max_layers);
    }

    tokenizer_ = std::make_unique<Tokenizer>();
    tokenizer_->load(opts_.model_path, cfg_.vocab_size);
    tokenizer_->set_special(static_cast<int>(cfg_.bos_token_id),
                            static_cast<int>(cfg_.eos_token_id));

    // KV cache sizing: enough blocks for max_concurrent sequences of max_model_len.
    int64_t blocks = opts_.kv_blocks;
    if (blocks <= 0) {
        int64_t per_seq = (opts_.max_model_len + opts_.block_size - 1) / opts_.block_size;
        blocks = per_seq * std::max(1, opts_.max_concurrent) + 4;
    }
    // MLA materializes per-head K/V into the cache; under tensor parallelism a
    // rank stores only its head slice (local_kv_heads()), == num_attention_heads
    // in the single-process build. Slot width is the (qk==v) per-head dim.
    kv_ = std::make_unique<KVCache>(model_->num_layers(), model_->local_kv_heads(),
                                    cfg_.kv_cache_head_dim(), opts_.block_size, blocks,
                                    cfg_.use_dsa ? cfg_.index_head_dim : 0);
    sched_ = std::make_unique<Scheduler>(opts_.max_concurrent);
    GLM_INFO("engine ready: %s, ctx=%lld, kv_blocks=%lld, max_concurrent=%d",
             opts_.served_model_name.c_str(), (long long)opts_.max_model_len,
             (long long)blocks, opts_.max_concurrent);
}

// Upload weights to the GPU once (lazily). If CUDA is unavailable (CPU build or
// no device) upload_to_gpu() returns false and we transparently stay on the CPU
// reference path. Caller must hold gen_mu_.
void Engine::barrier() {
    if (comm_) comm_->barrier();
}

// Upload weights to the GPU once (lazily). If CUDA is unavailable (CPU build or
// no device) upload_to_gpu() returns false and we transparently stay on the CPU
// reference path. Caller must hold gen_mu_.
void Engine::ensure_gpu() {
    if (!opts_.use_gpu || gpu_active_ || gpu_tried_) return;
    gpu_tried_ = true;   // attempt the upload at most once
    // Size the device KV cache to the full context budget so incremental decode
    // never overflows it within a single sequence's lifetime.
    gpu_active_ = model_->upload_to_gpu(opts_.max_model_len);
    if (gpu_active_)
        GLM_INFO("forward path: CUDA (device-resident weights)");
    else
        GLM_WARN("forward path: CPU (GPU requested but unavailable; rebuild with GPU=1 on a GPU node)");
}

// ---------------------------------------------------------------------------
// Generation core
// ---------------------------------------------------------------------------
Completion Engine::run(const std::vector<int>& prompt_in, const SamplingParams& p,
                       const std::function<bool(const std::string&)>& on_delta) {
    std::lock_guard<std::mutex> guard(gen_mu_);   // V0: one forward at a time
    ensure_gpu();

    // Clamp prompt to leave room for at least 1 generated token.
    std::vector<int> prompt = prompt_in;
    int64_t cap = opts_.max_model_len - 1;
    if (static_cast<int64_t>(prompt.size()) > cap) {
        GLM_WARN("prompt %zu tokens > ctx-1 (%lld); truncating to the tail",
                 prompt.size(), (long long)cap);
        prompt.erase(prompt.begin(), prompt.end() - cap);
    }

    int64_t id = sched_->admit();
    SequenceKV kv = kv_->make_sequence(id);
    Completion c;
    c.prompt_tokens = static_cast<int>(prompt.size());

    size_t max_stop = 0;
    for (auto& s : p.stop) max_stop = std::max(max_stop, s.size());

    try {
        // GPU prefill is a one-shot over the whole sequence (it manages its own
        // single KV block); the CPU path uses the persistent paged KV cache and
        // appends incrementally. Decode on the GPU re-prefills the growing
        // sequence each step (correct; incremental device KV is a later phase).
        std::vector<int> all_tokens = prompt;
        std::vector<float> logits = gpu_active_ ? model_->forward_gpu_prefill(all_tokens)
                                                : model_->forward(prompt, 0, kv);
        Sampler sampler(p.seed);
        const bool mtp_enabled = p.mtp_draft_k > 0;
        const bool mtp_gpu = mtp_enabled && gpu_active_;
        if (mtp_enabled) {
            if (gpu_active_)
                GLM_CHECK(model_->mtp_gpu_ready(),
                          "MTP speculative decode requested but the MTP block is not GPU-resident");
            else
                GLM_CHECK(model_->mtp_ready(), "MTP speculative generation requested but MTP weights are not loaded");
            GLM_CHECK(p.temperature <= 0.0f && p.top_k == 0 && p.top_p >= 1.0f,
                      "MTP speculative generation currently supports greedy sampling only");
            GLM_CHECK(p.repetition_penalty == 1.0f && p.frequency_penalty == 0.0f &&
                      p.presence_penalty == 0.0f,
                      "MTP speculative generation currently does not support sampling penalties");
            c.mtp_used = true;
        }
        // GPU spec decode: seed the MTP cache from the prompt's trunk hiddens
        // (a chunked long prefill leaves only its last chunk in scratch, so
        // absorb from there; earlier MTP cache rows stay zero).
        if (mtp_gpu) {
            const int64_t row0 = model_->gpu_chunk_row0();
            model_->mtp_gpu_absorb(
                std::vector<int>(prompt.begin() + row0 + 1, prompt.end()), row0);
        }

        size_t emitted = 0;  // bytes of c.text already streamed
        auto flush = [&](bool final_flush) -> bool {
            size_t upto = final_flush ? c.text.size()
                                      : utf8_trunc(c.text, c.text.size() > (max_stop ? max_stop - 1 : 0)
                                                              ? c.text.size() - (max_stop ? max_stop - 1 : 0)
                                                              : 0);
            if (upto > emitted && on_delta) {
                std::string delta = c.text.substr(emitted, upto - emitted);
                emitted = upto;
                return on_delta(delta);
            }
            emitted = std::max(emitted, upto);
            return true;
        };
        auto argmax_logits = [](const std::vector<float>& v) -> int {
            int a = 0;
            for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[a]) a = static_cast<int>(i);
            return a;
        };
        auto argmax_row = [&](const std::vector<float>& rows, int64_t row) -> int {
            const float* p0 = rows.data() + row * cfg_.vocab_size;
            int a = 0;
            for (int64_t i = 1; i < cfg_.vocab_size; ++i)
                if (p0[i] > p0[a]) a = static_cast<int>(i);
            return a;
        };
        auto is_stop_token = [&](int tok) -> bool {
            if (!p.ignore_eos) {
                if (tokenizer_->is_stop(tok)) return true;
                for (int64_t e : cfg_.eos_token_ids)
                    if (tok == static_cast<int>(e)) return true;
            }
            for (int sid : p.stop_token_ids) if (tok == sid) return true;
            return false;
        };
        auto record_token = [&](int tok) -> bool {
            all_tokens.push_back(tok);
            c.tokens.push_back(tok);
            c.completion_tokens++;
            c.text += tokenizer_->decode_token(tok);

            for (auto& s : p.stop) {
                if (!s.empty()) {
                    auto pos = c.text.find(s);
                    if (pos != std::string::npos) {
                        c.text.resize(pos);
                        c.finish_reason = "stop";
                        return false;
                    }
                }
            }
            if (!flush(false)) { c.finish_reason = "cancel"; return false; }
            if (sched_->is_cancelled(id)) { c.finish_reason = "cancel"; return false; }
            if (c.completion_tokens >= p.max_tokens) { c.finish_reason = "length"; return false; }
            return true;
        };

        c.finish_reason = "length";
        int gpu_seed = mtp_gpu ? argmax_logits(logits) : -1;
        bool mtp_gpu_active = mtp_gpu;
        while (c.completion_tokens < p.max_tokens) {
            if (mtp_gpu && !mtp_gpu_active) {
                const int tok = gpu_seed;
                if (is_stop_token(tok)) { c.finish_reason = "stop"; break; }
                if (!record_token(tok)) break;
                const int64_t pos = static_cast<int64_t>(all_tokens.size()) - 1;
                if (pos >= model_->gpu_kv_ctx()) { c.finish_reason = "length"; break; }
                logits = model_->forward_gpu_decode(tok, pos);
                gpu_seed = argmax_logits(logits);
                continue;
            }
            if (mtp_gpu_active) {
                // Groups: chunk = [committed next token | k-1 MTP drafts],
                // one trunk verify pass, longest matching prefix accepted, the
                // rejected tail rewound. The corrected token after a mismatch
                // is simply the next group's seed (argmax of the last accepted
                // row), so the emitted stream equals plain greedy decode.
                const int remaining = p.max_tokens - c.completion_tokens;
                const int group_k = std::min({p.mtp_draft_k, remaining, kSpecGroupMax});
                const int64_t L = static_cast<int64_t>(all_tokens.size());
                if (L + group_k > model_->gpu_kv_ctx()) { c.finish_reason = "length"; break; }
                std::vector<int> chunk;
                chunk.push_back(gpu_seed);
                // Drafts: n-gram lookup when it can match MTP's length (it may
                // draft past group_k, up to the verify-chunk cap), else MTP.
                const int draft_cap = static_cast<int>(std::min<int64_t>(
                    std::min(kSpecGroupMax, remaining),
                    model_->gpu_kv_ctx() - L)) - 1;
                std::vector<int> ds = ngram_draft(all_tokens, chunk[0], draft_cap);
                const bool from_ngram =
                    !ds.empty() && static_cast<int>(ds.size()) >= group_k - 1;
                if (!from_ngram) {
                    ds.clear();
                    if (group_k > 1) ds = model_->mtp_gpu_draft(chunk[0], group_k - 1);
                }
                chunk.insert(chunk.end(), ds.begin(), ds.end());
                const int chunk_n = static_cast<int>(chunk.size());
                ++c.mtp_groups;
                if (from_ngram) ++c.ngram_groups;
                std::vector<int> am;
                model_->forward_gpu_chunk_greedy(chunk, L, &am);
                int a = 1;
                while (a < chunk_n && chunk[a] == am[a - 1]) ++a;
                c.mtp_accepted += a;
                if (a < chunk_n) ++c.mtp_rejected;
                model_->gpu_rewind(L + a);
                bool finish_generation = false;
                for (int i = 0; i < a; ++i) {
                    const int tok = chunk[i];
                    if (is_stop_token(tok)) { c.finish_reason = "stop"; finish_generation = true; break; }
                    if (!record_token(tok)) { finish_generation = true; break; }
                }
                if (finish_generation) break;
                gpu_seed = am[a - 1];
                if (gpu_spec_should_fallback(c.mtp_groups, c.mtp_accepted)) {
                    mtp_gpu_active = false;
                    continue;
                }
                model_->mtp_gpu_absorb(std::vector<int>(chunk.begin() + 1, chunk.begin() + a), L);
                continue;
            }
            if (mtp_enabled) {
                const int remaining = p.max_tokens - c.completion_tokens;
                const int group_k = std::min(p.mtp_draft_k, remaining);
                std::vector<int> draft;
                draft.reserve(group_k);
                draft.push_back(argmax_logits(logits));
                while (static_cast<int>(draft.size()) < group_k) {
                    std::vector<float> mtp = model_->mtp_draft_logits(all_tokens, draft);
                    draft.push_back(argmax_row(mtp, static_cast<int64_t>(draft.size()) - 1));
                }

                ++c.mtp_groups;
                const int64_t old_kv_len = kv.length;
                std::vector<float> target_rows;
                std::vector<float> target_last = model_->forward(draft, kv.length, kv, &target_rows);
                bool finish_generation = false;
                for (int i = 0; i < group_k; ++i) {
                    const int target = (i == 0) ? argmax_logits(logits)
                                                : argmax_row(target_rows, i - 1);
                    if (draft[i] == target) {
                        ++c.mtp_accepted;
                        if (is_stop_token(target)) { c.finish_reason = "stop"; finish_generation = true; break; }
                        if (!record_token(target)) { finish_generation = true; break; }
                        logits = (i + 1 < group_k) ? std::vector<float>(
                                     target_rows.begin() + static_cast<int64_t>(i) * cfg_.vocab_size,
                                     target_rows.begin() + (static_cast<int64_t>(i) + 1) * cfg_.vocab_size)
                                                   : std::move(target_last);
                    } else {
                        ++c.mtp_rejected;
                        kv.length = old_kv_len + i;
                        if (is_stop_token(target)) { c.finish_reason = "stop"; finish_generation = true; break; }
                        logits = model_->forward({target}, kv.length, kv);
                        if (!record_token(target)) { finish_generation = true; break; }
                        break;
                    }
                }
                if (finish_generation) break;
                continue;
            }

            int next = sampler.sample(logits, all_tokens, p);

            // Keep TP ranks in lockstep: rank 0's sampled token is the only one
            // that matters (it has the full logits; under TP every rank computes
            // the same logits, but a stochastic sampler would diverge). Broadcast
            // it so every rank decodes the same sequence and the per-layer
            // all-reduces stay matched. (Greedy sampling is already identical.)
            if (comm_) comm_->bcast_int(&next, 1, 0);

            if (is_stop_token(next)) { c.finish_reason = "stop"; break; }
            if (!record_token(next)) break;

            // Incremental decode: the just-pushed token sits at the last position;
            // its K/V is appended to the persistent device cache and attention
            // reads the whole sequence so far (O(ctx) per token, not a re-prefill).
            int64_t pos = static_cast<int64_t>(all_tokens.size()) - 1;
            if (gpu_active_ && pos >= model_->gpu_kv_ctx()) { c.finish_reason = "length"; break; }
            logits = gpu_active_ ? model_->forward_gpu_decode(next, pos)
                                 : model_->forward({next}, kv.length, kv);
        }
        flush(true);
    } catch (...) {
        kv.release();
        sched_->complete(id);
        throw;
    }
    kv.release();
    sched_->complete(id);
    return c;
}

Completion Engine::generate(const std::vector<ChatMessage>& msgs, const SamplingParams& p,
                            const std::function<bool(const std::string&)>& on_delta) {
    return run(tokenizer_->encode_chat(msgs), p, on_delta);
}

Completion Engine::generate_text(const std::string& prompt, const SamplingParams& p,
                                 const std::function<bool(const std::string&)>& on_delta) {
    return run(tokenizer_->encode(prompt, /*add_special=*/true), p, on_delta);
}

Completion Engine::generate_tokens(const std::vector<int>& prompt_ids, const SamplingParams& p,
                                   const std::function<bool(const std::string&)>& on_delta) {
    return run(prompt_ids, p, on_delta);
}

std::vector<float> Engine::prefill_logits(const std::vector<int>& prompt_ids,
                                          std::vector<float>* all_logits) {
    std::lock_guard<std::mutex> guard(gen_mu_);
    ensure_gpu();
    // The GPU one-shot prefill returns last-position logits only. If a caller
    // needs every position (all_logits) fall back to the CPU reference, which is
    // also the correctness oracle the GPU path is validated against.
    if (gpu_active_ && all_logits == nullptr)
        return model_->forward_gpu_prefill(prompt_ids);

    int64_t id = sched_->admit();
    SequenceKV kv = kv_->make_sequence(id);
    std::vector<float> last;
    try {
        last = model_->forward(prompt_ids, 0, kv, all_logits);
    } catch (...) {
        kv.release();
        sched_->complete(id);
        throw;
    }
    kv.release();
    sched_->complete(id);
    return last;
}

// ---------------------------------------------------------------------------
// Throughput probe
// ---------------------------------------------------------------------------
static int host_argmax(const std::vector<float>& v) {
    int a = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[a]) a = static_cast<int>(i);
    return a;
}

static int host_argmax_row(const float* v, int64_t n) {
    int a = 0;
    for (int64_t i = 1; i < n; ++i) if (v[i] > v[a]) a = static_cast<int>(i);
    return a;
}

Engine::BenchResult Engine::profile(int prompt_len, int gen_len, int draft_k,
                                    const std::vector<int>* prompt_ids) {
    std::lock_guard<std::mutex> guard(gen_mu_);
    ensure_gpu();

    if (prompt_ids && !prompt_ids->empty())
        prompt_len = static_cast<int>(prompt_ids->size());

    BenchResult r;
    r.prompt_len = prompt_len;
    r.gen_len = gen_len;
    r.gpu = gpu_active_;

    // Synthetic but valid token ids (deterministic, spread across the vocab),
    // unless the caller supplied real ones.
    std::vector<int> prompt(prompt_len);
    if (prompt_ids && !prompt_ids->empty()) {
        prompt = *prompt_ids;
    } else {
        for (int i = 0; i < prompt_len; ++i)
            prompt[i] = static_cast<int>((static_cast<int64_t>(i) * 1234577 + 7) % cfg_.vocab_size);
    }

    if (gpu_active_) {
        // Warm-up prefill (primes cuBLAS, allocates scratch/KV) — not timed.
        model_->forward_gpu_prefill(prompt);
        gpu_prof_report("warmup", false);  // reset GLMSERVE_PROF accumulator

        Timer tp;
        std::vector<float> logits = model_->forward_gpu_prefill(prompt);
        r.prefill_ms = tp.ms();
        gpu_prof_report("prefill", is_root());

        if (draft_k > 0) {
            // Speculative decode: MTP drafts + trunk verify chunks (greedy).
            GLM_CHECK(model_->mtp_gpu_ready(),
                      "bench --mtp-draft-k requires a GPU-resident MTP block");
            const int64_t row0 = model_->gpu_chunk_row0();
            model_->mtp_gpu_absorb(
                std::vector<int>(prompt.begin() + row0 + 1, prompt.end()), row0);
            int seed = host_argmax(logits);
            std::vector<int> all_tokens = prompt;  // n-gram drafter history
            int64_t L = prompt_len;
            int gen = 0;
            Timer td;
            while (gen < gen_len) {
                if (r.spec_fallback) {
                    if (L >= model_->gpu_kv_ctx()) break;
                    logits = model_->forward_gpu_decode(seed, L);
                    all_tokens.push_back(seed);
                    seed = host_argmax(logits);
                    ++L;
                    ++gen;
                    continue;
                }
                const int group_k = std::min({draft_k, gen_len - gen, kSpecGroupMax});
                if (L + group_k > model_->gpu_kv_ctx()) break;
                std::vector<int> chunk;
                chunk.push_back(seed);
                const int draft_cap = static_cast<int>(std::min<int64_t>(
                    std::min(kSpecGroupMax, gen_len - gen),
                    model_->gpu_kv_ctx() - L)) - 1;
                std::vector<int> ds = ngram_draft(all_tokens, chunk[0], draft_cap);
                const bool from_ngram =
                    !ds.empty() && static_cast<int>(ds.size()) >= group_k - 1;
                if (!from_ngram) {
                    ds.clear();
                    if (group_k > 1) ds = model_->mtp_gpu_draft(chunk[0], group_k - 1);
                }
                chunk.insert(chunk.end(), ds.begin(), ds.end());
                const int chunk_n = static_cast<int>(chunk.size());
                if (from_ngram) ++r.spec_ngram_groups;
                std::vector<int> am;
                model_->forward_gpu_chunk_greedy(chunk, L, &am);
                int a = 1;
                while (a < chunk_n && chunk[a] == am[a - 1]) ++a;
                model_->gpu_rewind(L + a);
                all_tokens.insert(all_tokens.end(), chunk.begin(), chunk.begin() + a);
                seed = am[a - 1];
                L += a;
                gen += a;
                ++r.spec_groups;
                r.spec_accepted += a;
                if (gpu_spec_should_fallback(r.spec_groups, r.spec_accepted)) {
                    r.spec_fallback = true;
                    continue;
                }
                model_->mtp_gpu_absorb(std::vector<int>(chunk.begin() + 1, chunk.begin() + a), L - a);
            }
            r.decode_ms = td.ms();
            r.gen_len = gen;
            gpu_prof_report("decode", is_root());
            return r;
        }

        int next = host_argmax(logits);
        int64_t pos = prompt_len;
        Timer td;
        for (int s = 0; s < gen_len && pos < model_->gpu_kv_ctx(); ++s) {
            logits = model_->forward_gpu_decode(next, pos);
            next = host_argmax(logits);
            ++pos;
        }
        r.decode_ms = td.ms();
        gpu_prof_report("decode", is_root());
        return r;
    }

    // CPU reference path (incremental KV cache).
    int64_t id = sched_->admit();
    SequenceKV kv = kv_->make_sequence(id);
    try {
        Timer tp;
        std::vector<float> logits = model_->forward(prompt, 0, kv);
        r.prefill_ms = tp.ms();

        int next = host_argmax(logits);
        Timer td;
        for (int s = 0; s < gen_len; ++s) {
            logits = model_->forward({next}, kv.length, kv);
            next = host_argmax(logits);
        }
        r.decode_ms = td.ms();
    } catch (...) {
        kv.release();
        sched_->complete(id);
        throw;
    }
    kv.release();
    sched_->complete(id);
    return r;
}

Engine::DecodeCheck Engine::check_decode(const std::vector<int>& prompt, int steps) {
    std::lock_guard<std::mutex> guard(gen_mu_);
    ensure_gpu();

    DecodeCheck r;
    r.steps = steps;
    r.gpu = gpu_active_;
    GLM_CHECK(!prompt.empty(), "check_decode: empty prompt");

    if (!gpu_active_) {  // CPU path has no separate incremental path to diverge.
        GLM_WARN("check_decode: GPU inactive; nothing to compare (CPU path is single-path)");
        return r;
    }

    const int64_t P = static_cast<int64_t>(prompt.size());

    // Incremental: one prefill, then forward_gpu_decode per token.
    std::vector<float> logits = model_->forward_gpu_prefill(prompt);
    int64_t pos = P;
    for (int s = 0; s < steps; ++s) {
        int t = host_argmax(logits);
        r.inc_tokens.push_back(t);
        logits = model_->forward_gpu_decode(t, pos);
        ++pos;
    }
    std::vector<float> inc_last = logits;

    // Reference: re-prefill the growing sequence each step (independent of the
    // KV cache state above — each prefill resets it).
    std::vector<int> cur = prompt;
    logits = model_->forward_gpu_prefill(cur);
    for (int s = 0; s < steps; ++s) {
        int t = host_argmax(logits);
        r.ref_tokens.push_back(t);
        cur.push_back(t);
        logits = model_->forward_gpu_prefill(cur);
    }
    std::vector<float> ref_last = logits;

    for (int s = 0; s < steps; ++s) {
        if (r.inc_tokens[s] != r.ref_tokens[s]) {
            r.tokens_match = false;
            if (r.mismatch_at < 0) r.mismatch_at = s;
        }
    }
    double md = 0;
    for (size_t i = 0; i < inc_last.size() && i < ref_last.size(); ++i)
        md = std::max(md, static_cast<double>(std::fabs(inc_last[i] - ref_last[i])));
    r.max_logit_diff = md;
    return r;
}

Engine::ChunkCheck Engine::check_chunk_parity(const std::vector<int>& prompt, int k) {
    std::lock_guard<std::mutex> guard(gen_mu_);
    ensure_gpu();

    ChunkCheck r;
    r.k = k;
    r.gpu = gpu_active_;
    GLM_CHECK(!prompt.empty(), "check_chunk_parity: empty prompt");
    GLM_CHECK(k >= 2, "check_chunk_parity: k must be >= 2 (k == 1 is plain decode)");

    if (!gpu_active_) {
        GLM_WARN("check_chunk_parity: GPU inactive; the CPU path has no chunk variant");
        return r;
    }

    const int64_t P = static_cast<int64_t>(prompt.size());

    // Plain decode: token i's logits row (the reference a verify chunk re-checks).
    std::vector<float> logits = model_->forward_gpu_prefill(prompt);
    std::vector<int> chunk;
    chunk.push_back(host_argmax(logits));
    std::vector<std::vector<float>> dec_rows;
    for (int i = 0; i < k; ++i) {
        dec_rows.push_back(model_->forward_gpu_decode(chunk.back(), P + i));
        const int t = host_argmax(dec_rows.back());
        r.dec_tokens.push_back(t);
        if (i + 1 < k) chunk.push_back(t);
    }

    // Same tokens as one verify chunk over the rewound cache. Row i holds the
    // logits after chunk[0..i] — the decode step i's row above.
    model_->gpu_rewind(P);
    std::vector<float> rows;
    model_->forward_gpu_chunk(chunk, P, &rows);
    const int64_t V = static_cast<int64_t>(dec_rows[0].size());
    GLM_CHECK(static_cast<int64_t>(rows.size()) == static_cast<int64_t>(k) * V,
              "check_chunk_parity: chunk returned %zu logits, want %lld",
              rows.size(), (long long)(static_cast<int64_t>(k) * V));
    for (int i = 0; i < k; ++i) {
        const float* row = rows.data() + static_cast<int64_t>(i) * V;
        r.chunk_tokens.push_back(host_argmax_row(row, V));
        double md = 0;
        for (int64_t j = 0; j < V; ++j)
            md = std::max(md, static_cast<double>(std::fabs(row[j] - dec_rows[i][j])));
        r.row_diff.push_back(md);
        if (md > r.max_abs_diff) { r.max_abs_diff = md; r.worst_row = i; }
        if (r.chunk_tokens[i] != r.dec_tokens[i]) r.argmax_match = false;
    }
    return r;
}

Engine::MTPCheck Engine::check_mtp_speculative(const std::vector<int>& prompt, int steps,
                                               int draft_k) {
    std::lock_guard<std::mutex> guard(gen_mu_);
    ensure_gpu();
    GLM_CHECK(!prompt.empty(), "check_mtp_speculative: empty prompt");
    GLM_CHECK(steps > 0 && draft_k > 0, "check_mtp_speculative: steps/draft_k must be positive");

    MTPCheck r;
    r.steps = steps;
    r.draft_k = draft_k;
    r.gpu = gpu_active_;

    if (gpu_active_) {
        // GPU gate: the speculative stream must equal plain greedy decode.
        GLM_CHECK(model_->mtp_gpu_ready(),
                  "check_mtp_speculative: MTP block is not GPU-resident");
        // Plain greedy reference (incremental decode).
        std::vector<float> logits = model_->forward_gpu_prefill(prompt);
        int64_t pos = static_cast<int64_t>(prompt.size());
        for (int s = 0; s < steps; ++s) {
            const int t = host_argmax(logits);
            r.ref_tokens.push_back(t);
            if (s + 1 < steps) logits = model_->forward_gpu_decode(t, pos++);
        }
        // Speculative run (re-prefill resets the trunk cache).
        logits = model_->forward_gpu_prefill(prompt);
        {
            const int64_t row0 = model_->gpu_chunk_row0();
            model_->mtp_gpu_absorb(
                std::vector<int>(prompt.begin() + row0 + 1, prompt.end()), row0);
        }
        int64_t L = static_cast<int64_t>(prompt.size());
        int seed = host_argmax(logits);
        std::vector<int> all_tokens = prompt;  // n-gram drafter history
        while (static_cast<int>(r.output_tokens.size()) < steps) {
            const int remaining = steps - static_cast<int>(r.output_tokens.size());
            const int group_k = std::min({draft_k, remaining, kSpecGroupMax});
            std::vector<int> chunk;
            chunk.push_back(seed);
            const int draft_cap = static_cast<int>(std::min<int64_t>(
                std::min(kSpecGroupMax, remaining),
                model_->gpu_kv_ctx() - L)) - 1;
            std::vector<int> ds = ngram_draft(all_tokens, chunk[0], draft_cap);
            const bool from_ngram =
                !ds.empty() && static_cast<int>(ds.size()) >= group_k - 1;
            if (!from_ngram) {
                ds.clear();
                if (group_k > 1) ds = model_->mtp_gpu_draft(chunk[0], group_k - 1);
            }
            chunk.insert(chunk.end(), ds.begin(), ds.end());
            const int chunk_n = static_cast<int>(chunk.size());
            ++r.groups;
            std::vector<int> am;
            model_->forward_gpu_chunk_greedy(chunk, L, &am);
            int a = 1;
            for (int i = 1; i < chunk_n; ++i) {
                r.proposed_tokens.push_back(chunk[i]);
                r.target_tokens.push_back(am[i - 1]);
            }
            while (a < chunk_n && chunk[a] == am[a - 1]) ++a;
            r.accepted += a;
            if (a < chunk_n) ++r.rejected;
            all_tokens.insert(all_tokens.end(), chunk.begin(), chunk.begin() + a);
            model_->gpu_rewind(L + a);
            model_->mtp_gpu_absorb(std::vector<int>(chunk.begin() + 1, chunk.begin() + a), L);
            for (int i = 0; i < a && static_cast<int>(r.output_tokens.size()) < steps; ++i)
                r.output_tokens.push_back(chunk[i]);
            seed = am[a - 1];
            L += a;
        }
        r.match = r.output_tokens == r.ref_tokens;
        return r;
    }

    GLM_CHECK(model_->mtp_ready(), "check_mtp_speculative: MTP weights are not loaded");

    int64_t id = sched_->admit();
    SequenceKV kv = kv_->make_sequence(id);
    std::vector<int> all_tokens = prompt;
    std::vector<float> logits;
    try {
        logits = model_->forward(prompt, 0, kv);
        while (static_cast<int>(r.output_tokens.size()) < steps) {
            ++r.groups;
            const int remaining = steps - static_cast<int>(r.output_tokens.size());
            const int group_k = std::min(draft_k, remaining);

            std::vector<int> draft;
            draft.reserve(group_k);
            draft.push_back(host_argmax(logits));  // target-seeded first token
            while (static_cast<int>(draft.size()) < group_k) {
                std::vector<float> mtp = model_->mtp_draft_logits(all_tokens, draft);
                const float* row = mtp.data() +
                    (static_cast<int64_t>(draft.size()) - 1) * cfg_.vocab_size;
                int next = 0;
                for (int64_t i = 1; i < cfg_.vocab_size; ++i)
                    if (row[i] > row[next]) next = static_cast<int>(i);
                draft.push_back(next);
            }

            for (int tok : draft) {
                const int target = host_argmax(logits);
                r.proposed_tokens.push_back(tok);
                r.target_tokens.push_back(target);
                if (tok == target) {
                    ++r.accepted;
                    r.output_tokens.push_back(tok);
                    all_tokens.push_back(tok);
                    logits = model_->forward({tok}, kv.length, kv);
                } else {
                    ++r.rejected;
                    r.output_tokens.push_back(target);
                    all_tokens.push_back(target);
                    logits = model_->forward({target}, kv.length, kv);
                    break;
                }
                if (static_cast<int>(r.output_tokens.size()) >= steps) break;
            }
        }
    } catch (...) {
        kv.release();
        sched_->complete(id);
        throw;
    }
    kv.release();
    sched_->complete(id);
    return r;
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------
static SamplingParams parse_params(const json::Value& r) {
    SamplingParams p;
    p.temperature = static_cast<float>(r.get_double("temperature", 1.0));
    p.top_p       = static_cast<float>(r.get_double("top_p", 1.0));
    p.top_k       = static_cast<int>(r.get_int("top_k", 0));
    p.max_tokens  = static_cast<int>(r.get_int("max_tokens", r.get_int("max_completion_tokens", 512)));
    p.repetition_penalty = static_cast<float>(r.get_double("repetition_penalty", 1.0));
    p.frequency_penalty  = static_cast<float>(r.get_double("frequency_penalty", 0.0));
    p.presence_penalty   = static_cast<float>(r.get_double("presence_penalty", 0.0));
    p.seed = static_cast<uint64_t>(r.get_int("seed", 0));
    p.ignore_eos = r.get_bool("ignore_eos", false);
    p.mtp_draft_k = static_cast<int>(r.get_int("mtp_draft_k", 0));
    if (r.has("stop")) {
        auto s = r.at("stop");
        if (s->type == json::Type::String) p.stop.push_back(s->as_string());
        else if (s->is_array()) for (auto& e : s->arr) p.stop.push_back(e->as_string());
    }
    return p;
}

static std::vector<ChatMessage> parse_messages(const json::Value& r) {
    std::vector<ChatMessage> msgs;
    auto m = r.at("messages");
    GLM_CHECK(m && m->is_array(), "request missing 'messages' array");
    for (auto& e : m->arr) {
        ChatMessage cm;
        cm.role = e->get_string("role", "user");
        auto content = e->at("content");
        if (content) {
            if (content->type == json::Type::String) {
                cm.content = content->as_string();
            } else if (content->is_array()) {  // multimodal parts: concat text
                for (auto& part : content->arr) {
                    if (part->get_string("type") == "text")
                        cm.content += part->get_string("text");
                }
            }
        }
        msgs.push_back(std::move(cm));
    }
    return msgs;
}

void Engine::handle_chat(const HttpRequest& req, HttpResponder& res) {
    auto root = json::parse(req.body);
    GLM_CHECK(root && root->is_object(), "request body is not a JSON object");
    bool stream = root->get_bool("stream", false);
    SamplingParams p = parse_params(*root);
    std::vector<ChatMessage> msgs = parse_messages(*root);

    std::string cid = gen_id("chatcmpl-");
    int64_t created = now_unix();

    if (stream) {
        res.start_sse();
        // role announcement chunk
        {
            std::ostringstream o;
            o << "{\"id\":\"" << cid << "\",\"object\":\"chat.completion.chunk\",\"created\":"
              << created << ",\"model\":\"" << opts_.served_model_name
              << "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                 "\"finish_reason\":null}]}";
            res.sse(o.str());
        }
        Completion c = generate(msgs, p, [&](const std::string& delta) -> bool {
            if (!res.client_alive()) return false;
            std::ostringstream o;
            o << "{\"id\":\"" << cid << "\",\"object\":\"chat.completion.chunk\",\"created\":"
              << created << ",\"model\":\"" << opts_.served_model_name
              << "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
              << json_escape(delta) << "\"},\"finish_reason\":null}]}";
            res.sse(o.str());
            return res.client_alive();
        });
        std::ostringstream o;
        o << "{\"id\":\"" << cid << "\",\"object\":\"chat.completion.chunk\",\"created\":"
          << created << ",\"model\":\"" << opts_.served_model_name
          << "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\""
          << c.finish_reason << "\"}],\"usage\":{\"prompt_tokens\":" << c.prompt_tokens
          << ",\"completion_tokens\":" << c.completion_tokens << ",\"total_tokens\":"
          << (c.prompt_tokens + c.completion_tokens) << "}}";
        res.sse(o.str());
        res.sse_done();
    } else {
        Completion c = generate(msgs, p, nullptr);
        std::ostringstream o;
        o << "{\"id\":\"" << cid << "\",\"object\":\"chat.completion\",\"created\":" << created
          << ",\"model\":\"" << opts_.served_model_name
          << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
          << json_escape(c.text) << "\"},\"finish_reason\":\"" << c.finish_reason
          << "\"}],\"usage\":{\"prompt_tokens\":" << c.prompt_tokens
          << ",\"completion_tokens\":" << c.completion_tokens << ",\"total_tokens\":"
          << (c.prompt_tokens + c.completion_tokens) << "}}";
        res.send_json(200, o.str());
    }
}

void Engine::handle_completions(const HttpRequest& req, HttpResponder& res) {
    auto root = json::parse(req.body);
    GLM_CHECK(root && root->is_object(), "request body is not a JSON object");
    bool stream = root->get_bool("stream", false);
    SamplingParams p = parse_params(*root);
    std::string prompt = root->get_string("prompt");
    std::string cid = gen_id("cmpl-");
    int64_t created = now_unix();

    if (stream) {
        res.start_sse();
        Completion c = generate_text(prompt, p, [&](const std::string& delta) -> bool {
            std::ostringstream o;
            o << "{\"id\":\"" << cid << "\",\"object\":\"text_completion\",\"created\":" << created
              << ",\"model\":\"" << opts_.served_model_name
              << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(delta)
              << "\",\"finish_reason\":null}]}";
            res.sse(o.str());
            return res.client_alive();
        });
        std::ostringstream o;
        o << "{\"id\":\"" << cid << "\",\"object\":\"text_completion\",\"created\":" << created
          << ",\"model\":\"" << opts_.served_model_name
          << "\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\""
          << c.finish_reason << "\"}]}";
        res.sse(o.str());
        res.sse_done();
    } else {
        Completion c = generate_text(prompt, p, nullptr);
        std::ostringstream o;
        o << "{\"id\":\"" << cid << "\",\"object\":\"text_completion\",\"created\":" << created
          << ",\"model\":\"" << opts_.served_model_name
          << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(c.text)
          << "\",\"finish_reason\":\"" << c.finish_reason
          << "\"}],\"usage\":{\"prompt_tokens\":" << c.prompt_tokens
          << ",\"completion_tokens\":" << c.completion_tokens << ",\"total_tokens\":"
          << (c.prompt_tokens + c.completion_tokens) << "}}";
        res.send_json(200, o.str());
    }
}

void Engine::handle_models(const HttpRequest&, HttpResponder& res) {
    std::ostringstream o;
    o << "{\"object\":\"list\",\"data\":[{\"id\":\"" << opts_.served_model_name
      << "\",\"object\":\"model\",\"created\":" << now_unix()
      << ",\"owned_by\":\"glmserve\"}]}";
    res.send_json(200, o.str());
}

void Engine::register_routes(HttpServer& server) {
    server.route("POST", "/v1/chat/completions",
                 [this](const HttpRequest& q, HttpResponder& r) { handle_chat(q, r); });
    server.route("POST", "/v1/completions",
                 [this](const HttpRequest& q, HttpResponder& r) { handle_completions(q, r); });
    server.route("GET", "/v1/models",
                 [this](const HttpRequest& q, HttpResponder& r) { handle_models(q, r); });
    server.route("GET", "/health",
                 [](const HttpRequest&, HttpResponder& r) { r.send_json(200, "{\"status\":\"ok\"}"); });
}

}  // namespace glmserve
