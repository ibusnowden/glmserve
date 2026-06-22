#include "server.hpp"
#include "common.hpp"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
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

static std::string gen_id(const char* prefix) {
    static std::atomic<uint64_t> ctr{0};
    return std::string(prefix) + std::to_string(now_unix()) + "-" +
           std::to_string(ctr.fetch_add(1));
}

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------
Engine::Engine(EngineOptions opts) : opts_(std::move(opts)) {
    cfg_ = load_config(opts_.model_path);
    cfg_.summarize();

    SafeTensors st;
    st.load(opts_.model_path);

    model_ = std::make_unique<GLM52Model>(cfg_);
    model_->load(st, opts_.max_layers);

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
                                    cfg_.kv_cache_head_dim(), opts_.block_size, blocks);
    sched_ = std::make_unique<Scheduler>(opts_.max_concurrent);
    GLM_INFO("engine ready: %s, ctx=%lld, kv_blocks=%lld, max_concurrent=%d",
             opts_.served_model_name.c_str(), (long long)opts_.max_model_len,
             (long long)blocks, opts_.max_concurrent);
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

        c.finish_reason = "length";
        for (int step = 0; step < p.max_tokens; ++step) {
            int next = sampler.sample(logits, all_tokens, p);

            bool is_eos = false;
            if (!p.ignore_eos) {
                if (tokenizer_->is_stop(next)) is_eos = true;
                for (int64_t e : cfg_.eos_token_ids) if (next == static_cast<int>(e)) is_eos = true;
            }
            for (int sid : p.stop_token_ids) if (next == sid) is_eos = true;
            if (is_eos) { c.finish_reason = "stop"; break; }

            all_tokens.push_back(next);
            c.tokens.push_back(next);
            c.completion_tokens++;
            c.text += tokenizer_->decode_token(next);

            // stop strings
            bool stopped = false;
            for (auto& s : p.stop) {
                if (!s.empty()) {
                    auto pos = c.text.find(s);
                    if (pos != std::string::npos) { c.text.resize(pos); stopped = true; break; }
                }
            }
            if (stopped) { c.finish_reason = "stop"; break; }

            if (!flush(false)) { c.finish_reason = "cancel"; break; }
            if (sched_->is_cancelled(id)) { c.finish_reason = "cancel"; break; }
            if (c.completion_tokens >= p.max_tokens) { c.finish_reason = "length"; break; }

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

Engine::BenchResult Engine::profile(int prompt_len, int gen_len) {
    std::lock_guard<std::mutex> guard(gen_mu_);
    ensure_gpu();

    BenchResult r;
    r.prompt_len = prompt_len;
    r.gen_len = gen_len;
    r.gpu = gpu_active_;

    // Synthetic but valid token ids (deterministic, spread across the vocab).
    std::vector<int> prompt(prompt_len);
    for (int i = 0; i < prompt_len; ++i)
        prompt[i] = static_cast<int>((static_cast<int64_t>(i) * 1234577 + 7) % cfg_.vocab_size);

    if (gpu_active_) {
        // Warm-up prefill (primes cuBLAS, allocates scratch/KV) — not timed.
        model_->forward_gpu_prefill(prompt);

        Timer tp;
        std::vector<float> logits = model_->forward_gpu_prefill(prompt);
        r.prefill_ms = tp.ms();

        int next = host_argmax(logits);
        int64_t pos = prompt_len;
        Timer td;
        for (int s = 0; s < gen_len && pos < model_->gpu_kv_ctx(); ++s) {
            logits = model_->forward_gpu_decode(next, pos);
            next = host_argmax(logits);
            ++pos;
        }
        r.decode_ms = td.ms();
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
