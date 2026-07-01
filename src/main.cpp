// glmserve — CLI entry point.
//
//   glmserve serve    --model DIR [--port 8000] [--host 0.0.0.0] [--ctx 65536]
//                     [--name glm-5.2-local] [--max-layers N] [--block-size 16]
//   glmserve generate --model DIR --prompt "..." [--max-tokens 128] [--temp 0]
//                     [--top-p 1.0] [--top-k 0] [--seed 0] [--system "..."]
//   glmserve inspect  --model DIR
//
// `serve` exposes the OpenAI-compatible API for Pi Code / Cline / OpenCode.
#include "common.hpp"
#include "config.hpp"
#include "safetensors.hpp"
#include "server.hpp"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace glmserve;

namespace {

struct Args {
    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = kv.find(key);
        return it == kv.end() ? def : it->second;
    }
    int64_t get_int(const std::string& key, int64_t def) const {
        auto it = kv.find(key);
        return it == kv.end() ? def : std::stoll(it->second);
    }
    double get_double(const std::string& key, double def) const {
        auto it = kv.find(key);
        return it == kv.end() ? def : std::stod(it->second);
    }
    bool has(const std::string& key) const { return kv.count(key) > 0; }
    std::map<std::string, std::string> kv;
};

Args parse_args(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            std::string key = s.substr(2);
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                a.kv[key] = argv[++i];
            } else {
                a.kv[key] = "true";  // boolean flag
            }
        }
    }
    return a;
}

HttpServer* g_server = nullptr;
void on_sigint(int) { if (g_server) g_server->stop(); }

int cmd_inspect(const Args& a) {
    std::string model = a.get("model");
    GLM_CHECK(!model.empty(), "inspect requires --model DIR");
    GLM52Config cfg = load_config(model);
    cfg.summarize();

    SafeTensors st;
    st.load(model);
    auto names = st.names();
    GLM_INFO("safetensors: %zu tensors, %.2f GiB total", names.size(),
             st.total_bytes() / (1024.0 * 1024.0 * 1024.0));
    int shown = 0;
    for (auto& n : names) {
        if (shown++ >= 24) { GLM_INFO("  ... (%zu more)", names.size() - 24); break; }
        const TensorInfo* ti = st.info(n);
        std::string shape = "[";
        for (size_t i = 0; i < ti->shape.size(); ++i)
            shape += std::to_string(ti->shape[i]) + (i + 1 < ti->shape.size() ? "," : "");
        shape += "]";
        GLM_INFO("  %-52s %-8s %s", n.c_str(), dtype_name(ti->dtype), shape.c_str());
    }
    return 0;
}

EngineOptions engine_opts(const Args& a) {
    EngineOptions o;
    o.model_path = a.get("model");
    GLM_CHECK(!o.model_path.empty(), "requires --model DIR");
    o.served_model_name = a.get("name", "glm-5.2-local");
    o.max_model_len = a.get_int("ctx", 65536);
    o.block_size    = a.get_int("block-size", 16);
    o.kv_blocks     = a.get_int("kv-blocks", 0);
    o.max_layers    = a.get_int("max-layers", -1);
    o.max_concurrent = static_cast<int>(a.get_int("max-seqs", 1));
    o.use_gpu       = a.has("gpu");   // run the forward on the CUDA path (GPU=1 build)
    return o;
}

int cmd_serve(const Args& a) {
    Engine engine(engine_opts(a));
    HttpServer server;
    engine.register_routes(server);
    g_server = &server;
    std::signal(SIGINT, on_sigint);
    std::signal(SIGPIPE, SIG_IGN);   // don't die on broken client sockets
    server.listen_and_serve(a.get("host", "0.0.0.0"),
                            static_cast<int>(a.get_int("port", 8000)));
    return 0;
}

std::vector<int> parse_int_list(const std::string& s);

int cmd_generate(const Args& a) {
    Engine engine(engine_opts(a));
    SamplingParams p;
    p.temperature = static_cast<float>(a.get_double("temp", 0.0));  // greedy default
    p.top_p = static_cast<float>(a.get_double("top-p", 1.0));
    p.top_k = static_cast<int>(a.get_int("top-k", 0));
    p.max_tokens = static_cast<int>(a.get_int("max-tokens", 128));
    p.seed = static_cast<uint64_t>(a.get_int("seed", 0));
    p.mtp_draft_k = static_cast<int>(a.get_int("mtp-draft-k", 0));

    std::vector<ChatMessage> msgs;
    if (a.has("system")) msgs.push_back({"system", a.get("system")});
    msgs.push_back({"user", a.get("prompt", "Hello!")});

    std::fprintf(stderr, "--- generating ---\n");
    Completion c = engine.generate(msgs, p, [](const std::string& d) -> bool {
        std::fputs(d.c_str(), stdout);
        std::fflush(stdout);
        return true;
    });
    std::fprintf(stderr, "\n--- done: %d prompt + %d completion tokens, finish=%s ---\n",
                 c.prompt_tokens, c.completion_tokens, c.finish_reason.c_str());
    if (c.mtp_used) {
        std::fprintf(stderr, "--- mtp: groups=%d accepted=%d rejected=%d ---\n",
                     c.mtp_groups, c.mtp_accepted, c.mtp_rejected);
    }
    return 0;
}

int cmd_tokgen(const Args& a) {
    EngineOptions o = engine_opts(a);
    std::vector<int> prompt = parse_int_list(a.get("tokens", "3 1 4 1 5 9 2 6"));
    GLM_CHECK(!prompt.empty(), "tokgen requires --tokens \"id id id\"");
    SamplingParams p;
    p.temperature = static_cast<float>(a.get_double("temp", 0.0));
    p.top_p = static_cast<float>(a.get_double("top-p", 1.0));
    p.top_k = static_cast<int>(a.get_int("top-k", 0));
    p.max_tokens = static_cast<int>(a.get_int("max-tokens", 16));
    p.seed = static_cast<uint64_t>(a.get_int("seed", 0));
    p.ignore_eos = a.has("ignore-eos");
    p.mtp_draft_k = static_cast<int>(a.get_int("mtp-draft-k", 0));
    if (o.max_model_len < static_cast<int64_t>(prompt.size()) + p.max_tokens + 1)
        o.max_model_len = static_cast<int64_t>(prompt.size()) + p.max_tokens + 1;

    Engine engine(o);
    Completion c = engine.generate_tokens(prompt, p, nullptr);
    std::printf("tokgen: prompt=%d generated=%d finish=%s mtp=%s groups=%d accepted=%d rejected=%d\n",
                c.prompt_tokens, c.completion_tokens, c.finish_reason.c_str(),
                c.mtp_used ? "on" : "off", c.mtp_groups, c.mtp_accepted, c.mtp_rejected);
    std::printf("  tokens: ");
    for (int t : c.tokens) std::printf("%d ", t);
    std::printf("\n");
    return 0;
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ' ' || ch == ',') { if (!cur.empty()) { out.push_back(std::stoi(cur)); cur.clear(); } }
        else cur += ch;
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

// Correctness harness: single prefill over raw token ids -> dump last-position
// logits (and argmax) as JSON. Used by tests/test_logits_match.py.
int cmd_dump(const Args& a) {
    Engine engine(engine_opts(a));
    std::vector<int> prompt = parse_int_list(a.get("tokens", "1"));
    GLM_CHECK(!prompt.empty(), "dump requires --tokens \"id id id\"");

    std::vector<float> all;
    bool want_all = a.has("all");
    std::vector<float> last = engine.prefill_logits(prompt, want_all ? &all : nullptr);

    int argmax = 0;
    for (size_t i = 1; i < last.size(); ++i) if (last[i] > last[argmax]) argmax = static_cast<int>(i);

    std::string out_path = a.get("out");
    std::ostringstream o;
    o << "{\"prompt\":[";
    for (size_t i = 0; i < prompt.size(); ++i) o << prompt[i] << (i + 1 < prompt.size() ? "," : "");
    o << "],\"vocab\":" << last.size() << ",\"argmax\":" << argmax << ",\"logits\":[";
    for (size_t i = 0; i < last.size(); ++i) {
        o << last[i];
        if (i + 1 < last.size()) o << ",";
    }
    o << "]}";

    if (out_path.empty()) {
        std::fputs(o.str().c_str(), stdout);
        std::fputc('\n', stdout);
    } else {
        FILE* f = std::fopen(out_path.c_str(), "w");
        GLM_CHECK(f, "cannot open --out %s", out_path.c_str());
        std::fputs(o.str().c_str(), f);
        std::fclose(f);
        GLM_INFO("wrote logits -> %s (argmax=%d, vocab=%zu)", out_path.c_str(), argmax, last.size());
    }
    return 0;
}

// Throughput probe: time a prefill of --prompt-len tokens then a greedy decode
// of --gen-len tokens, reporting prefill and decode tok/s for the active backend.
int cmd_bench(const Args& a) {
    EngineOptions o = engine_opts(a);
    int prompt_len = static_cast<int>(a.get_int("prompt-len", 512));
    int gen_len    = static_cast<int>(a.get_int("gen-len", 128));
    // Make sure the device KV cache spans prompt + generation.
    if (o.max_model_len < prompt_len + gen_len + 1)
        o.max_model_len = prompt_len + gen_len + 1;
    Engine engine(o);

    std::fprintf(stderr, "--- benchmark: prompt_len=%d gen_len=%d ---\n", prompt_len, gen_len);
    Engine::BenchResult r = engine.profile(prompt_len, gen_len);

    std::printf("backend        : %s\n", r.gpu ? "CUDA (device-resident)" : "CPU reference");
    std::printf("prefill        : %d tok in %.2f ms  = %.1f tok/s\n",
                r.prompt_len, r.prefill_ms, r.prefill_tps());
    std::printf("decode         : %d tok in %.2f ms  = %.1f tok/s  (%.3f ms/tok)\n",
                r.gen_len, r.decode_ms, r.decode_tps(), r.decode_ms_per_tok());
    std::printf("target         : 1300 tok/s  (prefill %.0f%%, decode %.0f%%)\n",
                100.0 * r.prefill_tps() / 1300.0, 100.0 * r.decode_tps() / 1300.0);
    return 0;
}

// Validate the incremental GPU decode path against the re-prefill reference:
// greedily generate --gen tokens both ways and compare token streams + logits.
int cmd_gencheck(const Args& a) {
    EngineOptions o = engine_opts(a);
    std::vector<int> prompt = parse_int_list(a.get("tokens", "3 1 4 1 5 9 2 6"));
    int steps = static_cast<int>(a.get_int("gen", 32));
    if (o.max_model_len < static_cast<int64_t>(prompt.size()) + steps + 1)
        o.max_model_len = static_cast<int64_t>(prompt.size()) + steps + 1;
    Engine engine(o);

    Engine::DecodeCheck c = engine.check_decode(prompt, steps);
    if (!c.gpu) {
        std::printf("gencheck: GPU inactive (CPU build or no device / no --gpu) — skipped\n");
        return 0;
    }
    std::printf("gencheck: backend=CUDA steps=%d\n", c.steps);
    std::printf("  incremental tokens: ");
    for (int t : c.inc_tokens) std::printf("%d ", t);
    std::printf("\n  re-prefill  tokens: ");
    for (int t : c.ref_tokens) std::printf("%d ", t);
    std::printf("\n  tokens_match=%s  mismatch_at=%d  max_logit_diff=%.3e\n",
                c.tokens_match ? "YES" : "NO", c.mismatch_at, c.max_logit_diff);
    bool ok = c.tokens_match && c.max_logit_diff < 1e-2;
    std::printf("  RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int cmd_mtp(const Args& a) {
    std::string model = a.get("model");
    GLM_CHECK(!model.empty(), "mtp requires --model DIR");
    std::vector<int> context = parse_int_list(a.get("tokens", "3 1 4 1 5 9 2 6"));
    std::vector<int> draft = parse_int_list(a.get("draft", "2 7 1"));
    GLM_CHECK(!context.empty() && !draft.empty(), "mtp requires --tokens and --draft");

    GLM52Config cfg = load_config(model);
    SafeTensors st;
    st.load(model);
    GLM52Model m(cfg);
    m.load(st, a.get_int("max-layers", -1));
    std::vector<float> logits = m.mtp_draft_logits(context, draft);

    std::vector<int> argmax(draft.size(), 0);
    for (size_t r = 0; r < draft.size(); ++r) {
        const float* row = logits.data() + static_cast<int64_t>(r) * cfg.vocab_size;
        for (int64_t i = 1; i < cfg.vocab_size; ++i)
            if (row[i] > row[argmax[r]]) argmax[r] = static_cast<int>(i);
    }

    std::string out_path = a.get("out");
    std::ostringstream o;
    o << "{\"context\":[";
    for (size_t i = 0; i < context.size(); ++i) o << context[i] << (i + 1 < context.size() ? "," : "");
    o << "],\"draft\":[";
    for (size_t i = 0; i < draft.size(); ++i) o << draft[i] << (i + 1 < draft.size() ? "," : "");
    o << "],\"vocab\":" << cfg.vocab_size << ",\"argmax\":[";
    for (size_t i = 0; i < argmax.size(); ++i) o << argmax[i] << (i + 1 < argmax.size() ? "," : "");
    o << "],\"logits\":[";
    for (size_t i = 0; i < logits.size(); ++i) {
        o << logits[i];
        if (i + 1 < logits.size()) o << ",";
    }
    o << "]}";

    if (out_path.empty()) {
        std::fputs(o.str().c_str(), stdout);
        std::fputc('\n', stdout);
    } else {
        FILE* f = std::fopen(out_path.c_str(), "w");
        GLM_CHECK(f, "cannot open --out %s", out_path.c_str());
        std::fputs(o.str().c_str(), f);
        std::fclose(f);
        GLM_INFO("wrote MTP logits -> %s (draft=%zu, vocab=%lld)",
                 out_path.c_str(), draft.size(), (long long)cfg.vocab_size);
    }
    return 0;
}

int cmd_mtpcheck(const Args& a) {
    GLM_CHECK(!a.has("gpu"), "mtpcheck currently validates the CPU MTP path; omit --gpu");
    EngineOptions o = engine_opts(a);
    std::vector<int> prompt = parse_int_list(a.get("tokens", "3 1 4 1 5 9 2 6"));
    int steps = static_cast<int>(a.get_int("gen", 16));
    int draft_k = static_cast<int>(a.get_int("draft-k", 5));
    GLM_CHECK(!prompt.empty(), "mtpcheck requires --tokens \"id id id\"");
    GLM_CHECK(steps > 0 && draft_k > 0, "mtpcheck requires positive --gen and --draft-k");
    if (o.max_model_len < static_cast<int64_t>(prompt.size()) + steps + 1)
        o.max_model_len = static_cast<int64_t>(prompt.size()) + steps + 1;

    Engine engine(o);
    Engine::MTPCheck c = engine.check_mtp_speculative(prompt, steps, draft_k);
    std::printf("mtpcheck: steps=%d draft_k=%d groups=%d accepted=%d rejected=%d\n",
                c.steps, c.draft_k, c.groups, c.accepted, c.rejected);
    std::printf("  proposed: ");
    for (int t : c.proposed_tokens) std::printf("%d ", t);
    std::printf("\n  target  : ");
    for (int t : c.target_tokens) std::printf("%d ", t);
    std::printf("\n  output  : ");
    for (int t : c.output_tokens) std::printf("%d ", t);
    std::printf("\n  RESULT: PASS\n");
    return 0;
}

void usage() {
    std::fprintf(stderr,
        "glmserve — GLM-5.2 C++/CUDA inference engine\n\n"
        "usage:\n"
        "  glmserve serve    --model DIR [--port 8000] [--host 0.0.0.0] [--ctx 65536]\n"
        "                    [--name glm-5.2-local] [--max-layers N] [--max-seqs 1]\n"
        "  glmserve generate --model DIR --prompt \"...\" [--max-tokens 128] [--temp 0]\n"
        "                    [--top-p 1.0] [--top-k 0] [--seed 0] [--system \"...\"] [--mtp-draft-k K]\n"
        "  glmserve tokgen   --model DIR --tokens \"...\" [--max-tokens 16] [--mtp-draft-k K]\n"
        "  glmserve bench    --model DIR [--prompt-len 512] [--gen-len 128] [--gpu]\n"
        "  glmserve mtp      --model DIR --tokens \"...\" --draft \"...\" [--out JSON]\n"
        "  glmserve mtpcheck --model DIR --tokens \"...\" [--gen N] [--draft-k K]\n"
        "  glmserve inspect  --model DIR\n"
        "\n"
        "  --gpu   run the forward on the CUDA path (requires a GPU=1 build + a GPU)\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    try {
        Args a = parse_args(argc, argv, 2);
        if (cmd == "serve")    return cmd_serve(a);
        if (cmd == "generate") return cmd_generate(a);
        if (cmd == "tokgen")   return cmd_tokgen(a);
        if (cmd == "inspect")  return cmd_inspect(a);
        if (cmd == "dump")     return cmd_dump(a);
        if (cmd == "bench")    return cmd_bench(a);
        if (cmd == "gencheck") return cmd_gencheck(a);
        if (cmd == "mtp")      return cmd_mtp(a);
        if (cmd == "mtpcheck") return cmd_mtpcheck(a);
        usage();
        return 1;
    } catch (const std::exception& e) {
        GLM_ERROR("%s", e.what());
        return 1;
    }
}
