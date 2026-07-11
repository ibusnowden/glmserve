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
#include "gguf.hpp"
#include "gguf_quant.hpp"
#include "model.hpp"
#include "model_gguf.hpp"
#include "safetensors.hpp"
#include "server.hpp"

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
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
    if (gguf_path_like(model)) {
        GGUFModel gguf;
        gguf.load(model);
        if (a.has("require-glm52")) gguf.validate_glm52();
        if (a.has("check-glmserve-map")) {
            GGUFGLM52Layout layout = gguf.build_glm52_layout();
            GLM_INFO("gguf->glmserve layout: modules=%zu dense_layers=%zu moe_layers=%zu mtp=%s "
                     "quantized=%zu tensors %.2f GiB",
                     layout.modules.size(), layout.dense_layers, layout.moe_layers,
                     layout.has_mtp ? "yes" : "no", layout.quantized_tensors,
                     layout.quantized_tensor_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        if (a.has("touch-gguf-payloads")) {
            uint64_t checksum = gguf.touch_payloads();
            GLM_INFO("gguf payload mmap/touch: tensors=%zu checksum=%016llx",
                     gguf.tensors().size(), (unsigned long long)checksum);
        }

        GLM_INFO("gguf: %zu shards, %zu tensors, %.2f GiB files, %.2f GiB tensor payload",
                 gguf.shards().size(), gguf.tensors().size(),
                 gguf.total_file_bytes() / (1024.0 * 1024.0 * 1024.0),
                 gguf.total_tensor_bytes() / (1024.0 * 1024.0 * 1024.0));
        for (const auto& sh : gguf.shards()) {
            GLM_INFO("  shard %-44s tensors=%llu file=%.2f GiB data_off=%llu",
                     sh.path.substr(sh.path.find_last_of('/') + 1).c_str(),
                     (unsigned long long)sh.tensor_count,
                     sh.file_size / (1024.0 * 1024.0 * 1024.0),
                     (unsigned long long)sh.data_offset);
        }
        const char* keys[] = {
            "general.architecture", "general.name", "general.basename",
            "general.quantized_by", "general.size_label", "glm-dsa.block_count",
            "glm-dsa.context_length", "glm-dsa.expert_count", "tokenizer.ggml.model",
        };
        for (const char* key : keys) {
            if (gguf.has_metadata(key)) GLM_INFO("  %s=%s", key, gguf.metadata_string(key).c_str());
        }
        std::string qsummary;
        for (const auto& kv : gguf.quant_counts()) {
            if (!qsummary.empty()) qsummary += ",";
            qsummary += kv.first + ":" + std::to_string(kv.second);
        }
        GLM_INFO("  quant_types=%s", qsummary.c_str());

        int shown = 0;
        for (const auto& t : gguf.tensors()) {
            if (shown++ >= 24) {
                GLM_INFO("  ... (%zu more)", gguf.tensors().size() - 24);
                break;
            }
            std::string shape = "[";
            for (size_t i = 0; i < t.shape.size(); ++i)
                shape += std::to_string(t.shape[i]) + (i + 1 < t.shape.size() ? "," : "");
            shape += "]";
            GLM_INFO("  %-52s %-8s %s %.2f MiB", t.name.c_str(), ggml_type_name(t.type),
                     shape.c_str(), t.n_bytes / (1024.0 * 1024.0));
        }
        return 0;
    }

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

int cmd_load_gguf(const Args& a) {
    std::string model = a.get("model");
    GLM_CHECK(!model.empty(), "load-gguf requires --model GGUF_FILE_OR_DIR");
    GLM_CHECK(gguf_path_like(model), "load-gguf requires a .gguf file or directory");
    // This is a metadata/dequant-smoke gate touching a few blocks; skip the
    // serving path's full payload prefault (hundreds of GiB of reads).
    setenv("GLMSERVE_NO_PREFAULT", "1", 0);

    GLM52Config cfg;
    cfg.index_n_heads = 32;  // real GGUF tensors: indexer.attn_q_b [2048,4096]
    cfg.summarize();

    GLM52Model m(cfg);
    m.load_gguf(model, a.get_int("max-layers", -1), a.has("touch-payloads"));
    GLM_CHECK(m.gguf_ready(), "GGUF load did not leave model in ready state");

    const GLM52GGUFWeights* weights = m.gguf_weights();
    GLM_CHECK(weights != nullptr, "GGUF weights missing after load");
    const char* smoke_roles[] = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.3.mlp.experts.*.gate_proj.weight",
        "model.layers.78.eh_proj.weight",
        "lm_head.weight",
    };
    for (const char* role : smoke_roles) {
        const GGUFWeightView* v = weights->role(role);
        GLM_CHECK(v, "missing loaded GGUF role after model load: %s", role);
        GLM_INFO("  role %-48s <- %-36s %-8s %.2f MiB",
                 role, v->name.c_str(), ggml_type_name(v->tensor->type),
                 v->tensor->n_bytes / (1024.0 * 1024.0));
    }

    if (a.has("dequant-smoke")) {
        std::set<uint32_t> seen;
        for (const GGUFWeightView& v : weights->views()) {
            uint32_t type = v.tensor->type;
            if (!seen.insert(type).second) continue;
            GLM_CHECK(gguf_type_can_dequantize(type),
                      "no CPU dequant smoke path for GGUF type %s (%u)",
                      ggml_type_name(type), type);
            uint64_t n = std::min<uint64_t>(v.tensor->n_elements,
                                            std::max<uint64_t>(1, gguf_type_block_elements(type)));
            std::vector<float> f = gguf_dequantize_prefix(type, v.data, n);
            double sum_abs = 0.0;
            float max_abs = 0.0f;
            for (float x : f) {
                GLM_CHECK(std::isfinite(x), "dequant produced non-finite value for %s",
                          v.name.c_str());
                float ax = std::fabs(x);
                sum_abs += ax;
                max_abs = std::max(max_abs, ax);
            }
            GLM_INFO("  dequant %-8s %-44s n=%llu checksum=%016llx max_abs=%.6g sum_abs=%.6g",
                     ggml_type_name(type), v.name.c_str(), (unsigned long long)n,
                     (unsigned long long)gguf_f32_checksum(f.data(), f.size()),
                     max_abs, sum_abs);
        }
    }

    if (a.has("require-dequant-checksums")) {
        struct Expected {
            const char* role;
            uint64_t checksum;
        };
        const Expected expected[] = {
            {"model.embed_tokens.weight", 0xf53a19f5f5d7d8cbull},
            {"model.norm.weight", 0x2f1283a084b74c14ull},
            {"lm_head.weight", 0x238cce2bf9fa0161ull},
            {"model.layers.3.mlp.experts.*.gate_proj.weight", 0xebec65e8bc88d245ull},
            {"model.layers.3.mlp.experts.*.down_proj.weight", 0xb418d9e878d4c0d2ull},
            {"model.layers.8.mlp.experts.*.down_proj.weight", 0x00d4f4cb86c63fd6ull},
            {"model.layers.78.mlp.experts.*.gate_proj.weight", 0x84d8fa95be0b504dull},
            {"model.layers.78.mlp.experts.*.down_proj.weight", 0x23063095b22bed2eull},
        };
        for (const Expected& e : expected) {
            const GGUFWeightView* v = weights->role(e.role);
            GLM_CHECK(v, "missing required dequant checksum role: %s", e.role);
            uint64_t n = std::min<uint64_t>(v->tensor->n_elements,
                                            std::max<uint64_t>(1, gguf_type_block_elements(v->tensor->type)));
            std::vector<float> f = gguf_dequantize_prefix(v->tensor->type, v->data, n);
            uint64_t got = gguf_f32_checksum(f.data(), f.size());
            GLM_CHECK(got == e.checksum,
                      "dequant checksum mismatch for %s (%s): got %016llx expected %016llx",
                      v->name.c_str(), ggml_type_name(v->tensor->type),
                      (unsigned long long)got, (unsigned long long)e.checksum);
        }
        GLM_INFO("  dequant checksum gate: PASS (%zu reference blocks)", sizeof(expected) / sizeof(expected[0]));
    }

    if (a.has("linear-smoke")) {
        auto make_input = [](uint64_t n) {
            std::vector<float> x(static_cast<size_t>(n));
            uint32_t state = 0x9e3779b9u;
            for (uint64_t i = 0; i < n; ++i) {
                state = state * 1664525u + 1013904223u;
                int v = static_cast<int>((state >> 8) & 0x3ffu) - 512;
                x[static_cast<size_t>(i)] = static_cast<float>(v) / 2048.0f;
            }
            return x;
        };
        auto dot_role = [&](const char* role, uint64_t row) {
            GGUFLinearView lin = weights->linear(role);
            GLM_CHECK(row < lin.out_features,
                      "linear smoke row %llu out of range for %s with out=%llu",
                      (unsigned long long)row, role, (unsigned long long)lin.out_features);
            std::vector<float> x = make_input(lin.in_features);
            double dot = gguf_row_dot(lin.weight->tensor->type, lin.weight->data,
                                      lin.in_features, row, x.data());
            GLM_CHECK(std::isfinite(dot), "linear smoke produced non-finite dot for %s", role);
            GLM_INFO("  linear %-48s row=%llu in=%llu out=%llu type=%-8s dot=%.9g",
                     role, (unsigned long long)row,
                     (unsigned long long)lin.in_features,
                     (unsigned long long)lin.out_features,
                     ggml_type_name(lin.weight->tensor->type), dot);
            return dot;
        };
        const double d0 = dot_role("lm_head.weight", 154820);
        const double d1 = dot_role("model.layers.0.self_attn.q_a_proj.weight", 17);
        const double d2 = dot_role("model.layers.3.mlp.experts.*.gate_proj.weight", 257);
        const double d3 = dot_role("model.layers.3.mlp.experts.*.down_proj.weight", 1024);
        if (a.has("require-linear-checksums")) {
            auto close = [](double got, double want) {
                return std::fabs(got - want) <= 1e-7 * std::max(1.0, std::fabs(want));
            };
            GLM_CHECK(close(d0, -0.3679340725), "linear checksum mismatch lm_head: %.12g", d0);
            GLM_CHECK(close(d1, -0.0584288574), "linear checksum mismatch q_a: %.12g", d1);
            GLM_CHECK(close(d2, -0.0459673857), "linear checksum mismatch iq3 expert gate: %.12g", d2);
            GLM_CHECK(close(d3, -0.1075344397), "linear checksum mismatch iq4 expert down: %.12g", d3);
            GLM_INFO("  linear checksum gate: PASS (4 reference row dots)");
        }
        GLM_INFO("  linear smoke aggregate=%.9g", d0 + d1 + d2 + d3);
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
    std::vector<int> prompt;
    std::string tokens_file = a.get("tokens-file");
    if (!tokens_file.empty()) {
        std::ifstream f(tokens_file);
        GLM_CHECK(f.good(), "cannot open --tokens-file %s", tokens_file.c_str());
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = parse_int_list(ss.str());
    } else {
        prompt = parse_int_list(a.get("tokens", "3 1 4 1 5 9 2 6"));
    }
    GLM_CHECK(!prompt.empty(), "tokgen requires --tokens \"id id id\" or --tokens-file");
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
    std::printf("tokgen: prompt=%d generated=%d finish=%s mtp=%s groups=%d accepted=%d rejected=%d ngram_groups=%d\n",
                c.prompt_tokens, c.completion_tokens, c.finish_reason.c_str(),
                c.mtp_used ? "on" : "off", c.mtp_groups, c.mtp_accepted, c.mtp_rejected,
                c.ngram_groups);
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
// In a distributed (TP/PP) run every rank runs the same forward in lockstep (the
// NCCL all-reduces synchronize them); only rank 0 prints, then all ranks barrier.
int cmd_bench(const Args& a) {
    EngineOptions o = engine_opts(a);
    int prompt_len = static_cast<int>(a.get_int("prompt-len", 512));
    int gen_len    = static_cast<int>(a.get_int("gen-len", 128));
    int draft_k    = static_cast<int>(a.get_int("mtp-draft-k", 0));
    // Real token ids (whitespace-separated) instead of the synthetic prompt —
    // spec-decode acceptance is only meaningful on real text.
    std::vector<int> prompt_ids;
    std::string prompt_file = a.get("prompt-file");
    if (!prompt_file.empty()) {
        std::ifstream f(prompt_file);
        GLM_CHECK(f.good(), "cannot open --prompt-file %s", prompt_file.c_str());
        std::stringstream ss;
        ss << f.rdbuf();
        prompt_ids = parse_int_list(ss.str());
        GLM_CHECK(!prompt_ids.empty(), "--prompt-file %s has no token ids", prompt_file.c_str());
        prompt_len = static_cast<int>(prompt_ids.size());
    }
    // Make sure the device KV cache spans prompt + generation.
    if (o.max_model_len < prompt_len + gen_len + 1)
        o.max_model_len = prompt_len + gen_len + 1;
    Engine engine(o);
    const bool root = engine.is_root();

    if (root)
        std::fprintf(stderr, "--- benchmark: prompt_len=%d gen_len=%d draft_k=%d (world=%d tp=%d pp=%d rank=%d) ---\n",
                     prompt_len, gen_len, draft_k, engine.dist().world_size,
                     engine.dist().tp_size, engine.dist().pp_size, engine.dist().rank);
    Engine::BenchResult r = engine.profile(prompt_len, gen_len, draft_k,
                                           prompt_ids.empty() ? nullptr : &prompt_ids);
    engine.barrier();

    if (root) {
        std::printf("backend        : %s\n", r.gpu ? "CUDA (device-resident)" : "CPU reference");
        std::printf("prefill        : %d tok in %.2f ms  = %.1f tok/s\n",
                    r.prompt_len, r.prefill_ms, r.prefill_tps());
        std::printf("decode         : %d tok in %.2f ms  = %.1f tok/s  (%.3f ms/tok)\n",
                    r.gen_len, r.decode_ms, r.decode_tps(), r.decode_ms_per_tok());
        if (r.spec_groups > 0)
            std::printf("speculative    : %d groups (%d ngram), %.2f tok/group accepted (draft_k=%d)%s\n",
                        r.spec_groups, r.spec_ngram_groups,
                        (double)r.spec_accepted / r.spec_groups, draft_k,
                        r.spec_fallback ? " adaptive-fallback" : "");
        // Matched serving reference: llama.cpp ngram-mod on the same real
        // 1024-token repetitive prompt, 128-token continuation, TP=8 RTX
        // 6000 Ada node. Avoid printing a comparison for unrelated benches.
        if (draft_k == 64 && prompt_file.find("repetitive_1024.ids") != std::string::npos) {
            constexpr double kLlamaCppNgramTps = 126.22;
            const double speedup = r.decode_tps() / kLlamaCppNgramTps;
            std::printf("llama.cpp ref  : %.2f tok/s  (glmserve %.3fx, %s)\n",
                        kLlamaCppNgramTps, speedup, speedup > 1.0 ? "PASS" : "FAIL");
        }
    }
    engine.barrier();
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
    std::vector<float> logits;
    if (a.has("gpu")) {
        const int64_t need = static_cast<int64_t>(context.size() + draft.size()) + 2;
        GLM_CHECK(m.upload_to_gpu(need), "mtp --gpu requires a CUDA device");
        logits = m.mtp_draft_logits_gpu(context, draft);
    } else {
        logits = m.mtp_draft_logits(context, draft);
    }

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

int cmd_chunkcheck(const Args& a) {
    EngineOptions o = engine_opts(a);
    std::vector<int> prompt = parse_int_list(a.get("tokens", "3 1 4 1 5 9 2 6"));
    int k = static_cast<int>(a.get_int("k", 5));
    GLM_CHECK(!prompt.empty(), "chunkcheck requires --tokens \"id id id\"");
    if (o.max_model_len < static_cast<int64_t>(prompt.size()) + k + 1)
        o.max_model_len = static_cast<int64_t>(prompt.size()) + k + 1;

    Engine engine(o);
    Engine::ChunkCheck c = engine.check_chunk_parity(prompt, k);
    std::printf("chunkcheck: backend=%s k=%d max_abs_diff=%.3e worst_row=%d argmax_match=%s\n",
                c.gpu ? "CUDA" : "CPU", c.k, c.max_abs_diff, c.worst_row,
                c.argmax_match ? "YES" : "NO");
    std::printf("  row_diff: ");
    for (double d : c.row_diff) std::printf("%.3e ", d);
    std::printf("\n  decode : ");
    for (int t : c.dec_tokens) std::printf("%d ", t);
    std::printf("\n  chunk  : ");
    for (int t : c.chunk_tokens) std::printf("%d ", t);
    const bool exact = c.argmax_match && c.max_abs_diff == 0.0;
    std::printf("\n  RESULT: %s (exact parity %s)\n",
                c.argmax_match ? "PASS" : "FAIL", exact ? "YES" : "NO");
    return c.argmax_match ? 0 : 1;
}

int cmd_mtpcheck(const Args& a) {
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
    std::printf("mtpcheck: backend=%s steps=%d draft_k=%d groups=%d accepted=%d rejected=%d\n",
                c.gpu ? "CUDA" : "CPU", c.steps, c.draft_k, c.groups, c.accepted, c.rejected);
    std::printf("  proposed: ");
    for (int t : c.proposed_tokens) std::printf("%d ", t);
    std::printf("\n  target  : ");
    for (int t : c.target_tokens) std::printf("%d ", t);
    std::printf("\n  output  : ");
    for (int t : c.output_tokens) std::printf("%d ", t);
    if (c.gpu) {
        std::printf("\n  greedy  : ");
        for (int t : c.ref_tokens) std::printf("%d ", t);
        std::printf("\n  spec==greedy: %s", c.match ? "YES" : "NO");
    }
    std::printf("\n  RESULT: %s\n", c.match ? "PASS" : "FAIL");
    return c.match ? 0 : 1;
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
        "  glmserve load-gguf --model GGUF_FILE_OR_DIR [--touch-payloads] [--dequant-smoke]\n"
        "                    [--require-dequant-checksums] [--linear-smoke]\n"
        "                    [--require-linear-checksums]\n"
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
        if (cmd == "load-gguf") return cmd_load_gguf(a);
        if (cmd == "dump")     return cmd_dump(a);
        if (cmd == "bench")    return cmd_bench(a);
        if (cmd == "gencheck") return cmd_gencheck(a);
        if (cmd == "mtp")      return cmd_mtp(a);
        if (cmd == "mtpcheck") return cmd_mtpcheck(a);
        if (cmd == "chunkcheck") return cmd_chunkcheck(a);
        usage();
        return 1;
    } catch (const std::exception& e) {
        GLM_ERROR("%s", e.what());
        return 1;
    }
}
