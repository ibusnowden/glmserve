// glmserve bench — decode throughput. Loads a model dir, prefills a short
// prompt, then times greedy decode of K tokens. Usage:
//   bench_decode <model_dir> [n_decode]
#include "server.hpp"
#include "common.hpp"

#include <cstdio>

using namespace glmserve;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: bench_decode <model_dir> [n_decode]\n"); return 1; }
    int k = (argc > 2) ? std::atoi(argv[2]) : 64;

    EngineOptions o;
    o.model_path = argv[1];
    o.max_model_len = k + 64;
    Engine engine(o);

    SamplingParams p;
    p.temperature = 0.0f;  // greedy
    p.max_tokens = k;

    std::vector<ChatMessage> msgs{{"user", "benchmark decode throughput please"}};
    Timer t;
    Completion c = engine.generate(msgs, p, nullptr);
    double ms = t.ms();
    std::printf("decode: %d tokens (prompt %d) in %.1f ms = %.1f tok/s  finish=%s\n",
                c.completion_tokens, c.prompt_tokens, ms,
                c.completion_tokens / (ms / 1000.0), c.finish_reason.c_str());
    return 0;
}
