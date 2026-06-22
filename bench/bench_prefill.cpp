// glmserve bench — prefill throughput. Loads a model dir and times a single
// forward over a synthetic prompt of N tokens. Usage:
//   bench_prefill <model_dir> [n_tokens]
#include "server.hpp"
#include "common.hpp"

#include <cstdio>
#include <vector>

using namespace glmserve;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: bench_prefill <model_dir> [n_tokens]\n"); return 1; }
    int n = (argc > 2) ? std::atoi(argv[2]) : 256;

    EngineOptions o;
    o.model_path = argv[1];
    o.max_model_len = n + 16;
    Engine engine(o);

    std::vector<int> prompt(n);
    int V = (int)engine.config().vocab_size;
    for (int i = 0; i < n; ++i) prompt[i] = (i * 131 + 7) % V;

    // warmup
    engine.prefill_logits(std::vector<int>(prompt.begin(), prompt.begin() + std::min(n, 8)));

    Timer t;
    auto logits = engine.prefill_logits(prompt);
    double ms = t.ms();
    std::printf("prefill: %d tokens in %.1f ms = %.1f tok/s  (vocab=%zu)\n",
                n, ms, n / (ms / 1000.0), logits.size());
    return 0;
}
