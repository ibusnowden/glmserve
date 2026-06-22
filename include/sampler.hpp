// glmserve — token sampling: greedy, temperature, top-k, top-p (nucleus),
// repetition penalty, and stop handling.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace glmserve {

struct SamplingParams {
    float  temperature = 1.0f;      // 0 => greedy
    float  top_p       = 1.0f;      // nucleus
    int    top_k       = 0;         // 0 => disabled
    float  repetition_penalty = 1.0f;
    float  frequency_penalty  = 0.0f;
    float  presence_penalty   = 0.0f;
    int    max_tokens  = 512;
    uint64_t seed      = 0;         // 0 => nondeterministic-ish default seed
    bool   ignore_eos  = false;
    std::vector<std::string> stop;  // stop strings (checked on decoded text)
    std::vector<int> stop_token_ids;
};

class Sampler {
public:
    explicit Sampler(uint64_t seed) : rng_(seed ? seed : 0x9E3779B97F4A7C15ULL) {}

    // Sample a token id from logits [vocab]. prev_tokens used for penalties.
    int sample(std::vector<float>& logits, const std::vector<int>& prev_tokens,
               const SamplingParams& p);

private:
    std::mt19937_64 rng_;
};

}  // namespace glmserve
