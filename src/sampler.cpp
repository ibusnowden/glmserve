#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace glmserve {

int Sampler::sample(std::vector<float>& logits, const std::vector<int>& prev_tokens,
                    const SamplingParams& p) {
    const int V = static_cast<int>(logits.size());

    // --- penalties (applied on logits) ---
    if (p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f ||
        p.presence_penalty != 0.0f) {
        std::unordered_map<int, int> counts;
        for (int t : prev_tokens) if (t >= 0 && t < V) counts[t]++;
        for (auto& [tok, cnt] : counts) {
            if (p.repetition_penalty != 1.0f) {
                float& l = logits[tok];
                l = (l > 0) ? l / p.repetition_penalty : l * p.repetition_penalty;
            }
            logits[tok] -= p.frequency_penalty * static_cast<float>(cnt);
            logits[tok] -= p.presence_penalty;
        }
    }

    // --- greedy ---
    if (p.temperature <= 0.0f) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // --- temperature ---
    float inv_t = 1.0f / p.temperature;

    // Candidate index set; start with all, optionally narrow by top-k.
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);

    if (p.top_k > 0 && p.top_k < V) {
        std::partial_sort(idx.begin(), idx.begin() + p.top_k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        idx.resize(p.top_k);
    } else {
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return logits[a] > logits[b]; });
    }

    // softmax over candidates
    float maxl = logits[idx[0]];
    std::vector<float> probs(idx.size());
    float sum = 0.0f;
    for (size_t i = 0; i < idx.size(); ++i) {
        probs[i] = std::exp((logits[idx[i]] - maxl) * inv_t);
        sum += probs[i];
    }
    for (auto& pr : probs) pr /= sum;

    // --- top-p (nucleus): keep smallest prefix with cumulative prob >= top_p ---
    if (p.top_p < 1.0f) {
        float cum = 0.0f;
        size_t keep = probs.size();
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (cum >= p.top_p) { keep = i + 1; break; }
        }
        probs.resize(keep);
        idx.resize(keep);
        float s = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (auto& pr : probs) pr /= s;
    }

    // --- sample ---
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return idx[i];
    }
    return idx.back();
}

}  // namespace glmserve
