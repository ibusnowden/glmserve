// glmserve — request scheduler.
//
// V0 admits up to `max_concurrent` requests (1 for coding-agent bring-up) and
// serializes the rest. It owns request lifecycle: admission gating, per-request
// cancellation, and active-count accounting. This is the seam where continuous
// decode batching (V2+) will later interleave multiple sequences; the public
// API is intentionally stable so the engine above it need not change.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace glmserve {

class Scheduler {
public:
    explicit Scheduler(int max_concurrent = 1) : max_concurrent_(max_concurrent) {}

    // Blocks until a slot is free; returns a unique request id.
    int64_t admit();

    // Release the slot held by `id`.
    void complete(int64_t id);

    // Request cooperative cancellation (engine checks is_cancelled()).
    void cancel(int64_t id);
    bool is_cancelled(int64_t id);

    int active() const { return active_.load(); }
    int64_t total_admitted() const { return next_id_.load(); }

private:
    int max_concurrent_;
    std::atomic<int> active_{0};
    std::atomic<int64_t> next_id_{0};
    std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<int64_t, bool> cancelled_;
};

}  // namespace glmserve
