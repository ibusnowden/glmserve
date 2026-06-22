#include "scheduler.hpp"

namespace glmserve {

int64_t Scheduler::admit() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return active_.load() < max_concurrent_; });
    active_.fetch_add(1);
    int64_t id = next_id_.fetch_add(1);
    cancelled_[id] = false;
    return id;
}

void Scheduler::complete(int64_t id) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancelled_.erase(id);
        active_.fetch_sub(1);
    }
    cv_.notify_one();
}

void Scheduler::cancel(int64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cancelled_.find(id);
    if (it != cancelled_.end()) it->second = true;
}

bool Scheduler::is_cancelled(int64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cancelled_.find(id);
    return it != cancelled_.end() && it->second;
}

}  // namespace glmserve
