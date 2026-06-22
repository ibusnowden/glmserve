// glmserve bench — paged KV cache: block allocation/free throughput and
// per-token slot-write bandwidth. Pure CPU; runnable anywhere.
#include "kv_cache.hpp"
#include "common.hpp"

#include <cstdio>
#include <vector>

using namespace glmserve;

int main(int argc, char** argv) {
    int64_t layers = 78, kv_heads = 8, head_dim = 128, block_size = 16;
    int64_t blocks = (argc > 1) ? std::atoll(argv[1]) : 4096;

    Timer t;
    KVCache cache(layers, kv_heads, head_dim, block_size, blocks);
    std::printf("alloc %lld layers x %lld blocks (%.2f GiB) in %.1f ms\n",
                (long long)layers, (long long)blocks,
                cache.bytes() / (1024.0 * 1024.0 * 1024.0), t.ms());

    // allocate/free churn
    t.reset();
    const int iters = 100000;
    for (int i = 0; i < iters; ++i) {
        int b = cache.allocate_block();
        cache.free_block(b);
    }
    double ms = t.ms();
    std::printf("alloc+free: %d iters in %.1f ms = %.1f M ops/s\n",
                iters, ms, iters / ms / 1000.0);

    // slot-write bandwidth across one sequence
    SequenceKV seq = cache.make_sequence(1);
    int64_t toks = block_size * 200;
    seq.reserve(toks);
    std::vector<float> row(kv_heads * head_dim, 1.0f);
    t.reset();
    for (int64_t p = 0; p < toks; ++p)
        for (int64_t l = 0; l < layers; ++l) {
            std::memcpy(seq.k_slot(l, p), row.data(), row.size() * sizeof(float));
            std::memcpy(seq.v_slot(l, p), row.data(), row.size() * sizeof(float));
        }
    ms = t.ms();
    double gb = (double)toks * layers * 2 * row.size() * sizeof(float) / 1e9;
    std::printf("slot writes: %lld tok x %lld layers = %.2f GB in %.1f ms = %.1f GB/s\n",
                (long long)toks, (long long)layers, gb, ms, gb / (ms / 1000.0));
    seq.release();
    return 0;
}
