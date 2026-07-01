// glmserve — paged KV cache.
//
// Memory is carved into fixed-size blocks (block_size tokens each). A sequence
// owns a block table (logical block -> physical block id), so its KV storage
// need not be contiguous — this is what lets long, variable-length prompts
// coexist without fragmentation. The reference implementation stores float32 KV
// in host memory; the GPU path mirrors this with device buffers and an optional
// FP8 KV dtype (same block-table logic).
//
//   layout per (layer, physical block):
//     K: [block_size, num_kv_heads, head_dim]   (row-major)
//     V: [block_size, num_kv_heads, head_dim]
//     I: [block_size, indexer_dim]              (optional DSA indexer keys)
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glmserve {

class KVCache;

// Per-sequence view into the KV cache.
struct SequenceKV {
    int64_t request_id = 0;
    std::vector<int> block_table;  // logical block index -> physical block id
    int64_t length = 0;            // number of tokens currently cached
    KVCache* cache = nullptr;

    // Ensure room for `additional` more tokens, allocating blocks as needed.
    void reserve(int64_t additional);
    // Free all blocks back to the pool.
    void release();

    // Pointers to the K/V slot for an absolute token position at a given layer.
    float* k_slot(int64_t layer, int64_t pos);
    float* v_slot(int64_t layer, int64_t pos);
    float* index_slot(int64_t layer, int64_t pos);
    const float* k_slot(int64_t layer, int64_t pos) const;
    const float* v_slot(int64_t layer, int64_t pos) const;
    const float* index_slot(int64_t layer, int64_t pos) const;
};

class KVCache {
public:
    KVCache(int64_t num_layers, int64_t num_kv_heads, int64_t head_dim,
            int64_t block_size, int64_t num_blocks, int64_t indexer_dim = 0);

    int64_t block_size() const { return block_size_; }
    int64_t num_free()   const { return static_cast<int64_t>(free_list_.size()); }
    int64_t num_blocks() const { return num_blocks_; }
    int64_t kv_dim()     const { return num_kv_heads_ * head_dim_; }
    int64_t num_kv_heads() const { return num_kv_heads_; }
    int64_t head_dim()   const { return head_dim_; }
    int64_t indexer_dim() const { return indexer_dim_; }

    // Allocate / return one physical block. allocate() returns -1 if exhausted.
    int allocate_block();
    void free_block(int block_id);

    // Raw pointers into the per-layer K/V pools for a physical block.
    float* k_block(int64_t layer, int block_id);
    float* v_block(int64_t layer, int block_id);
    float* index_block(int64_t layer, int block_id);

    // Total bytes the cache occupies (both K and V), for reporting.
    size_t bytes() const;

    SequenceKV make_sequence(int64_t request_id);

private:
    int64_t num_layers_, num_kv_heads_, head_dim_, block_size_, num_blocks_, indexer_dim_;
    int64_t block_stride_;   // floats per (layer, block) = block_size*kv_dim
    int64_t layer_stride_;   // floats per layer = num_blocks*block_stride_
    int64_t index_block_stride_ = 0;
    int64_t index_layer_stride_ = 0;
    std::vector<float> k_;   // [num_layers, num_blocks, block_size, kv_dim]
    std::vector<float> v_;
    std::vector<float> index_; // [num_layers, num_blocks, block_size, indexer_dim]
    std::vector<int> free_list_;
};

}  // namespace glmserve
