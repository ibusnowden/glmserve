#include "kv_cache.hpp"
#include "common.hpp"

namespace glmserve {

KVCache::KVCache(int64_t num_layers, int64_t num_kv_heads, int64_t head_dim,
                 int64_t block_size, int64_t num_blocks, int64_t indexer_dim)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      block_size_(block_size), num_blocks_(num_blocks), indexer_dim_(indexer_dim) {
    block_stride_ = block_size_ * num_kv_heads_ * head_dim_;
    layer_stride_ = num_blocks_ * block_stride_;
    size_t total = static_cast<size_t>(num_layers_) * static_cast<size_t>(layer_stride_);
    k_.assign(total, 0.0f);
    v_.assign(total, 0.0f);
    if (indexer_dim_ > 0) {
        index_block_stride_ = block_size_ * indexer_dim_;
        index_layer_stride_ = num_blocks_ * index_block_stride_;
        size_t itotal = static_cast<size_t>(num_layers_) * static_cast<size_t>(index_layer_stride_);
        index_.assign(itotal, 0.0f);
    }
    free_list_.reserve(num_blocks_);
    for (int64_t i = num_blocks_ - 1; i >= 0; --i)
        free_list_.push_back(static_cast<int>(i));
    GLM_INFO("KV cache: %lld layers x %lld blocks x %lld tokens x %lld kv-dim = %.2f GiB (K+V)",
             (long long)num_layers_, (long long)num_blocks_, (long long)block_size_,
             (long long)(num_kv_heads_ * head_dim_), bytes() / (1024.0 * 1024.0 * 1024.0));
    if (indexer_dim_ > 0) {
        GLM_INFO("DSA indexer cache: %lld layers x %lld blocks x %lld tokens x %lld dim = %.2f MiB",
                 (long long)num_layers_, (long long)num_blocks_, (long long)block_size_,
                 (long long)indexer_dim_, index_.size() * sizeof(float) / (1024.0 * 1024.0));
    }
}

size_t KVCache::bytes() const {
    return (k_.size() + v_.size() + index_.size()) * sizeof(float);
}

int KVCache::allocate_block() {
    if (free_list_.empty()) return -1;
    int id = free_list_.back();
    free_list_.pop_back();
    return id;
}

void KVCache::free_block(int block_id) {
    if (block_id >= 0) free_list_.push_back(block_id);
}

float* KVCache::k_block(int64_t layer, int block_id) {
    return k_.data() + layer * layer_stride_ + static_cast<int64_t>(block_id) * block_stride_;
}
float* KVCache::v_block(int64_t layer, int block_id) {
    return v_.data() + layer * layer_stride_ + static_cast<int64_t>(block_id) * block_stride_;
}
float* KVCache::index_block(int64_t layer, int block_id) {
    GLM_CHECK(indexer_dim_ > 0, "DSA indexer cache not enabled");
    return index_.data() + layer * index_layer_stride_ +
           static_cast<int64_t>(block_id) * index_block_stride_;
}

SequenceKV KVCache::make_sequence(int64_t request_id) {
    SequenceKV s;
    s.request_id = request_id;
    s.cache = this;
    return s;
}

// ---------------------------------------------------------------------------
// SequenceKV
// ---------------------------------------------------------------------------
void SequenceKV::reserve(int64_t additional) {
    GLM_CHECK(cache != nullptr, "SequenceKV not attached to a cache");
    int64_t bs = cache->block_size();
    int64_t needed_tokens = length + additional;
    int64_t needed_blocks = (needed_tokens + bs - 1) / bs;
    while (static_cast<int64_t>(block_table.size()) < needed_blocks) {
        int blk = cache->allocate_block();
        GLM_CHECK(blk >= 0, "KV cache exhausted (request %lld needs %lld blocks, "
                            "cache has %lld total)",
                  (long long)request_id, (long long)needed_blocks,
                  (long long)cache->num_blocks());
        block_table.push_back(blk);
    }
}

void SequenceKV::release() {
    if (!cache) return;
    for (int blk : block_table) cache->free_block(blk);
    block_table.clear();
    length = 0;
}

float* SequenceKV::k_slot(int64_t layer, int64_t pos) {
    int64_t bs = cache->block_size();
    int phys = block_table[pos / bs];
    int64_t off = (pos % bs) * cache->kv_dim();
    return cache->k_block(layer, phys) + off;
}
float* SequenceKV::v_slot(int64_t layer, int64_t pos) {
    int64_t bs = cache->block_size();
    int phys = block_table[pos / bs];
    int64_t off = (pos % bs) * cache->kv_dim();
    return cache->v_block(layer, phys) + off;
}
float* SequenceKV::index_slot(int64_t layer, int64_t pos) {
    int64_t bs = cache->block_size();
    int phys = block_table[pos / bs];
    int64_t off = (pos % bs) * cache->indexer_dim();
    return cache->index_block(layer, phys) + off;
}
const float* SequenceKV::k_slot(int64_t layer, int64_t pos) const {
    int64_t bs = cache->block_size();
    int phys = block_table[pos / bs];
    int64_t off = (pos % bs) * cache->kv_dim();
    return cache->k_block(layer, phys) + off;
}
const float* SequenceKV::v_slot(int64_t layer, int64_t pos) const {
    int64_t bs = cache->block_size();
    int phys = block_table[pos / bs];
    int64_t off = (pos % bs) * cache->kv_dim();
    return cache->v_block(layer, phys) + off;
}
const float* SequenceKV::index_slot(int64_t layer, int64_t pos) const {
    int64_t bs = cache->block_size();
    int phys = block_table[pos / bs];
    int64_t off = (pos % bs) * cache->indexer_dim();
    return cache->index_block(layer, phys) + off;
}

}  // namespace glmserve
