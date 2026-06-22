// glmserve — safetensors loader (single-file or sharded via index.json).
//
// Memory-maps each shard read-only and exposes tensors as zero-copy views
// (Tensor keeps the mapping alive). This is the M0 deliverable: read config,
// list every tensor, map names to engine modules, load selected tensors.
#pragma once

#include "tensor.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace glmserve {

struct TensorInfo {
    std::string name;
    DType dtype = DType::kUnknown;
    std::vector<int64_t> shape;
    size_t shard_index = 0;     // which mapped shard
    size_t begin = 0, end = 0;  // byte offsets within that shard's data block
    int64_t numel() const { return Tensor::num_elements(shape); }
};

class SafeTensors {
public:
    // path may be a directory (auto-detects model.safetensors[.index.json])
    // or a direct .safetensors file.
    void load(const std::string& path);

    bool has(const std::string& name) const { return index_.count(name) > 0; }

    // Returns a zero-copy view; throws if missing.
    Tensor get(const std::string& name) const;

    // Returns nullptr-equivalent (invalid Tensor) if missing.
    Tensor try_get(const std::string& name) const;

    const TensorInfo* info(const std::string& name) const {
        auto it = index_.find(name);
        return it == index_.end() ? nullptr : &it->second;
    }

    std::vector<std::string> names() const;
    size_t size() const { return index_.size(); }
    size_t total_bytes() const { return total_bytes_; }

private:
    struct Shard {
        std::string path;
        std::shared_ptr<void> mapping;  // mmap base, munmap on release
        const uint8_t* base = nullptr;  // start of mmap
        size_t file_size = 0;
        size_t data_offset = 0;         // 8 + header_len: start of tensor data
    };

    size_t load_shard(const std::string& file);  // returns shard index
    void parse_header(size_t shard_idx);

    std::vector<Shard> shards_;
    std::map<std::string, TensorInfo> index_;
    size_t total_bytes_ = 0;
};

}  // namespace glmserve
