// glmserve — minimal GGUF metadata/tensor-table reader.
//
// This reader is deliberately small: it opens split GGUF files, parses metadata
// and tensor descriptors, and computes byte sizes from GGML quant block sizes.
// It does not dequantize or execute tensors yet.  Its first job is making the
// glmserve binary validate the real GLM-5.2 UD-Q3_K_XL weight set used by the
// llama.cpp reference stack.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace glmserve {

struct GGUFValue {
    enum class Kind { kString, kUInt, kInt, kFloat, kBool, kArray, kUnknown };
    Kind kind = Kind::kUnknown;
    std::string s;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    uint32_t array_type = 0;
    uint64_t array_len = 0;

    std::string str() const;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;       // relative to this shard's aligned data start
    uint64_t n_elements = 0;
    uint64_t n_bytes = 0;
    size_t shard_index = 0;
};

struct GGUFModuleView {
    std::string role;   // glmserve/HF-style logical role
    std::string name;   // GGUF tensor name
    const GGUFTensorInfo* tensor = nullptr;
};

struct GGUFGLM52Layout {
    std::vector<GGUFModuleView> modules;
    uint64_t tensor_bytes = 0;
    uint64_t quantized_tensor_bytes = 0;
    size_t quantized_tensors = 0;
    size_t dense_layers = 0;
    size_t moe_layers = 0;
    bool has_mtp = false;
};

struct GGUFShardInfo {
    std::string path;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    uint64_t file_size = 0;
    uint64_t data_offset = 0;
    std::shared_ptr<void> mapping;
    const uint8_t* base = nullptr;
    int fd = -1;  // kept open for posix_fadvise; closed with the mapping
    std::map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;
};

class GGUFModel {
public:
    void load(const std::string& path);

    const std::vector<GGUFShardInfo>& shards() const { return shards_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, GGUFValue>& metadata() const { return metadata_; }

    uint64_t total_file_bytes() const { return total_file_bytes_; }
    uint64_t total_tensor_bytes() const { return total_tensor_bytes_; }

    void map_payloads();
    const uint8_t* tensor_data(const GGUFTensorInfo& tensor) const;
    uint64_t touch_payloads();
    // Sequentially read every shard (or a strided subset for multi-rank parallel
    // prefetch) in 1 MiB chunks to warm the shared page cache before the lazy
    // mmap reads. Returns total bytes read.
    uint64_t prefault_payloads(int start = 0, int stride = 1) const;
    // Drop an already-consumed payload range from this process's page tables
    // (madvise DONTNEED) and from the shared page cache (posix_fadvise
    // DONTNEED). Shrinks inward to page boundaries so neighboring tensors'
    // pages survive. No-op for pointers outside the mapped shards.
    void evict_range(const uint8_t* p, size_t n) const;
    // Drop every shard's pages from this process and the page cache. Run
    // before a sequential upload on a node whose RAM is already full of stale
    // GGUF cache (e.g. from an earlier killed job): faulting into a full page
    // cache pays synchronous direct-reclaim latency per page and collapses to
    // random-fault NFS reads, so starting from a clean cache is faster even
    // though warm ranges are re-read.
    void evict_all() const;

    bool has_metadata(const std::string& key) const { return metadata_.count(key) != 0; }
    bool has_tensor(const std::string& name) const;
    const GGUFTensorInfo* tensor_info(const std::string& name) const;
    std::string metadata_string(const std::string& key, const std::string& def = "") const;
    uint64_t metadata_u64(const std::string& key, uint64_t def = 0) const;

    GGUFGLM52Layout build_glm52_layout() const;
    std::map<std::string, size_t> quant_counts() const;
    void validate_glm52() const;
    void validate_glmserve_mapping() const;

private:
    std::vector<GGUFShardInfo> shards_;
    std::vector<GGUFTensorInfo> tensors_;
    std::map<std::string, GGUFValue> metadata_;  // first shard wins
    uint64_t total_file_bytes_ = 0;
    uint64_t total_tensor_bytes_ = 0;
};

bool gguf_path_like(const std::string& path);
const char* ggml_type_name(uint32_t type);
uint64_t ggml_type_size_bytes(uint32_t type, uint64_t n_elements);

}  // namespace glmserve
