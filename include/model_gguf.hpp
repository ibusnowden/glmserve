// glmserve — GLM-5.2 GGUF weight views.
//
// This bridges the GGUF tensor table into the GLM52Model runtime: it owns the
// mmap-backed GGUF model and exposes stable, role-addressed tensor views for
// the real quantized payloads. Execution kernels are wired separately.
#pragma once

#include "config.hpp"
#include "gguf.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace glmserve {

struct GGUFWeightView {
    std::string role;
    std::string name;
    const GGUFTensorInfo* tensor = nullptr;
    const uint8_t* data = nullptr;
};

struct GGUFLinearView {
    const GGUFWeightView* weight = nullptr;
    uint64_t in_features = 0;
    uint64_t out_features = 0;
    uint64_t row_bytes = 0;
    bool valid() const { return weight != nullptr && in_features > 0 && out_features > 0; }
};

class GLM52GGUFWeights {
public:
    void load(const std::string& path, const GLM52Config& cfg, int64_t max_layers = -1,
              bool touch_payloads = false);

    const GGUFModel& gguf() const { return gguf_; }
    const GGUFGLM52Layout& layout() const { return layout_; }
    const std::vector<GGUFWeightView>& views() const { return views_; }

    const GGUFWeightView* role(const std::string& role) const;
    GGUFLinearView linear(const std::string& role) const;
    // Warm the page cache for exactly the payload ranges this model loaded
    // (views_): ranges are merged per shard, split into large chunks, and
    // striped across ranks (chunk i read by rank i % world). The page cache is
    // node-shared, so after all local ranks finish their stripes every rank's
    // mmap reads hit memory instead of 4 KiB random NFS faults. Returns bytes
    // read by this rank.
    uint64_t prefault(int rank = 0, int world = 1) const;
    bool ready() const { return ready_; }
    uint64_t payload_checksum() const { return payload_checksum_; }
    uint64_t mapped_payload_bytes() const { return mapped_payload_bytes_; }

private:
    GGUFModel gguf_;
    GGUFGLM52Layout layout_;
    std::vector<GGUFWeightView> views_;
    std::map<std::string, size_t> role_index_;
    uint64_t payload_checksum_ = 0;
    uint64_t mapped_payload_bytes_ = 0;
    bool ready_ = false;
};

// Build a GLM52Config from a split GGUF's metadata (general.architecture=glm-dsa).
// Used when serving directly from a GGUF checkpoint (no config.json on disk):
// the GGUF carries the full architecture under glm-dsa.* keys. Light parse —
// opens the shards for metadata/tensor tables only (no payload mmap).
GLM52Config load_glm52_config_gguf(const std::string& gguf_path);

}  // namespace glmserve
