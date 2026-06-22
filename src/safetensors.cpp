#include "safetensors.hpp"
#include "common.hpp"
#include "json.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace glmserve {

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GLM_CHECK(f.good(), "cannot open %s", path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

size_t SafeTensors::load_shard(const std::string& file) {
    int fd = ::open(file.c_str(), O_RDONLY);
    GLM_CHECK(fd >= 0, "cannot open safetensors shard: %s", file.c_str());

    struct stat st;
    GLM_CHECK(::fstat(fd, &st) == 0, "fstat failed: %s", file.c_str());
    size_t fsize = static_cast<size_t>(st.st_size);
    GLM_CHECK(fsize >= 8, "shard too small: %s", file.c_str());

    void* m = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    GLM_CHECK(m != MAP_FAILED, "mmap failed: %s", file.c_str());

    Shard sh;
    sh.path = file;
    sh.file_size = fsize;
    sh.base = static_cast<const uint8_t*>(m);
    sh.mapping = std::shared_ptr<void>(m, [fsize](void* p) { ::munmap(p, fsize); });

    shards_.push_back(std::move(sh));
    size_t idx = shards_.size() - 1;
    parse_header(idx);
    return idx;
}

void SafeTensors::parse_header(size_t shard_idx) {
    Shard& sh = shards_[shard_idx];
    uint64_t header_len = 0;
    std::memcpy(&header_len, sh.base, 8);  // little-endian u64
    GLM_CHECK(8 + header_len <= sh.file_size, "corrupt header length in %s",
              sh.path.c_str());
    sh.data_offset = 8 + header_len;

    std::string header_json(reinterpret_cast<const char*>(sh.base + 8), header_len);
    auto root = json::parse(header_json);
    GLM_CHECK(root && root->is_object(), "safetensors header is not an object");

    for (auto& [name, v] : root->obj) {
        if (name == "__metadata__") continue;
        GLM_CHECK(v->is_object(), "tensor entry %s is not an object", name.c_str());

        TensorInfo ti;
        ti.name = name;
        ti.dtype = dtype_from_string(v->get_string("dtype"));
        GLM_CHECK(ti.dtype != DType::kUnknown, "tensor %s has unknown dtype %s",
                  name.c_str(), v->get_string("dtype").c_str());

        auto shape = v->at("shape");
        GLM_CHECK(shape && shape->is_array(), "tensor %s missing shape", name.c_str());
        for (auto& d : shape->arr) ti.shape.push_back(d->as_int());
        if (ti.shape.empty()) ti.shape.push_back(1);  // scalar -> [1]

        auto off = v->at("data_offsets");
        GLM_CHECK(off && off->is_array() && off->arr.size() == 2,
                  "tensor %s missing data_offsets", name.c_str());
        ti.begin = static_cast<size_t>(off->arr[0]->as_int());
        ti.end   = static_cast<size_t>(off->arr[1]->as_int());
        ti.shard_index = shard_idx;

        GLM_CHECK(sh.data_offset + ti.end <= sh.file_size,
                  "tensor %s extends past shard %s", name.c_str(), sh.path.c_str());

        total_bytes_ += (ti.end - ti.begin);
        index_[name] = std::move(ti);
    }
}

void SafeTensors::load(const std::string& path) {
    struct stat st;
    bool is_dir = (::stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);

    if (!is_dir && ends_with(path, ".safetensors")) {
        load_shard(path);
        GLM_INFO("loaded safetensors: %s (%zu tensors, %.2f GiB)",
                 path.c_str(), index_.size(), total_bytes_ / (1024.0 * 1024.0 * 1024.0));
        return;
    }

    std::string dir = path;
    if (!dir.empty() && dir.back() != '/') dir += '/';

    std::string index_json = dir + "model.safetensors.index.json";
    std::string single     = dir + "model.safetensors";

    if (file_exists(index_json)) {
        // Sharded: collect the set of shard files from weight_map.
        auto root = json::parse(read_text(index_json));
        GLM_CHECK(root && root->has("weight_map"), "index.json missing weight_map");
        std::set<std::string> shard_files;
        for (auto& [tensor_name, file_v] : root->at("weight_map")->obj) {
            (void)tensor_name;
            shard_files.insert(file_v->as_string());
        }
        for (const auto& f : shard_files) load_shard(dir + f);
        GLM_INFO("loaded sharded safetensors: %zu shards, %zu tensors, %.2f GiB",
                 shard_files.size(), index_.size(),
                 total_bytes_ / (1024.0 * 1024.0 * 1024.0));
    } else if (file_exists(single)) {
        load_shard(single);
        GLM_INFO("loaded safetensors: %s (%zu tensors, %.2f GiB)",
                 single.c_str(), index_.size(),
                 total_bytes_ / (1024.0 * 1024.0 * 1024.0));
    } else {
        GLM_CHECK(false, "no model.safetensors[.index.json] found in %s", dir.c_str());
    }
}

Tensor SafeTensors::get(const std::string& name) const {
    auto it = index_.find(name);
    GLM_CHECK(it != index_.end(), "tensor not found: %s", name.c_str());
    const TensorInfo& ti = it->second;
    const Shard& sh = shards_[ti.shard_index];
    void* ptr = const_cast<uint8_t*>(sh.base + sh.data_offset + ti.begin);
    return Tensor(ti.dtype, ti.shape, ptr, sh.mapping);
}

Tensor SafeTensors::try_get(const std::string& name) const {
    if (!has(name)) return Tensor();
    return get(name);
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(index_.size());
    for (auto& [n, _] : index_) out.push_back(n);
    return out;
}

}  // namespace glmserve
