#include "gguf.hpp"
#include "common.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace glmserve {
namespace {

constexpr uint32_t GGUF_MAGIC_LE = 0x46554747u;  // "GGUF" as little-endian u32
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;

enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool is_dir(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static uint64_t file_size(const std::string& p) {
    struct stat st;
    GLM_CHECK(::stat(p.c_str(), &st) == 0, "stat failed: %s", p.c_str());
    return static_cast<uint64_t>(st.st_size);
}

template <typename T>
static T read_pod(std::ifstream& in, const std::string& path) {
    T v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    GLM_CHECK(in.good(), "short read in %s", path.c_str());
    return v;
}

static std::string read_string(std::ifstream& in, const std::string& path) {
    uint64_t n = read_pod<uint64_t>(in, path);
    GLM_CHECK(n < (1ull << 32), "unreasonable GGUF string length %llu in %s",
              (unsigned long long)n, path.c_str());
    std::string s(static_cast<size_t>(n), '\0');
    if (n) in.read(&s[0], static_cast<std::streamsize>(n));
    GLM_CHECK(in.good(), "short string read in %s", path.c_str());
    return s;
}

static uint64_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            GLM_CHECK(false, "unsupported scalar GGUF value type %u", type);
    }
}

static void skip_value_payload(std::ifstream& in, const std::string& path, uint32_t type);

static GGUFValue read_value(std::ifstream& in, const std::string& path, uint32_t type) {
    GGUFValue v;
    switch (type) {
        case GGUF_TYPE_STRING:
            v.kind = GGUFValue::Kind::kString;
            v.s = read_string(in, path);
            break;
        case GGUF_TYPE_UINT8:
            v.kind = GGUFValue::Kind::kUInt;
            v.u = read_pod<uint8_t>(in, path);
            break;
        case GGUF_TYPE_UINT16:
            v.kind = GGUFValue::Kind::kUInt;
            v.u = read_pod<uint16_t>(in, path);
            break;
        case GGUF_TYPE_UINT32:
            v.kind = GGUFValue::Kind::kUInt;
            v.u = read_pod<uint32_t>(in, path);
            break;
        case GGUF_TYPE_UINT64:
            v.kind = GGUFValue::Kind::kUInt;
            v.u = read_pod<uint64_t>(in, path);
            break;
        case GGUF_TYPE_INT8:
            v.kind = GGUFValue::Kind::kInt;
            v.i = read_pod<int8_t>(in, path);
            break;
        case GGUF_TYPE_INT16:
            v.kind = GGUFValue::Kind::kInt;
            v.i = read_pod<int16_t>(in, path);
            break;
        case GGUF_TYPE_INT32:
            v.kind = GGUFValue::Kind::kInt;
            v.i = read_pod<int32_t>(in, path);
            break;
        case GGUF_TYPE_INT64:
            v.kind = GGUFValue::Kind::kInt;
            v.i = read_pod<int64_t>(in, path);
            break;
        case GGUF_TYPE_FLOAT32:
            v.kind = GGUFValue::Kind::kFloat;
            v.f = read_pod<float>(in, path);
            break;
        case GGUF_TYPE_FLOAT64:
            v.kind = GGUFValue::Kind::kFloat;
            v.f = read_pod<double>(in, path);
            break;
        case GGUF_TYPE_BOOL:
            v.kind = GGUFValue::Kind::kBool;
            v.b = read_pod<uint8_t>(in, path) != 0;
            break;
        case GGUF_TYPE_ARRAY: {
            v.kind = GGUFValue::Kind::kArray;
            v.array_type = read_pod<uint32_t>(in, path);
            v.array_len = read_pod<uint64_t>(in, path);
            for (uint64_t i = 0; i < v.array_len; ++i) {
                skip_value_payload(in, path, v.array_type);
            }
            break;
        }
        default:
            GLM_CHECK(false, "unsupported GGUF value type %u in %s", type, path.c_str());
    }
    return v;
}

static void skip_value_payload(std::ifstream& in, const std::string& path, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        (void)read_string(in, path);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_pod<uint32_t>(in, path);
        uint64_t n = read_pod<uint64_t>(in, path);
        for (uint64_t i = 0; i < n; ++i) skip_value_payload(in, path, elem_type);
        return;
    }
    uint64_t n = scalar_size(type);
    in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    GLM_CHECK(in.good(), "short skip in %s", path.c_str());
}

static uint64_t align_up(uint64_t x, uint64_t a) {
    return ((x + a - 1) / a) * a;
}

static std::vector<std::string> discover_shards(const std::string& path) {
    if (is_dir(path)) {
        DIR* d = ::opendir(path.c_str());
        GLM_CHECK(d, "opendir failed: %s", path.c_str());
        std::vector<std::string> out;
        while (dirent* ent = ::readdir(d)) {
            std::string name = ent->d_name;
            if (!ends_with(name, ".gguf")) continue;
            std::string full = path;
            if (!full.empty() && full.back() != '/') full += "/";
            full += name;
            out.push_back(full);
        }
        ::closedir(d);
        std::sort(out.begin(), out.end());
        GLM_CHECK(!out.empty(), "no .gguf files found in %s", path.c_str());
        return out;
    }
    GLM_CHECK(file_exists(path), "GGUF path not found: %s", path.c_str());
    return {path};
}

static uint64_t num_elements(const std::vector<uint64_t>& shape) {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return n;
}

static const GGUFTensorInfo& require_tensor_shape(const GGUFModel& model,
                                                  const std::string& name,
                                                  const std::string& role,
                                                  std::vector<uint64_t> shape) {
    const GGUFTensorInfo* t = model.tensor_info(name);
    GLM_CHECK(t, "missing GGUF tensor for glmserve %s: %s", role.c_str(), name.c_str());
    GLM_CHECK(t->shape == shape, "GGUF tensor %s for %s has shape %s, expected %s",
              name.c_str(), role.c_str(), [&]() {
                  std::string s = "[";
                  for (size_t i = 0; i < t->shape.size(); ++i)
                      s += std::to_string(t->shape[i]) + (i + 1 < t->shape.size() ? "," : "");
                  return s + "]";
              }().c_str(), [&]() {
                  std::string s = "[";
                  for (size_t i = 0; i < shape.size(); ++i)
                      s += std::to_string(shape[i]) + (i + 1 < shape.size() ? "," : "");
                  return s + "]";
              }().c_str());
    return *t;
}

static bool is_quant_type(uint32_t type) {
    switch (type) {
        case 2: case 3: case 6: case 7: case 8: case 9:
        case 10: case 11: case 12: case 13: case 14: case 15:
        case 16: case 17: case 18: case 19: case 20: case 21:
        case 22: case 23: case 29: case 34: case 35: case 39:
            return true;
        default:
            return false;
    }
}

static GGUFShardInfo load_one_shard(const std::string& path, size_t shard_index) {
    std::ifstream in(path, std::ios::binary);
    GLM_CHECK(in.good(), "cannot open GGUF shard: %s", path.c_str());

    GGUFShardInfo sh;
    sh.path = path;
    sh.file_size = file_size(path);

    uint32_t magic = read_pod<uint32_t>(in, path);
    GLM_CHECK(magic == GGUF_MAGIC_LE, "%s is not a GGUF file", path.c_str());
    sh.version = read_pod<uint32_t>(in, path);
    sh.tensor_count = read_pod<uint64_t>(in, path);
    sh.metadata_count = read_pod<uint64_t>(in, path);

    uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < sh.metadata_count; ++i) {
        std::string key = read_string(in, path);
        uint32_t type = read_pod<uint32_t>(in, path);
        GGUFValue v = read_value(in, path, type);
        if (key == "general.alignment" && v.kind == GGUFValue::Kind::kUInt && v.u > 0) {
            alignment = v.u;
        }
        sh.metadata.emplace(std::move(key), std::move(v));
    }

    sh.tensors.reserve(static_cast<size_t>(sh.tensor_count));
    for (uint64_t i = 0; i < sh.tensor_count; ++i) {
        GGUFTensorInfo ti;
        ti.name = read_string(in, path);
        uint32_t ndim = read_pod<uint32_t>(in, path);
        GLM_CHECK(ndim > 0 && ndim <= 8, "tensor %s has invalid ndim=%u",
                  ti.name.c_str(), ndim);
        ti.shape.reserve(ndim);
        for (uint32_t d = 0; d < ndim; ++d) ti.shape.push_back(read_pod<uint64_t>(in, path));
        ti.type = read_pod<uint32_t>(in, path);
        ti.offset = read_pod<uint64_t>(in, path);
        ti.n_elements = num_elements(ti.shape);
        ti.n_bytes = ggml_type_size_bytes(ti.type, ti.n_elements);
        ti.shard_index = shard_index;
        sh.tensors.push_back(std::move(ti));
    }

    uint64_t pos = static_cast<uint64_t>(in.tellg());
    sh.data_offset = align_up(pos, alignment);
    GLM_CHECK(sh.data_offset <= sh.file_size, "GGUF data offset past EOF in %s",
              path.c_str());
    for (const auto& ti : sh.tensors) {
        uint64_t begin = sh.data_offset + ti.offset;
        uint64_t end = begin + ti.n_bytes;
        GLM_CHECK(begin <= end && end <= sh.file_size,
                  "tensor %s extends past GGUF shard %s (end=%llu file=%llu)",
                  ti.name.c_str(), path.c_str(), (unsigned long long)end,
                  (unsigned long long)sh.file_size);
    }
    return sh;
}

}  // namespace

std::string GGUFValue::str() const {
    std::ostringstream ss;
    switch (kind) {
        case Kind::kString: return s;
        case Kind::kUInt: ss << u; return ss.str();
        case Kind::kInt: ss << i; return ss.str();
        case Kind::kFloat: ss << f; return ss.str();
        case Kind::kBool: return b ? "true" : "false";
        case Kind::kArray:
            ss << "array(type=" << array_type << ",len=" << array_len << ")";
            return ss.str();
        default: return "";
    }
}

bool gguf_path_like(const std::string& path) {
    if (ends_with(path, ".gguf")) return true;
    if (!is_dir(path)) return false;
    DIR* d = ::opendir(path.c_str());
    if (!d) return false;
    bool found = false;
    while (dirent* ent = ::readdir(d)) {
        if (ends_with(ent->d_name, ".gguf")) {
            found = true;
            break;
        }
    }
    ::closedir(d);
    return found;
}

const char* ggml_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        case 39: return "MXFP4";
        default: return "UNKNOWN";
    }
}

uint64_t ggml_type_size_bytes(uint32_t type, uint64_t n_elements) {
    uint64_t block = 0, bytes = 0;
    switch (type) {
        case 0: block = 1; bytes = 4; break;
        case 1: block = 1; bytes = 2; break;
        case 2: block = 32; bytes = 18; break;
        case 3: block = 32; bytes = 20; break;
        case 6: block = 32; bytes = 22; break;
        case 7: block = 32; bytes = 24; break;
        case 8: block = 32; bytes = 34; break;
        case 9: block = 32; bytes = 40; break;
        case 10: block = 256; bytes = 84; break;
        case 11: block = 256; bytes = 110; break;
        case 12: block = 256; bytes = 144; break;
        case 13: block = 256; bytes = 176; break;
        case 14: block = 256; bytes = 210; break;
        case 15: block = 256; bytes = 292; break;
        case 16: block = 256; bytes = 66; break;
        case 17: block = 256; bytes = 74; break;
        case 18: block = 256; bytes = 98; break;
        case 19: block = 256; bytes = 50; break;
        case 20: block = 32; bytes = 18; break;
        case 21: block = 256; bytes = 106; break;
        case 22: block = 256; bytes = 82; break;
        case 23: block = 256; bytes = 136; break;
        case 24: block = 1; bytes = 1; break;
        case 25: block = 1; bytes = 2; break;
        case 26: block = 1; bytes = 4; break;
        case 27: block = 1; bytes = 8; break;
        case 28: block = 1; bytes = 8; break;
        case 29: block = 256; bytes = 46; break;
        case 30: block = 1; bytes = 2; break;
        case 34: block = 256; bytes = 54; break;
        case 35: block = 256; bytes = 66; break;
        case 39: block = 32; bytes = 17; break;
        default:
            GLM_CHECK(false, "unknown GGML quant type %u", type);
    }
    GLM_CHECK(n_elements % block == 0, "GGML type %s requires %llu-element blocks; got %llu elements",
              ggml_type_name(type), (unsigned long long)block,
              (unsigned long long)n_elements);
    return (n_elements / block) * bytes;
}

void GGUFModel::load(const std::string& path) {
    shards_.clear();
    tensors_.clear();
    metadata_.clear();
    total_file_bytes_ = 0;
    total_tensor_bytes_ = 0;

    std::vector<std::string> shard_paths = discover_shards(path);
    shards_.reserve(shard_paths.size());
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        GGUFShardInfo sh = load_one_shard(shard_paths[i], i);
        total_file_bytes_ += sh.file_size;
        for (const auto& kv : sh.metadata) {
            metadata_.emplace(kv.first, kv.second);
        }
        for (const GGUFTensorInfo& t : sh.tensors) {
            total_tensor_bytes_ += t.n_bytes;
            tensors_.push_back(t);
        }
        shards_.push_back(std::move(sh));
    }
}

void GGUFModel::map_payloads() {
    for (auto& sh : shards_) {
        if (sh.mapping) continue;
        int fd = ::open(sh.path.c_str(), O_RDONLY);
        GLM_CHECK(fd >= 0, "cannot open GGUF shard for mmap: %s", sh.path.c_str());
        void* m = ::mmap(nullptr, static_cast<size_t>(sh.file_size), PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) ::close(fd);
        GLM_CHECK(m != MAP_FAILED, "mmap failed for GGUF shard: %s", sh.path.c_str());
        sh.base = static_cast<const uint8_t*>(m);
        sh.fd = fd;
        sh.mapping = std::shared_ptr<void>(m, [size = sh.file_size, fd](void* p) {
            ::munmap(p, static_cast<size_t>(size));
            ::close(fd);
        });
    }
}

// Sequentially read every shard in 1 MiB chunks to populate the page cache.
// The mmap is lazy (MAP_PRIVATE); without this, the per-tensor page faults during
// weight sharding/upload read 4 KiB at a time with no readahead, which is ~100x
// slower than sequential NFS reads on a network filesystem. Reading the shard
// once sequentially warms the shared page cache so the subsequent mmap reads hit
// memory. `start`/`stride` let multiple ranks split the shards (rank r reads
// shards where index % stride == start) so 8 GPUs prefetch in parallel.
uint64_t GGUFModel::prefault_payloads(int start, int stride) const {
    const uint64_t CHUNK = 1ull << 20;  // 1 MiB
    std::vector<uint8_t> buf(CHUNK);
    uint64_t total = 0;
    for (size_t i = 0; i < shards_.size(); ++i) {
        if (stride > 1 && static_cast<int>(i) % stride != start) continue;
        int fd = ::open(shards_[i].path.c_str(), O_RDONLY);
        if (fd < 0) continue;
        off_t off = 0;
        while (off < static_cast<off_t>(shards_[i].file_size)) {
            ssize_t n = ::read(fd, buf.data(), CHUNK);
            if (n <= 0) break;
            off += n;
            total += static_cast<uint64_t>(n);
        }
        ::close(fd);
    }
    return total;
}

void GGUFModel::evict_all() const {
    for (const auto& sh : shards_) {
        if (sh.base)
            ::madvise(const_cast<uint8_t*>(sh.base), static_cast<size_t>(sh.file_size),
                      MADV_DONTNEED);
        if (sh.fd >= 0)
            ::posix_fadvise(sh.fd, 0, static_cast<off_t>(sh.file_size),
                            POSIX_FADV_DONTNEED);
    }
}

void GGUFModel::evict_range(const uint8_t* p, size_t n) const {
    if (!p || n == 0) return;
    for (const auto& sh : shards_) {
        if (!sh.base || p < sh.base || p >= sh.base + sh.file_size) continue;
        if (n > static_cast<size_t>(sh.base + sh.file_size - p))
            n = static_cast<size_t>(sh.base + sh.file_size - p);
        // Interior pages only: a boundary page may carry bytes of a neighboring
        // tensor that is still host-resident (e.g. the TP embed table).
        const size_t PAGE = 4096;
        uintptr_t a = (reinterpret_cast<uintptr_t>(p) + PAGE - 1) & ~(PAGE - 1);
        uintptr_t e = (reinterpret_cast<uintptr_t>(p) + n) & ~(PAGE - 1);
        if (e <= a) return;
        ::madvise(reinterpret_cast<void*>(a), e - a, MADV_DONTNEED);
        if (sh.fd >= 0) {
            off_t off = static_cast<off_t>(a - reinterpret_cast<uintptr_t>(sh.base));
            ::posix_fadvise(sh.fd, off, static_cast<off_t>(e - a), POSIX_FADV_DONTNEED);
        }
        return;
    }
}

const uint8_t* GGUFModel::tensor_data(const GGUFTensorInfo& tensor) const {
    GLM_CHECK(tensor.shard_index < shards_.size(), "tensor %s has invalid shard index %zu",
              tensor.name.c_str(), tensor.shard_index);
    const GGUFShardInfo& sh = shards_[tensor.shard_index];
    GLM_CHECK(sh.base != nullptr, "GGUF payloads are not mapped; call map_payloads() first");
    uint64_t begin = sh.data_offset + tensor.offset;
    uint64_t end = begin + tensor.n_bytes;
    GLM_CHECK(begin <= end && end <= sh.file_size, "tensor %s points outside mapped shard",
              tensor.name.c_str());
    return sh.base + begin;
}

uint64_t GGUFModel::touch_payloads() {
    map_payloads();
    uint64_t checksum = 1469598103934665603ull;  // FNV offset basis
    for (const auto& t : tensors_) {
        if (t.n_bytes == 0) continue;
        const uint8_t* p = tensor_data(t);
        volatile uint8_t first = p[0];
        volatile uint8_t last = p[t.n_bytes - 1];
        checksum ^= static_cast<uint64_t>(first);
        checksum *= 1099511628211ull;
        checksum ^= static_cast<uint64_t>(last);
        checksum *= 1099511628211ull;
    }
    return checksum;
}

std::string GGUFModel::metadata_string(const std::string& key, const std::string& def) const {
    auto it = metadata_.find(key);
    return it == metadata_.end() ? def : it->second.str();
}

uint64_t GGUFModel::metadata_u64(const std::string& key, uint64_t def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const GGUFValue& v = it->second;
    if (v.kind == GGUFValue::Kind::kUInt) return v.u;
    if (v.kind == GGUFValue::Kind::kInt && v.i >= 0) return static_cast<uint64_t>(v.i);
    return def;
}

bool GGUFModel::has_tensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return true;
    }
    return false;
}

const GGUFTensorInfo* GGUFModel::tensor_info(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

GGUFGLM52Layout GGUFModel::build_glm52_layout() const {
    validate_glm52();
    GGUFGLM52Layout layout;
    auto add = [&](const std::string& role, const std::string& name,
                   std::vector<uint64_t> shape) {
        const GGUFTensorInfo& t = require_tensor_shape(*this, name, role, std::move(shape));
        layout.modules.push_back({role, name, &t});
        layout.tensor_bytes += t.n_bytes;
        if (is_quant_type(t.type)) {
            layout.quantized_tensors += 1;
            layout.quantized_tensor_bytes += t.n_bytes;
        }
    };

    const uint64_t H = 6144;
    const uint64_t V = 154880;
    const uint64_t L = metadata_u64("glm-dsa.block_count");
    const uint64_t heads = 64;
    const uint64_t q_lora = 2048;
    const uint64_t kv_lora = 512;
    const uint64_t qk_head = 256;
    const uint64_t qk_nope = 192;
    const uint64_t qk_rope = 64;
    const uint64_t v_head = 256;
    const uint64_t dense_ffn = 12288;
    const uint64_t experts = 256;
    const uint64_t moe_ffn = 2048;
    const uint64_t index_dim = 128;
    const uint64_t index_heads = 32;

    add("model.embed_tokens.weight", "token_embd.weight", {H, V});
    add("model.norm.weight", "output_norm.weight", {H});
    add("lm_head.weight", "output.weight", {H, V});

    for (uint64_t i = 0; i < L; ++i) {
        const std::string hf = "model.layers." + std::to_string(i) + ".";
        const std::string gg = "blk." + std::to_string(i) + ".";
        add(hf + "input_layernorm.weight", gg + "attn_norm.weight", {H});
        add(hf + "post_attention_layernorm.weight", gg + "ffn_norm.weight", {H});
        add(hf + "self_attn.q_a_proj.weight", gg + "attn_q_a.weight", {H, q_lora});
        add(hf + "self_attn.q_a_layernorm.weight", gg + "attn_q_a_norm.weight", {q_lora});
        add(hf + "self_attn.q_b_proj.weight", gg + "attn_q_b.weight", {q_lora, heads * qk_head});
        add(hf + "self_attn.kv_a_proj_with_mqa.weight", gg + "attn_kv_a_mqa.weight",
            {H, kv_lora + qk_rope});
        add(hf + "self_attn.kv_a_layernorm.weight", gg + "attn_kv_a_norm.weight", {kv_lora});
        add(hf + "self_attn.kv_b_proj.weight[k]", gg + "attn_k_b.weight",
            {qk_nope, kv_lora, heads});
        add(hf + "self_attn.kv_b_proj.weight[v]", gg + "attn_v_b.weight",
            {kv_lora, v_head, heads});
        add(hf + "self_attn.o_proj.weight", gg + "attn_output.weight", {heads * v_head, H});

        add(hf + "self_attn.indexer.wq_b.weight", gg + "indexer.attn_q_b.weight",
            {q_lora, index_heads * index_dim});
        add(hf + "self_attn.indexer.wk.weight", gg + "indexer.attn_k.weight", {H, index_dim});
        add(hf + "self_attn.indexer.weights_proj.weight", gg + "indexer.proj.weight",
            {H, index_heads});
        add(hf + "self_attn.indexer.k_norm.weight", gg + "indexer.k_norm.weight", {index_dim});
        add(hf + "self_attn.indexer.k_norm.bias", gg + "indexer.k_norm.bias", {index_dim});

        if (i < 3) {
            layout.dense_layers += 1;
            add(hf + "mlp.gate_proj.weight", gg + "ffn_gate.weight", {H, dense_ffn});
            add(hf + "mlp.up_proj.weight", gg + "ffn_up.weight", {H, dense_ffn});
            add(hf + "mlp.down_proj.weight", gg + "ffn_down.weight", {dense_ffn, H});
        } else {
            layout.moe_layers += 1;
            add(hf + "mlp.gate.weight", gg + "ffn_gate_inp.weight", {H, experts});
            add(hf + "mlp.gate.e_score_correction_bias", gg + "exp_probs_b.bias", {experts});
            add(hf + "mlp.experts.*.gate_proj.weight", gg + "ffn_gate_exps.weight",
                {H, moe_ffn, experts});
            add(hf + "mlp.experts.*.up_proj.weight", gg + "ffn_up_exps.weight",
                {H, moe_ffn, experts});
            add(hf + "mlp.experts.*.down_proj.weight", gg + "ffn_down_exps.weight",
                {moe_ffn, H, experts});
            add(hf + "mlp.shared_experts.gate_proj.weight", gg + "ffn_gate_shexp.weight",
                {H, moe_ffn});
            add(hf + "mlp.shared_experts.up_proj.weight", gg + "ffn_up_shexp.weight",
                {H, moe_ffn});
            add(hf + "mlp.shared_experts.down_proj.weight", gg + "ffn_down_shexp.weight",
                {moe_ffn, H});
        }
    }

    if (L > 78) {
        const std::string hf = "model.layers.78.";
        const std::string gg = "blk.78.nextn.";
        layout.has_mtp = true;
        add(hf + "eh_proj.weight", gg + "eh_proj.weight", {2 * H, H});
        add(hf + "enorm.weight", gg + "enorm.weight", {H});
        add(hf + "hnorm.weight", gg + "hnorm.weight", {H});
        add(hf + "shared_head.norm.weight", gg + "shared_head_norm.weight", {H});
    }
    return layout;
}

std::map<std::string, size_t> GGUFModel::quant_counts() const {
    std::map<std::string, size_t> out;
    for (const auto& t : tensors_) out[ggml_type_name(t.type)] += 1;
    return out;
}

void GGUFModel::validate_glm52() const {
    std::string arch = metadata_string("general.architecture");
    uint64_t block_count = metadata_u64("glm-dsa.block_count");
    uint64_t experts = metadata_u64("glm-dsa.expert_count");
    GLM_CHECK(arch == "glm-dsa", "expected general.architecture=glm-dsa, got %s",
              arch.c_str());
    GLM_CHECK(block_count >= 78, "expected at least 78 GLM blocks, got %llu",
              (unsigned long long)block_count);
    GLM_CHECK(experts == 256, "expected 256 experts, got %llu",
              (unsigned long long)experts);
    GLM_CHECK(!tensors_.empty(), "GGUF model has no tensor payloads");

    const char* global[] = {"token_embd.weight", "output_norm.weight", "output.weight"};
    for (const char* name : global) {
        GLM_CHECK(has_tensor(name), "missing required GLM-5.2 GGUF tensor: %s", name);
    }

    auto q = quant_counts();
    GLM_CHECK(q["IQ3_XXS"] > 0, "expected real 3-bit IQ3_XXS tensors in GLM-5.2 GGUF");

    for (uint64_t i = 0; i < block_count; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        const char* suffixes[] = {
            "attn_norm.weight",
            "attn_q_a.weight",
            "attn_q_a_norm.weight",
            "attn_q_b.weight",
            "attn_kv_a_mqa.weight",
            "attn_kv_a_norm.weight",
            "attn_output.weight",
            "ffn_norm.weight",
        };
        for (const char* suffix : suffixes) {
            std::string name = p + suffix;
            GLM_CHECK(has_tensor(name), "missing required GLM-5.2 GGUF tensor: %s",
                      name.c_str());
        }
    }
}

void GGUFModel::validate_glmserve_mapping() const {
    (void)build_glm52_layout();
}

}  // namespace glmserve
