#include "nccl_comm.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>
#include <thread>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace glmserve {
namespace {

static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::atoi(v);
}

static int env_int_any(const std::initializer_list<const char*>& names, int fallback) {
    for (const char* n : names) {
        const char* v = std::getenv(n);
        if (v && *v) return std::atoi(v);
    }
    return fallback;
}

#ifdef GLMSERVE_CUDA
static void cuda_ok(cudaError_t e, const char* what) {
    GLM_CHECK(e == cudaSuccess, "%s failed: %s", what, cudaGetErrorString(e));
}

static void nccl_ok(ncclResult_t r, const char* what) {
    GLM_CHECK(r == ncclSuccess, "%s failed: %s", what, ncclGetErrorString(r));
}

struct NcclState {
    ncclComm_t world = nullptr;
    ncclComm_t tp = nullptr;
    cudaStream_t stream = nullptr;
    // Persistent device bounce buffer so PP send/recv can move HOST activation
    // buffers (the CPU reference forward keeps hidden states in host memory)
    // without a per-call cudaMalloc. Grows on demand; device-pointer callers
    // skip it entirely.
    float* bounce = nullptr;
    int64_t bounce_count = 0;
};

// True if `p` is a CUDA device (or managed) allocation. Unregistered host
// pointers return either an error (cleared here) or type Unregistered; both are
// treated as host so the caller stages through the bounce buffer.
static bool is_device_ptr(const void* p) {
    cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

static float* ensure_bounce(NcclState* s, int64_t count) {
    if (count > s->bounce_count) {
        if (s->bounce) cuda_ok(cudaFree(s->bounce), "cudaFree(pp bounce)");
        cuda_ok(cudaMalloc(&s->bounce, static_cast<size_t>(count) * sizeof(float)),
                "cudaMalloc(pp bounce)");
        s->bounce_count = count;
    }
    return s->bounce;
}

static int* ensure_bounce_int(NcclState* s, int64_t count) {
    return reinterpret_cast<int*>(ensure_bounce(s, count));
}

static std::string id_to_hex(const ncclUniqueId& id) {
    static const char* h = "0123456789abcdef";
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&id);
    std::string out;
    out.resize(sizeof(ncclUniqueId) * 2);
    for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
        out[2 * i] = h[p[i] >> 4];
        out[2 * i + 1] = h[p[i] & 15];
    }
    return out;
}

static bool hex_to_id(const std::string& s, ncclUniqueId* id) {
    if (s.size() < sizeof(ncclUniqueId) * 2) return false;
    unsigned char* p = reinterpret_cast<unsigned char*>(id);
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
        int a = val(s[2 * i]), b = val(s[2 * i + 1]);
        if (a < 0 || b < 0) return false;
        p[i] = static_cast<unsigned char>((a << 4) | b);
    }
    return true;
}

static ncclUniqueId rendezvous_id(int rank) {
    ncclUniqueId id;
    const char* direct = std::getenv("GLMSERVE_NCCL_ID");
    if (direct && *direct) {
        GLM_CHECK(hex_to_id(direct, &id), "GLMSERVE_NCCL_ID is not a valid ncclUniqueId hex string");
        return id;
    }

    std::string path = std::getenv("GLMSERVE_NCCL_ID_FILE")
        ? std::getenv("GLMSERVE_NCCL_ID_FILE")
        : "/tmp/glmserve_nccl_id";
    if (rank == 0) {
        nccl_ok(ncclGetUniqueId(&id), "ncclGetUniqueId");
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        GLM_CHECK(f.good(), "cannot write NCCL rendezvous id: %s", path.c_str());
        f << id_to_hex(id) << "\n";
        return id;
    }

    for (int i = 0; i < 300; ++i) {
        std::ifstream f(path, std::ios::binary);
        std::string s;
        if (f.good() && std::getline(f, s) && hex_to_id(s, &id)) return id;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    GLM_CHECK(false, "timed out waiting for NCCL rendezvous id: %s", path.c_str());
    return id;
}
#endif

}  // namespace

LayerRange partition_layers(int64_t num_layers, int pp_stage, int pp_size) {
    if (pp_size <= 1) return {0, num_layers};
    int64_t per = (num_layers + pp_size - 1) / pp_size;  // ceil split
    int64_t begin = static_cast<int64_t>(pp_stage) * per;
    int64_t end = std::min(num_layers, begin + per);
    return {begin, end};
}

DistConfig dist_config_from_env() {
    DistConfig cfg;
    cfg.rank = env_int_any({"GLMSERVE_RANK", "RANK", "SLURM_PROCID"}, cfg.rank);
    cfg.world_size = env_int_any({"GLMSERVE_WORLD_SIZE", "WORLD_SIZE", "SLURM_NTASKS"}, cfg.world_size);
    cfg.local_rank = env_int_any({"GLMSERVE_LOCAL_RANK", "LOCAL_RANK", "SLURM_LOCALID"}, cfg.local_rank);
    cfg.tp_size = env_int("GLMSERVE_TP_SIZE", cfg.world_size > 0 ? cfg.world_size : cfg.tp_size);
    cfg.pp_size = env_int("GLMSERVE_PP_SIZE", cfg.tp_size > 0 ? cfg.world_size / cfg.tp_size : cfg.pp_size);
    if (cfg.pp_size <= 0) cfg.pp_size = 1;
    GLM_CHECK(cfg.rank >= 0 && cfg.rank < cfg.world_size,
              "invalid distributed rank/world_size: rank=%d world_size=%d", cfg.rank, cfg.world_size);
    GLM_CHECK(cfg.tp_size > 0 && cfg.world_size % cfg.tp_size == 0,
              "invalid TP size %d for world_size %d", cfg.tp_size, cfg.world_size);
    GLM_CHECK(cfg.pp_size > 0 && cfg.tp_size * cfg.pp_size == cfg.world_size,
              "tp_size * pp_size must equal world_size (%d * %d != %d)",
              cfg.tp_size, cfg.pp_size, cfg.world_size);
    return cfg;
}

Communicator::Communicator(DistConfig cfg) : cfg_(cfg) {
#ifdef GLMSERVE_CUDA
    const bool force_single = std::getenv("GLMSERVE_NCCL_SINGLE") &&
                              std::getenv("GLMSERVE_NCCL_SINGLE")[0] &&
                              std::getenv("GLMSERVE_NCCL_SINGLE")[0] != '0';
    if (cfg_.world_size <= 1 && !force_single) return;
    int ndev = 0;
    cuda_ok(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    int dev = cfg_.local_rank;
    if (ndev == 1) dev = 0;  // common with Slurm --gpus-per-task=1
    GLM_CHECK(dev >= 0 && dev < ndev, "invalid local_rank/device: local_rank=%d visible_devices=%d",
              cfg_.local_rank, ndev);
    cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
    auto* s = new NcclState();
    cuda_ok(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    ncclUniqueId id = rendezvous_id(cfg_.rank);
    nccl_ok(ncclCommInitRank(&s->world, cfg_.world_size, id, cfg_.rank), "ncclCommInitRank(world)");

#if NCCL_VERSION_CODE >= 21400
    int color = cfg_.pp_stage();
    int key = cfg_.tp_rank();
    nccl_ok(ncclCommSplit(s->world, color, key, &s->tp, nullptr), "ncclCommSplit(tp)");
#else
    s->tp = s->world;
    GLM_WARN("NCCL < 2.14: TP communicator falls back to world communicator");
#endif
    state_ = s;
#endif
}

Communicator::~Communicator() {
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    if (!s) return;
    if (s->tp && s->tp != s->world) ncclCommDestroy(s->tp);
    if (s->world) ncclCommDestroy(s->world);
    if (s->bounce) cudaFree(s->bounce);
    if (s->stream) cudaStreamDestroy(s->stream);
    delete s;
    state_ = nullptr;
#endif
}

void Communicator::all_reduce_sum(float* data, int64_t count) {
    if (cfg_.tp_size <= 1) return;  // nothing to reduce
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    GLM_CHECK(s && s->tp, "NCCL communicator is not active");
    if (is_device_ptr(data)) {
        nccl_ok(ncclAllReduce(data, data, count, ncclFloat, ncclSum, s->tp, s->stream),
                "ncclAllReduce(tp)");
        cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(all_reduce)");
    } else {  // host activation buffer (CPU reference forward) -> stage round-trip
        float* b = ensure_bounce(s, count);
        cuda_ok(cudaMemcpyAsync(b, data, static_cast<size_t>(count) * sizeof(float),
                                cudaMemcpyHostToDevice, s->stream),
                "cudaMemcpyAsync(all_reduce H2D)");
        nccl_ok(ncclAllReduce(b, b, count, ncclFloat, ncclSum, s->tp, s->stream),
                "ncclAllReduce(tp)");
        cuda_ok(cudaMemcpyAsync(data, b, static_cast<size_t>(count) * sizeof(float),
                                cudaMemcpyDeviceToHost, s->stream),
                "cudaMemcpyAsync(all_reduce D2H)");
        cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(all_reduce)");
    }
#else
    (void)data; (void)count;
#endif
}

void Communicator::pipeline_send_next(const float* data, int64_t count) {
    if (cfg_.pp_size <= 1) return;
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    GLM_CHECK(s && s->world, "NCCL communicator is not active");
    int dst = cfg_.rank + cfg_.tp_size;
    GLM_CHECK(dst < cfg_.world_size, "pipeline_send_next called on last pipeline stage");
    const float* src = data;
    if (!is_device_ptr(data)) {  // host activation buffer -> stage to device
        float* b = ensure_bounce(s, count);
        cuda_ok(cudaMemcpyAsync(b, data, static_cast<size_t>(count) * sizeof(float),
                                cudaMemcpyHostToDevice, s->stream),
                "cudaMemcpyAsync(pp send H2D)");
        src = b;
    }
    nccl_ok(ncclSend(src, count, ncclFloat, dst, s->world, s->stream), "ncclSend(next)");
    cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(send)");
#else
    (void)data; (void)count;
#endif
}

void Communicator::pipeline_recv_prev(float* data, int64_t count) {
    if (cfg_.pp_size <= 1) return;
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    GLM_CHECK(s && s->world, "NCCL communicator is not active");
    int src = cfg_.rank - cfg_.tp_size;
    GLM_CHECK(src >= 0, "pipeline_recv_prev called on first pipeline stage");
    const bool host = !is_device_ptr(data);
    float* dst = host ? ensure_bounce(s, count) : data;
    nccl_ok(ncclRecv(dst, count, ncclFloat, src, s->world, s->stream), "ncclRecv(prev)");
    if (host) {  // bring the received activation back into the host buffer
        cuda_ok(cudaMemcpyAsync(data, dst, static_cast<size_t>(count) * sizeof(float),
                                cudaMemcpyDeviceToHost, s->stream),
                "cudaMemcpyAsync(pp recv D2H)");
    }
    cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(recv)");
#else
    (void)data; (void)count;
#endif
}

void Communicator::pipeline_send_next_int(const int* data, int64_t count) {
    if (cfg_.pp_size <= 1) return;
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    GLM_CHECK(s && s->world, "NCCL communicator is not active");
    int dst = cfg_.rank + cfg_.tp_size;
    GLM_CHECK(dst < cfg_.world_size, "pipeline_send_next_int called on last pipeline stage");
    const int* src = data;
    if (!is_device_ptr(data)) {
        int* b = ensure_bounce_int(s, count);
        cuda_ok(cudaMemcpyAsync(b, data, static_cast<size_t>(count) * sizeof(int),
                                cudaMemcpyHostToDevice, s->stream),
                "cudaMemcpyAsync(pp int send H2D)");
        src = b;
    }
    nccl_ok(ncclSend(src, count, ncclInt, dst, s->world, s->stream), "ncclSend(next int)");
    cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(send int)");
#else
    (void)data; (void)count;
#endif
}

void Communicator::pipeline_recv_prev_int(int* data, int64_t count) {
    if (cfg_.pp_size <= 1) return;
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    GLM_CHECK(s && s->world, "NCCL communicator is not active");
    int src = cfg_.rank - cfg_.tp_size;
    GLM_CHECK(src >= 0, "pipeline_recv_prev_int called on first pipeline stage");
    const bool host = !is_device_ptr(data);
    int* dst = host ? ensure_bounce_int(s, count) : data;
    nccl_ok(ncclRecv(dst, count, ncclInt, src, s->world, s->stream), "ncclRecv(prev int)");
    if (host) {
        cuda_ok(cudaMemcpyAsync(data, dst, static_cast<size_t>(count) * sizeof(int),
                                cudaMemcpyDeviceToHost, s->stream),
                "cudaMemcpyAsync(pp int recv D2H)");
    }
    cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(recv int)");
#else
    (void)data; (void)count;
#endif
}

void Communicator::barrier() {
#ifdef GLMSERVE_CUDA
    auto* s = static_cast<NcclState*>(state_);
    if (!s) return;
    float* d = nullptr;
    cuda_ok(cudaMalloc(&d, sizeof(float)), "cudaMalloc(barrier)");
    cuda_ok(cudaMemsetAsync(d, 0, sizeof(float), s->stream), "cudaMemsetAsync(barrier)");
    nccl_ok(ncclAllReduce(d, d, 1, ncclFloat, ncclSum, s->world, s->stream),
            "ncclAllReduce(barrier)");
    cuda_ok(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize(barrier)");
    cudaFree(d);
#endif
}

}  // namespace glmserve
