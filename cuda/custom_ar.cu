// glmserve — one-shot P2P all-reduce for decode-sized messages (custom_ar.h).
//
// Staging layout per rank (identical on every rank, so a peer base pointer +
// the shared constants locate any slot):
//   floats [0, 2*world*kArMaxFloats)                  double-buffered data
//          slot(parity, writer) = ((parity*world)+writer) * kArMaxFloats
//   flags  (unsigned, after the data region)          per-(parity,writer,block)
//          flag(parity, writer, block) = ((parity*world)+writer)*kArMaxBlocks + block
//
// Protocol per call (seq increments monotonically, parity = seq & 1):
//   push:    every block grid-strides its float4 range into slot(parity, rank)
//            of ALL peers, fences, then lane t < world publishes flag = seq on
//            peer t.
//   reduce:  each block spins on its LOCAL flags for all writers, fences, then
//            sums slot(parity, 0..world-1) over its own range in fixed order.
// Double buffering makes slot reuse safe: rank W can only reach call N+2
// (same parity as N) after every rank acknowledged N+1, which requires them
// to have finished N.
#include "custom_ar.h"

#include <cstdio>
#include <cstring>

namespace glmserve {
namespace cuda {

namespace {

constexpr int kArMaxWorld = 8;
constexpr int64_t kArMaxFloats = 8 * 6144;  // widest short verify chunk x hidden
constexpr int kArMaxBlocks = 32;
constexpr int kArThreads = 256;

constexpr int64_t data_floats_total(int world) {
    return 2LL * world * kArMaxFloats;
}
constexpr int64_t flags_count(int world) {
    return 2LL * world * kArMaxBlocks;
}
constexpr int64_t alloc_bytes(int world) {
    return data_floats_total(world) * (int64_t)sizeof(float) +
           flags_count(world) * (int64_t)sizeof(unsigned);
}

struct ArPeers {
    float* base[kArMaxWorld];
};

__global__ void one_shot_ar_kernel(ArPeers peers, float* __restrict__ data,
                                   int64_t count4, int rank, int world,
                                   unsigned seq, int parity) {
    const int b = blockIdx.x;
    const int nb = gridDim.x;
    const int t = threadIdx.x;
    const int64_t stride = (int64_t)nb * kArThreads;
    const int64_t i0 = (int64_t)b * kArThreads + t;
    const int64_t my_slot = ((int64_t)parity * world + rank) * kArMaxFloats;
    const int64_t flags_f = data_floats_total(world);  // flag region, in floats

    // Push this block's range into every peer's slot for this (parity, rank).
    const float4* src = reinterpret_cast<const float4*>(data);
    for (int p = 0; p < world; ++p) {
        float4* dst = reinterpret_cast<float4*>(peers.base[p] + my_slot);
        for (int64_t i = i0; i < count4; i += stride) dst[i] = src[i];
    }
    __threadfence_system();
    __syncthreads();
    if (t < world) {
        unsigned* f = reinterpret_cast<unsigned*>(peers.base[t] + flags_f);
        f[((int64_t)parity * world + rank) * kArMaxBlocks + b] = seq;
    }

    // Wait for every writer's flag for THIS block, then reduce locally in
    // fixed rank order (deterministic, identical on all ranks).
    volatile unsigned* lf =
        reinterpret_cast<volatile unsigned*>(peers.base[rank] + flags_f);
    if (t < world) {
        while (lf[((int64_t)parity * world + t) * kArMaxBlocks + b] < seq) {}
    }
    __syncthreads();
    __threadfence_system();

    float4* out = reinterpret_cast<float4*>(data);
    for (int64_t i = i0; i < count4; i += stride) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int w = 0; w < world; ++w) {
            const float4 v = reinterpret_cast<const float4*>(
                peers.base[rank] + ((int64_t)parity * world + w) * kArMaxFloats)[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        out[i] = acc;
    }
}

}  // namespace

struct CustomAr {
    int rank = 0;
    int world = 0;
    unsigned seq = 0;
    float* local = nullptr;            // this rank's staging allocation
    ArPeers peers = {};                // base[w]: mapped pointer to rank w's staging
    bool opened[kArMaxWorld] = {};     // which entries came from cudaIpcOpenMemHandle
};

int64_t custom_ar_max_count() { return kArMaxFloats; }

CustomAr* custom_ar_create(int rank, int world, void* out_handle) {
    static_assert(sizeof(cudaIpcMemHandle_t) == kCustomArHandleBytes,
                  "cudaIpcMemHandle_t size mismatch");
    if (rank < 0 || world < 2 || world > kArMaxWorld || rank >= world) return nullptr;
    auto* st = new CustomAr();
    st->rank = rank;
    st->world = world;
    if (cudaMalloc(&st->local, alloc_bytes(world)) != cudaSuccess) {
        delete st;
        return nullptr;
    }
    // Flags must start below seq 1; data slots need no init (flag-gated).
    if (cudaMemset(st->local, 0, alloc_bytes(world)) != cudaSuccess ||
        cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(out_handle),
                            st->local) != cudaSuccess) {
        custom_ar_destroy(st);
        return nullptr;
    }
    return st;
}

bool custom_ar_open(CustomAr* st, const void* handles) {
    const auto* h = reinterpret_cast<const cudaIpcMemHandle_t*>(handles);
    for (int w = 0; w < st->world; ++w) {
        if (w == st->rank) {
            st->peers.base[w] = st->local;
            continue;
        }
        void* p = nullptr;
        cudaError_t e =
            cudaIpcOpenMemHandle(&p, h[w], cudaIpcMemLazyEnablePeerAccess);
        if (e != cudaSuccess) {
            std::fprintf(stderr,
                         "[glmserve] custom_ar: cudaIpcOpenMemHandle(rank %d -> %d) "
                         "failed: %s\n", st->rank, w, cudaGetErrorString(e));
            cudaGetLastError();  // clear
            return false;
        }
        st->peers.base[w] = static_cast<float*>(p);
        st->opened[w] = true;
    }
    return true;
}

void custom_ar_run(CustomAr* st, float* data, int64_t count, cudaStream_t stream) {
    const int64_t count4 = count / 4;
    int nb = static_cast<int>((count4 + kArThreads - 1) / kArThreads);
    if (nb < 1) nb = 1;
    if (nb > kArMaxBlocks) nb = kArMaxBlocks;
    st->seq += 1;
    const int parity = static_cast<int>(st->seq & 1u);
    one_shot_ar_kernel<<<nb, kArThreads, 0, stream>>>(
        st->peers, data, count4, st->rank, st->world, st->seq, parity);
}

void custom_ar_destroy(CustomAr* st) {
    if (!st) return;
    for (int w = 0; w < st->world; ++w)
        if (st->opened[w] && st->peers.base[w]) cudaIpcCloseMemHandle(st->peers.base[w]);
    if (st->local) cudaFree(st->local);
    delete st;
}

}  // namespace cuda
}  // namespace glmserve
