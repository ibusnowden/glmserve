// glmserve — custom small-message TP all-reduce over CUDA IPC peer mappings.
//
// NCCL's per-call latency (~70 us launch + protocol) dominates decode-time
// all-reduces, which are tiny (n<=8 tokens x hidden fp32 = 24-192 KB). This
// one-shot push kernel writes each rank's contribution directly into every
// peer's staging slot over P2P, publishes a per-block sequence flag, then each
// rank reduces its (now local) copies in fixed rank order — one kernel launch,
// deterministic summation, ~15 us. Prefill/vocab-sized reductions stay on NCCL.
//
// Requirements: single-node TP group (tp_size == world_size <= 8), device
// pointers, count % 4 == 0, count <= custom_ar_max_count(). The caller owns
// the fallback path for everything else.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace glmserve {
namespace cuda {

struct CustomAr;  // opaque

// 64-byte IPC handle exchanged between ranks (mirrors cudaIpcMemHandle_t).
constexpr int kCustomArHandleBytes = 64;

// Allocates the local staging buffer and fills out_handle (kCustomArHandleBytes).
// Returns nullptr on failure (caller falls back to NCCL permanently).
CustomAr* custom_ar_create(int rank, int world, void* out_handle);

// Opens the peer handles (world x kCustomArHandleBytes, row-major; entry
// [rank] is ignored — the local buffer is used directly). Returns false on
// failure; the caller must then destroy the state and disable the path on
// EVERY rank (all-or-none), or the group deadlocks.
bool custom_ar_open(CustomAr* st, const void* handles);

// Largest float count the one-shot path accepts.
int64_t custom_ar_max_count();

// In-place all-reduce (sum) of `count` floats on `stream`. All ranks of the
// group must call this collectively with the same count.
void custom_ar_run(CustomAr* st, float* data, int64_t count, cudaStream_t stream);

void custom_ar_destroy(CustomAr* st);

}  // namespace cuda
}  // namespace glmserve
