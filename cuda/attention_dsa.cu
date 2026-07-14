// glmserve — GLM DSA (sparse) attention.
//
// GLM-5.2 uses a "lightning indexer" that scores keys and attends only to the
// top-`index_topk` per query (spec §3, §9.4). The roadmap is explicit: don't
// start with the perfect DSA kernel — first make logits match, then specialize.
// This file contains two paths:
//   * ctx <= index_topk  -> exact dense attention (indexer is a no-op)
//   * learned top-k indices -> sparse attention over the selected keys
//   * recent-window fallback -> a deterministic sparse baseline for layers whose
//     learned / shared IndexShare indices are not wired yet.
//
// ARCHITECTURE:
//   - Indexer: 64 heads, 128-dim index vectors per key
//   - For ctx > 2048: learned top-k selector scores all keys, attends to top-2048
//   - Fallback: recent window of last 2048 keys (deterministic)
//   - Shared indexer: some layers share the same top-k indices
//
// SELECTOR IMPLEMENTATION:
//   - Two-stage per query chunk:
//     1. dsa_score_kernel: one WARP per (key, query), lane h owns head h's dot
//     2. dsa_radix_select_kernel: exact top-k by radix descent
//   - Radix select: 64-bit ordering key = (score bits | ~index)
//     -> ties resolve to smallest index (CPU reference parity)
//   - Smem bitonic sort emits selected indices ascending
//     -> deterministic attention accumulation order
//
// GEMM SCORING PATH (GLMSERVE_DSA_GEMM=1):
//   - Per-head dots = plain GEMM: D[(tq,h), j] = index_q[t,h,:] . index_k[j,:]
//   - cuBLAS SGEMM on fp32-converted key tile
//   - ReLU-weighted head sum in cheap epilogue kernel
//   - Requires CUBLAS_PEDANTIC_MATH (true fp32 products, not TF32)
//     -> TF32 rotates scores enough to flip top-k selections
//   - Only summation order differs (ULP-level), absorbed by radix tiebreak
//
// PERFORMANCE:
//   - Scalar kernel: ~1.1 TFLOPS (instruction-issue-bound)
//   - Top prefill cost at long context: 49% at 8K, 65% at 16K
//   - 19 ms/tok of decode at 8K depth
//   - GEMM path: ~2-3x faster for scoring stage
#include "common.cuh"
#include "kernels.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <cstdlib>

namespace glmserve {
namespace cuda {

__device__ __forceinline__ const float* dsa_kv_at(const float* base, const int* block_table,
                                                   int64_t pos, int64_t block_size,
                                                   int64_t n_kv_heads, int64_t head_dim,
                                                   int64_t kvh) {
    int phys = block_table[pos / block_size];
    int64_t off = pos % block_size;
    return base + (((int64_t)phys * block_size + off) * n_kv_heads + kvh) * head_dim;
}

__global__ void attn_dsa_window_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                       const float* __restrict__ vc, const int* __restrict__ block_table,
                                       int64_t start_pos, int64_t n_heads, int64_t n_kv_heads,
                                       int64_t head_dim, int64_t block_size, int64_t topk,
                                       float scale, float* __restrict__ out) {
    extern __shared__ float red[];
    int64_t qi = blockIdx.x, h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int64_t group = n_heads / n_kv_heads;
    int64_t kvh = h / group;
    int64_t qpos = start_pos + qi;
    int64_t lo = (qpos + 1 > topk) ? (qpos + 1 - topk) : 0;   // recent window start

    float qd = q[(qi * n_heads + h) * head_dim + d];
    float m = -1e30f, l = 0.0f, acc = 0.0f;

    for (int64_t j = lo; j <= qpos; ++j) {
        const float* kj = dsa_kv_at(kc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        red[d] = qd * kj[d];
        __syncthreads();
        for (int stride = (int)head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();
        const float* vj = dsa_kv_at(vc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        float new_m = fmaxf(m, score);
        float corr = __expf(m - new_m);
        float p = __expf(score - new_m);
        l = l * corr + p;
        acc = acc * corr + p * vj[d];
        m = new_m;
    }
    out[(qi * n_heads + h) * head_dim + d] = (l > 0.0f) ? acc / l : 0.0f;
}

// ---- learned top-k indexer (parallel) --------------------------------------
// The selector runs in two stages per query chunk:
//   1. dsa_score_kernel — one WARP per (key, query): lane h owns head h's
//      128-dim dot against the shared fp16 key row, warp-reduces the
//      relu-weighted head sum into scores[tq, j].
//   2. dsa_radix_select_kernel — one block per query: exact top-k by radix
//      descent over a UNIQUE 64-bit ordering key (score bits | ~index), so
//      ties resolve to the smallest index — the same winner the CPU
//      reference's strict replace-min scan keeps — then a smem bitonic sort
//      emits the selected indices ascending (the deterministic attention
//      accumulation order).
// A serial single-thread predecessor walked ctx*heads*dim per query on one
// thread — unusable beyond the 2048-key dense window.

__global__ void dsa_score_kernel(const float* __restrict__ index_q,
                                 const __half* __restrict__ index_k_cache,
                                 const float* __restrict__ index_w,
                                 int64_t t0, int64_t start_pos, int64_t index_heads,
                                 int64_t index_dim, float score_scale, float weight_scale,
                                 int64_t n_keys, float* __restrict__ scores) {
    const int64_t j = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t tq = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (j >= n_keys) return;
    const int64_t t = t0 + tq;
    const int64_t qpos = start_pos + t;
    if (j > qpos) return;
    const __half* kj = index_k_cache + j * index_dim;
    float total = 0.0f;
    for (int64_t h = lane; h < index_heads; h += 32) {
        const float* qh = index_q + (t * index_heads + h) * index_dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < index_dim; ++d)
            dot += qh[d] * __half2float(kj[d]);
        total += index_w[t * index_heads + h] * weight_scale * fmaxf(0.0f, dot * score_scale);
    }
    total = warp_reduce_sum(total);
    if (lane == 0) scores[tq * n_keys + j] = total;
}

// ---- GEMM scoring path ------------------------------------------------------
// dsa_score_kernel is instruction-issue-bound (~1.1 TFLOPS: scalar fp32 FMA
// with a per-element half->float convert), which makes the selector the top
// prefill cost at long context (49% at 8K, 65% at 16K) and 19 ms/tok of
// decode at 8K depth. The per-head dots are a plain GEMM: D[(tq,h), j] =
// index_q[t,h,:] . index_k[j,:], so run cuBLAS SGEMM on an fp32-converted key
// tile and fold the ReLU-weighted head sum in a cheap epilogue.
//
// The GEMM runs in CUBLAS_PEDANTIC_MATH (gemm_fp32_pedantic): the default
// math mode rounds each fp32 product to TF32's 10-bit mantissa, which rotates
// the per-head dots enough to flip top-k selections and diverge the greedy
// stream from token 1 (phase Y gate). PEDANTIC keeps true fp32 products, so
// only the (ULP-level) cuBLAS summation order differs from the scalar kernel
// — a rounding-level residual the radix select's index tiebreak absorbs.
// GLMSERVE_DSA_GEMM=0 restores the scalar kernel.

__global__ void dsa_convert_k_kernel(const __half* __restrict__ src, int64_t n,
                                     float* __restrict__ dst) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        dst[i] = __half2float(src[i]);
}

// scores[tq, j0+j] = sum_h w[t,h]*wscale * relu(D[(tq*H+h), j] * sscale).
// Rows past a query's causal limit stay untouched semantically: the radix
// select only reads [0, qpos+1), so writing them is harmless.
__global__ void dsa_reduce_kernel(const float* __restrict__ D, const float* __restrict__ index_w,
                                  int64_t t0, int64_t index_heads, int64_t tile_n, int64_t j0,
                                  float score_scale, float weight_scale, int64_t row_stride,
                                  float* __restrict__ scores) {
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t tq = blockIdx.y;
    if (j >= tile_n) return;
    const float* w = index_w + (t0 + tq) * index_heads;
    const float* Dq = D + tq * index_heads * tile_n;
    float total = 0.0f;
    for (int64_t h = 0; h < index_heads; ++h)
        total += w[h] * weight_scale * fmaxf(0.0f, Dq[h * tile_n + j] * score_scale);
    scores[tq * row_stride + j0 + j] = total;
}

static bool dsa_gemm_enabled() {
    static const bool on = [] {
        const char* e = getenv("GLMSERVE_DSA_GEMM");
        return !e || atoi(e) != 0;
    }();
    return on;
}

// cub::DeviceRadixSort temp-storage sizing for the decode top-k path.
size_t dsa_cub_temp_bytes(int64_t max_ctx) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, bytes,
                                    (const uint64_t*)nullptr, (uint64_t*)nullptr,
                                    (const int*)nullptr, (int*)nullptr,
                                    (int)max_ctx, 0, 64, 0);
    return bytes;
}

// Unique ordering key: flipped score bits (ascending uint == ascending float)
// over ~index — descending key == (score desc, index asc), no duplicates.
__device__ __forceinline__ uint64_t dsa_key(float s, uint32_t j) {
    uint32_t u = __float_as_uint(s);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t)u << 32) | (uint64_t)(0xFFFFFFFFu - j);
}

__global__ void dsa_radix_select_kernel(const float* __restrict__ scores, int64_t row_stride,
                                        int64_t t0, int64_t start_pos, int64_t index_topk,
                                        int* __restrict__ topk_indices,
                                        float* __restrict__ topk_scores) {
    extern __shared__ int sel[];   // [next_pow2(index_topk)]
    __shared__ int hist[256];
    __shared__ int sh_scal[2];     // [0] remaining k at this level, [1] compact cursor
    const int64_t tq = blockIdx.x;
    const int64_t t = t0 + tq;
    const int64_t qpos = start_pos + t;
    const int64_t count = qpos + 1;
    const float* sc = scores + tq * row_stride;
    int* ids = topk_indices + t * index_topk;
    float* out_s = topk_scores + t * index_topk;
    const int tid = threadIdx.x;

    if (count <= index_topk) {
        // Causal prefix fits: every key is selected, already ascending.
        for (int64_t j = tid; j < index_topk; j += blockDim.x) {
            ids[j] = j < count ? (int)j : 0;
            out_s[j] = j < count ? sc[j] : -1e30f;
        }
        return;
    }

    // Radix descent, one byte per level, to the exact index_topk-th largest key.
    uint64_t prefix = 0;
    int remain = (int)index_topk;
    for (int level = 7; level >= 0; --level) {
        const int shift = 8 * level;
        const uint64_t mask_hi = level == 7 ? 0ull : ~((1ull << (shift + 8)) - 1ull);
        for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        for (int64_t j = tid; j < count; j += blockDim.x) {
            const uint64_t key = dsa_key(sc[j], (uint32_t)j);
            if ((key & mask_hi) == prefix)
                atomicAdd(&hist[(key >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int rem = remain;
            int b = 255;
            for (; b > 0; --b) {
                if (hist[b] >= rem) break;
                rem -= hist[b];
            }
            sh_scal[0] = rem;
            hist[0] = b;   // reuse as broadcast slot (histogram is consumed)
        }
        __syncthreads();
        remain = sh_scal[0];
        prefix |= (uint64_t)(unsigned)hist[0] << shift;
        __syncthreads();
    }
    // prefix is now the exact threshold key; keys are unique, so >= selects
    // exactly index_topk elements.
    if (tid == 0) sh_scal[1] = 0;
    __syncthreads();
    for (int64_t j = tid; j < count; j += blockDim.x) {
        if (dsa_key(sc[j], (uint32_t)j) >= prefix)
            sel[atomicAdd(&sh_scal[1], 1)] = (int)j;
    }
    __syncthreads();
    // Bitonic sort ascending over the pow2-padded smem slab.
    int P = 1;
    while (P < (int)index_topk) P <<= 1;
    for (int j = (int)index_topk + tid; j < P; j += blockDim.x) sel[j] = 0x7FFFFFFF;
    __syncthreads();
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < P; i += blockDim.x) {
                const int ixj = i ^ stride;
                if (ixj > i) {
                    const bool up = (i & size) == 0;
                    const int a = sel[i], b = sel[ixj];
                    if ((a > b) == up) { sel[i] = b; sel[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < (int)index_topk; i += blockDim.x) {
        ids[i] = sel[i];
        out_s[i] = sc[sel[i]];
    }
}

// ---- cub::DeviceRadixSort top-k path (decode, nc == 1) -------------------
// At decode the radix select launches a single block (nc == 1): 8 sequential
// byte-level passes over the whole ctx on one SM, which is the dominant
// decode cost (24% at 8K, 27% at 16K). cub::DeviceRadixSort is a multi-block
// GPU-wide radix sort — for count up to 16K it finishes in a few us vs the
// single-block select's ~100 us. The 64-bit ordering key (score asc, ~index)
// makes the ascending sort's last index_topk entries exactly the top-k by
// (score desc, index asc); a final single-block bitonic pass re-sorts those
// index_topk by index ascending (deterministic attention accumulation order,
// matching the radix path). GLMSERVE_DSA_CUB=0 restores the single-block path.
__global__ void dsa_build_keys_kernel(const float* __restrict__ scores, int64_t count,
                                      int64_t tq, int64_t row_stride,
                                      uint64_t* __restrict__ keys, int* __restrict__ indices) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    keys[j] = dsa_key(scores[tq * row_stride + j], (uint32_t)j);
    indices[j] = (int)j;
}

// Extract the top index_topk from the cub-sorted indices (ascending key =
// ascending score; the tail is score desc, index asc) and bitonic-sort them
// by index ascending to match the radix path's output order.
__global__ void dsa_extract_topk_cub_kernel(const int* __restrict__ sorted_idx,
                                            int64_t count, int64_t tq, int64_t index_topk,
                                            int64_t row_stride,
                                            const float* __restrict__ scores,
                                            int* __restrict__ topk_indices,
                                            float* __restrict__ topk_scores) {
    extern __shared__ int sel[];   // [next_pow2(index_topk)]
    const int tid = threadIdx.x;
    const float* sc = scores + tq * row_stride;
    int* ids = topk_indices + tq * index_topk;
    float* out_s = topk_scores + tq * index_topk;
    // The last index_topk of the ascending sort are the top-k.
    for (int i = tid; i < (int)index_topk; i += blockDim.x)
        sel[i] = sorted_idx[count - index_topk + i];
    int P = 1;
    while (P < (int)index_topk) P <<= 1;
    for (int i = (int)index_topk + tid; i < P; i += blockDim.x) sel[i] = 0x7FFFFFFF;
    __syncthreads();
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < P; i += blockDim.x) {
                const int ixj = i ^ stride;
                if (ixj > i) {
                    const bool up = (i & size) == 0;
                    const int a = sel[i], b = sel[ixj];
                    if ((a > b) == up) { sel[i] = b; sel[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < (int)index_topk; i += blockDim.x) {
        ids[i] = sel[i];
        out_s[i] = sc[sel[i]];
    }
}

static bool dsa_cub_enabled() {
    static const bool on = [] {
        const char* e = getenv("GLMSERVE_DSA_CUB");
        return !e || atoi(e) != 0;
    }();
    return on;
}

void dsa_select_topk(const float* index_q, const __half* index_k_cache,
                     const float* index_w, int64_t n_query, int64_t start_pos,
                     int64_t index_heads, int64_t index_dim, int64_t index_topk,
                     float score_scale, float weight_scale, float* score_scratch,
                     int* topk_indices, float* topk_scores,
                     float* gemm_dbuf, float* gemm_kf32,
                     DsaCubScratch* cub_scratch, cudaStream_t s) {
    int64_t P = 1;
    while (P < index_topk) P <<= 1;
    const size_t shmem = (size_t)P * sizeof(int);
    const bool use_gemm = gemm_dbuf && gemm_kf32 && dsa_gemm_enabled();
    for (int64_t t0 = 0; t0 < n_query; t0 += kDsaScoreChunk) {
        const int64_t nc = std::min<int64_t>(kDsaScoreChunk, n_query - t0);
        const int64_t n_keys = start_pos + t0 + nc;   // longest prefix in the chunk
        if (use_gemm) {
            for (int64_t j0 = 0; j0 < n_keys; j0 += kDsaKeyTile) {
                const int64_t tn = std::min<int64_t>(kDsaKeyTile, n_keys - j0);
                const int64_t cvt = tn * index_dim;
                dsa_convert_k_kernel<<<(unsigned)((cvt + 255) / 256), 256, 0, s>>>(
                    index_k_cache + j0 * index_dim, cvt, gemm_kf32);
                gemm_fp32_pedantic(index_q + t0 * index_heads * index_dim, gemm_kf32, nullptr,
                                    gemm_dbuf, nc * index_heads, index_dim, tn, s);
                dim3 rg((unsigned)((tn + 255) / 256), (unsigned)nc);
                dsa_reduce_kernel<<<rg, 256, 0, s>>>(gemm_dbuf, index_w, t0, index_heads,
                                                     tn, j0, score_scale, weight_scale,
                                                     n_keys, score_scratch);
            }
        } else {
            dim3 sg((unsigned)((n_keys + 3) / 4), (unsigned)nc);
            dsa_score_kernel<<<sg, 128, 0, s>>>(index_q, index_k_cache, index_w, t0, start_pos,
                                                index_heads, index_dim, score_scale, weight_scale,
                                                n_keys, score_scratch);
        }
        // At decode (nc == 1) the single-block radix select is the dominant
        // selector cost (8 passes over the whole ctx on one SM). cub's
        // multi-block radix sort is GPU-wide and finishes in a fraction of the
        // time for count up to 16K; the final bitonic re-sort of index_topk
        // is one block (same as the radix path's tail). At prefill (nc > 1)
        // the radix select already launches nc blocks, so keep it.
        const int64_t tq = t0;  // nc == 1 in the cub path, so one query
        const int64_t count = start_pos + tq + 1;
        DsaCubScratch* cub = cub_scratch;
        const bool use_cub = cub && cub->keys_in && cub->temp && dsa_cub_enabled() &&
                             nc == 1 && count > index_topk;
        if (use_cub) {
            // Build 64-bit ordering keys + index values for [0, count).
            const int n = (int)count;
            dsa_build_keys_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(
                score_scratch, n, tq, n_keys, cub->keys_in, cub->idx_in);
            // Ascending sort: last index_topk = top-k by (score desc, index asc).
            cub::DeviceRadixSort::SortPairs(
                cub->temp, cub->temp_bytes,
                cub->keys_in, cub->keys_out,
                cub->idx_in, cub->idx_out,
                n, 0, 64, s);
            dsa_extract_topk_cub_kernel<<<1, 1024, shmem, s>>>(
                cub->idx_out, n, tq, index_topk, n_keys,
                score_scratch, topk_indices, topk_scores);
        } else {
            dsa_radix_select_kernel<<<(unsigned)nc, 1024, shmem, s>>>(
                score_scratch, n_keys, t0, start_pos, index_topk, topk_indices, topk_scores);
        }
    }
}

__global__ void attn_dsa_indexed_kernel(const float* __restrict__ q,
                                        const float* __restrict__ kc,
                                        const float* __restrict__ vc,
                                        const int* __restrict__ block_table,
                                        const int* __restrict__ topk_indices,
                                        int64_t start_pos, int64_t n_heads,
                                        int64_t n_kv_heads, int64_t head_dim,
                                        int64_t block_size, int64_t index_topk,
                                        float scale, float* __restrict__ out) {
    extern __shared__ float red[];
    int64_t qi = blockIdx.x, h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int64_t group = n_heads / n_kv_heads;
    int64_t kvh = h / group;
    int64_t qpos = start_pos + qi;
    int64_t count = (qpos + 1 < index_topk) ? (qpos + 1) : index_topk;
    const int* ids = topk_indices + qi * index_topk;

    float qd = q[(qi * n_heads + h) * head_dim + d];
    float m = -1e30f, l = 0.0f, acc = 0.0f;
    for (int64_t kk = 0; kk < count; ++kk) {
        int64_t j = ids[kk];
        const float* kj = dsa_kv_at(kc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        red[d] = qd * kj[d];
        __syncthreads();
        for (int stride = (int)head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();
        const float* vj = dsa_kv_at(vc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        float new_m = fmaxf(m, score);
        float corr = __expf(m - new_m);
        float p = __expf(score - new_m);
        l = l * corr + p;
        acc = acc * corr + p * vj[d];
        m = new_m;
    }
    out[(qi * n_heads + h) * head_dim + d] = (l > 0.0f) ? acc / l : 0.0f;
}

void attention_dsa_indexed_paged(const float* q, const float* k_cache, const float* v_cache,
                                 const int* block_table, const int* topk_indices,
                                 int64_t n_query, int64_t start_pos, int64_t n_heads,
                                 int64_t n_kv_heads, int64_t head_dim, int64_t block_size,
                                 int64_t index_topk, float scale, float* out,
                                 cudaStream_t s) {
    dim3 grid((unsigned)n_query, (unsigned)n_heads);
    int threads = (int)head_dim;
    size_t shmem = (size_t)head_dim * sizeof(float);
    attn_dsa_indexed_kernel<<<grid, threads, shmem, s>>>(
        q, k_cache, v_cache, block_table, topk_indices, start_pos, n_heads, n_kv_heads,
        head_dim, block_size, index_topk, scale, out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "attention_dsa_indexed_paged kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

// Decode specialization for the recent-window baseline: split the sparse key
// window across many blocks, then merge online-softmax partials per head. This
// mirrors dense flash-decoding, but the key range is [max(0, qpos+1-topk), qpos].
__global__ void attn_dsa_decode_p1_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                          const float* __restrict__ vc,
                                          const int* __restrict__ block_table,
                                          int64_t qpos, int64_t n_heads, int64_t n_kv_heads,
                                          int64_t head_dim, int64_t block_size, int64_t topk,
                                          float scale, int64_t keys_per_split,
                                          float* __restrict__ part_acc,
                                          float* __restrict__ part_m,
                                          float* __restrict__ part_l) {
    extern __shared__ float red[];
    int64_t h = blockIdx.x;
    int64_t sp = blockIdx.y;
    int64_t S = gridDim.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int64_t group = n_heads / n_kv_heads;
    int64_t kvh = h / group;
    int64_t lo = (qpos + 1 > topk) ? (qpos + 1 - topk) : 0;
    int64_t j0 = lo + sp * keys_per_split;
    int64_t j1 = j0 + keys_per_split;
    if (j1 > qpos + 1) j1 = qpos + 1;

    float qd = q[h * head_dim + d];
    float m = -1e30f, l = 0.0f, acc = 0.0f;
    for (int64_t j = j0; j < j1; ++j) {
        const float* kj = dsa_kv_at(kc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        red[d] = qd * kj[d];
        __syncthreads();
        for (int stride = (int)head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();
        const float* vj = dsa_kv_at(vc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        float new_m = fmaxf(m, score);
        float corr = __expf(m - new_m);
        float p = __expf(score - new_m);
        l = l * corr + p;
        acc = acc * corr + p * vj[d];
        m = new_m;
    }
    int64_t idx = h * S + sp;
    part_acc[idx * head_dim + d] = acc;
    if (d == 0) {
        part_m[idx] = m;
        part_l[idx] = l;
    }
}

__global__ void attn_dsa_decode_p2_kernel(const float* __restrict__ part_acc,
                                          const float* __restrict__ part_m,
                                          const float* __restrict__ part_l,
                                          int64_t n_heads, int64_t head_dim, int64_t S,
                                          float* __restrict__ out) {
    int64_t h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    float gm = -1e30f;
    for (int64_t sp = 0; sp < S; ++sp) {
        int64_t idx = h * S + sp;
        if (part_l[idx] > 0.0f) gm = fmaxf(gm, part_m[idx]);
    }
    float l = 0.0f, acc = 0.0f;
    for (int64_t sp = 0; sp < S; ++sp) {
        int64_t idx = h * S + sp;
        float lp = part_l[idx];
        if (lp <= 0.0f) continue;
        float w = __expf(part_m[idx] - gm);
        l += lp * w;
        acc += part_acc[idx * head_dim + d] * w;
    }
    out[h * head_dim + d] = (l > 0.0f) ? acc / l : 0.0f;
}

void attention_dsa_paged(const float* q, const float* k_cache, const float* v_cache,
                         const int* block_table, int64_t n_query, int64_t start_pos,
                         int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                         int64_t block_size, int64_t index_topk, float scale,
                         float* out, cudaStream_t s, float* part_acc, float* part_m,
                         float* part_l, int64_t max_splits) {
    int64_t ctx = start_pos + n_query;
    if (ctx <= index_topk) {
        // exact dense path
        attention_dense_paged(q, k_cache, v_cache, block_table, n_query, start_pos,
                              n_heads, n_kv_heads, head_dim, block_size, scale, out, s);
        return;
    }
    if (n_query == 1 && part_acc && part_m && part_l && max_splits > 0) {
        int64_t window = index_topk < ctx ? index_topk : ctx;
        int64_t S = (window + 127) / 128;
        if (S < 1) S = 1;
        if (S > max_splits) S = max_splits;
        int threads = (int)head_dim;
        size_t shmem = (size_t)head_dim * sizeof(float);
        if (S == 1) {
            dim3 grid((unsigned)n_query, (unsigned)n_heads);
            attn_dsa_window_kernel<<<grid, threads, shmem, s>>>(q, k_cache, v_cache, block_table,
                                                                start_pos, n_heads, n_kv_heads,
                                                                head_dim, block_size, index_topk,
                                                                scale, out);
            return;
        }
        int64_t keys_per_split = (window + S - 1) / S;
        dim3 g1((unsigned)n_heads, (unsigned)S);
        attn_dsa_decode_p1_kernel<<<g1, threads, shmem, s>>>(q, k_cache, v_cache, block_table,
                                                             start_pos, n_heads, n_kv_heads,
                                                             head_dim, block_size, index_topk,
                                                             scale, keys_per_split, part_acc,
                                                             part_m, part_l);
        attn_dsa_decode_p2_kernel<<<(unsigned)n_heads, threads, 0, s>>>(part_acc, part_m,
                                                                        part_l, n_heads,
                                                                        head_dim, S, out);
        return;
    }
    dim3 grid((unsigned)n_query, (unsigned)n_heads);
    int threads = (int)head_dim;
    size_t shmem = (size_t)head_dim * sizeof(float);
    attn_dsa_window_kernel<<<grid, threads, shmem, s>>>(q, k_cache, v_cache, block_table,
                                                        start_pos, n_heads, n_kv_heads,
                                                        head_dim, block_size, index_topk,
                                                        scale, out);
}

}  // namespace cuda
}  // namespace glmserve
