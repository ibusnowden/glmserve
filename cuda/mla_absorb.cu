// glmserve — absorbed-MLA decode over an fp16 latent KV cache.
//
// MLA's structural win: every head's key/value is a projection of ONE shared
// per-token latent, so the cache only needs [ctx, kvlat + rope] (normed c_kv |
// roped k_pe) instead of [ctx, heads, head_dim] K and V — 576 vs 4096 floats
// per token per rank at GLM-5.2 dims (and fp16 halves it again). Decode
// attention "absorbs" the up-projections into the query and output:
//
//   score(h,t) = q_nope[h].(W_UK[h] c[t]) + q_pe[h].k_pe[t]
//              = [W_UK[h]^T q_nope[h] | q_pe[h]] . [c[t] | k_pe[t]]  = qhat.chat
//   out(h)     = W_UV[h] (sum_t p_t c[t]) = W_UV[h] ohat[h]
//
// i.e. decode becomes MQA with head_dim = kvlat+rope and value_dim = kvlat,
// reading each latent row ONCE for all local heads. W_UK/W_UV are the kv_b
// rows, kept device-resident in fp16 (dequant_rows_f16 in qgemm.cu).
//
// Kernels are dim-generic (validated on the tiny checkpoint's small dims).
#include "common.cuh"
#include "kernels.cuh"

#include <cuda_fp16.h>

namespace glmserve {
namespace cuda {

// Per-lane register fragment cap: supports dims up to 32*kMlaMaxFrag.
static constexpr int kMlaMaxFrag = 24;   // kvlat+rope <= 768

// Store [n] tokens' (normed ckv | roped kpe) as one fp16 latent row each.
__global__ void latent_store_kernel(const float* __restrict__ ckv, const float* __restrict__ kpe,
                                    int64_t n, int64_t kvlat, int64_t rope,
                                    __half* __restrict__ dst) {
    const int64_t dim = kvlat + rope;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * dim;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = i / dim, d = i % dim;
        const float v = d < kvlat ? ckv[t * kvlat + d] : kpe[t * rope + (d - kvlat)];
        dst[i] = __float2half(v);
    }
}

void latent_store(const float* ckv, const float* kpe, __half* latent, int64_t start_pos,
                  int64_t n, int64_t kvlat, int64_t rope, cudaStream_t s) {
    const int64_t total = n * (kvlat + rope);
    const unsigned blocks = (unsigned)((total + 255) / 256);
    latent_store_kernel<<<blocks, 256, 0, s>>>(ckv, kpe, n, kvlat, rope,
                                               latent + start_pos * (kvlat + rope));
}

// qhat[t, h] = [ W_UK[h]^T q_nope[t, h] | q_pe[t, h] ].  kvb is the fp16 kv_b
// shard [heads, nope+vd, kvlat]; W_UK[h] = its first `nope` rows. One block
// per (head, query); thread j owns output latent dim j (coalesced row reads).
__global__ void mla_absorb_q_kernel(const float* __restrict__ q, const __half* __restrict__ kvb,
                                    int64_t qk, int64_t nope, int64_t vd,
                                    int64_t kvlat, int64_t rope,
                                    float* __restrict__ qhat) {
    const int64_t h = blockIdx.x, t = blockIdx.y, H = gridDim.x;
    const float* qh = q + (t * H + h) * qk;
    const __half* Wk = kvb + h * (nope + vd) * kvlat;
    float* out = qhat + (t * H + h) * (kvlat + rope);
    for (int64_t j = threadIdx.x; j < kvlat; j += blockDim.x) {
        float acc = 0.0f;
        for (int64_t i = 0; i < nope; ++i)
            acc += qh[i] * __half2float(Wk[i * kvlat + j]);
        out[j] = acc;
    }
    for (int64_t r = threadIdx.x; r < rope; r += blockDim.x)
        out[kvlat + r] = qh[nope + r];
}

void mla_absorb_q(const float* q, const __half* kvb_f16, int64_t nq, int64_t n_heads,
                  int64_t qk, int64_t nope, int64_t vd, int64_t kvlat, int64_t rope,
                  float* qhat, cudaStream_t s) {
    dim3 grid((unsigned)n_heads, (unsigned)nq);
    mla_absorb_q_kernel<<<grid, 256, 0, s>>>(q, kvb_f16, qk, nope, vd, kvlat, rope, qhat);
}

// Per-query key range for the three decode modes. `indices` selects DSA top-k
// rows (count clamped to the causal prefix); `win > 0` is the recent-window
// fallback; otherwise dense [0, qpos].
struct KeyRange {
    int64_t lo, count;
    const int* ids;
};
__device__ __forceinline__ KeyRange mla_key_range(const int* indices, int64_t index_topk,
                                                  int64_t win, int64_t qpos, int64_t qi) {
    KeyRange r;
    if (indices) {
        r.ids = indices + qi * index_topk;
        r.lo = 0;
        r.count = min(qpos + 1, index_topk);
    } else if (win > 0) {
        r.ids = nullptr;
        r.lo = (qpos + 1 > win) ? (qpos + 1 - win) : 0;
        r.count = qpos + 1 - r.lo;
    } else {
        r.ids = nullptr;
        r.lo = 0;
        r.count = qpos + 1;
    }
    return r;
}

// Pass 1 (flash-decoding over latents): one WARP per (head, split, query).
// Each key row is read once and serves both the score dot (all dims) and the
// value accumulation (first kvlat dims). Lane l owns dim PAIRS (2l, 2l+1) +
// strides of 64: __half2 loads give the warp full 128-byte coalesced rows (a
// lane-strided scalar layout only moved 64 B per access). dim and kvlat must
// be even (256-multiples in practice).
//
// Split geometry is PER QUERY, mirroring the nq == 1 launch exactly (same
// S_i = ceil(count/16) clamp s_cap, same key ranges): a verify chunk's rows
// then merge bit-identically to the plain decode steps they re-check, so the
// speculative token stream equals plain greedy decode. Splits past a query's
// S_i write empty partials (l = 0), which the merge skips without touching
// the accumulation order.
__global__ void mla_decode_p1_kernel(const float* __restrict__ qhat, const __half* __restrict__ latent,
                                     const int* __restrict__ indices, int64_t index_topk,
                                     int64_t win, int64_t qpos0, int64_t kvlat, int64_t dim,
                                     float scale, int64_t s_cap,
                                     float* __restrict__ part_acc, float* __restrict__ part_m,
                                     float* __restrict__ part_l) {
    const int64_t h = blockIdx.x, sp = blockIdx.y, S = gridDim.y, qi = blockIdx.z;
    const int64_t H = gridDim.x;
    const int lane = threadIdx.x & 31;
    const int nfrag = (int)((dim + 63) >> 6);   // pairs per lane
    const int64_t qpos = qpos0 + qi;
    const KeyRange kr = mla_key_range(indices, index_topk, win, qpos, qi);

    int64_t s_i = (kr.count + 15) >> 4;
    if (s_i < 1) s_i = 1;
    if (s_i > s_cap) s_i = s_cap;
    const int64_t idx0 = (qi * H + h) * S + sp;
    if (sp >= s_i) {
        if (lane == 0) { part_m[idx0] = -1e30f; part_l[idx0] = 0.0f; }
        return;
    }
    const int64_t keys_per_split = (kr.count + s_i - 1) / s_i;

    float2 qf[kMlaMaxFrag];
    const float* qh = qhat + (qi * H + h) * dim;
    for (int i = 0; i < nfrag; ++i) {
        const int64_t d = (lane << 1) + ((int64_t)i << 6);
        qf[i] = d + 1 < dim ? *(const float2*)(qh + d) : make_float2(0.0f, 0.0f);
    }
    float2 acc[kMlaMaxFrag];
    for (int i = 0; i < nfrag; ++i) acc[i] = make_float2(0.0f, 0.0f);
    float m = -1e30f, l = 0.0f;

    const int64_t k0 = sp * keys_per_split;
    const int64_t k1 = min(k0 + keys_per_split, kr.count);
    for (int64_t kk = k0; kk < k1; ++kk) {
        const int64_t pos = kr.ids ? (int64_t)kr.ids[kk] : kr.lo + kk;
        if (pos < 0 || pos > qpos) continue;
        const __half* cj = latent + pos * dim;
        float2 cf[kMlaMaxFrag];
        float sdot = 0.0f;
        for (int i = 0; i < nfrag; ++i) {
            const int64_t d = (lane << 1) + ((int64_t)i << 6);
            if (d + 1 < dim) {
                const float2 v = __half22float2(*(const __half2*)(cj + d));
                cf[i] = v;
                sdot += qf[i].x * v.x + qf[i].y * v.y;
            } else {
                cf[i] = make_float2(0.0f, 0.0f);
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            sdot += __shfl_xor_sync(0xffffffffu, sdot, off);
        const float score = sdot * scale;
        const float new_m = fmaxf(m, score);
        const float corr = __expf(m - new_m);
        const float p = __expf(score - new_m);
        l = l * corr + p;
        for (int i = 0; i < nfrag; ++i) {
            const int64_t d = (lane << 1) + ((int64_t)i << 6);
            if (d + 1 < kvlat) {
                acc[i].x = acc[i].x * corr + p * cf[i].x;
                acc[i].y = acc[i].y * corr + p * cf[i].y;
            }
        }
        m = new_m;
    }
    const int64_t idx = (qi * H + h) * S + sp;
    for (int i = 0; i < nfrag; ++i) {
        const int64_t d = (lane << 1) + ((int64_t)i << 6);
        if (d + 1 < kvlat) *(float2*)(part_acc + idx * kvlat + d) = acc[i];
    }
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
}

// Pass 2 fused with the W_UV expansion: one block per (head, query) merges the
// S partial softmaxes into ohat[kvlat] in SHARED memory, then its warps expand
// out[t, h, vd] = W_UV[h] ohat straight from smem — no global ohat roundtrip
// and one launch instead of two.
__global__ void mla_decode_p2_expand_kernel(const float* __restrict__ part_acc,
                                            const float* __restrict__ part_m,
                                            const float* __restrict__ part_l,
                                            const __half* __restrict__ kvb,
                                            int64_t S, int64_t kvlat, int64_t nope,
                                            int64_t vd, float* __restrict__ out) {
    extern __shared__ float oh[];   // [kvlat]
    const int64_t h = blockIdx.x, qi = blockIdx.y, H = gridDim.x;
    const int64_t base = (qi * H + h) * S;
    float gm = -1e30f;
    for (int64_t sp = 0; sp < S; ++sp)
        if (part_l[base + sp] > 0.0f) gm = fmaxf(gm, part_m[base + sp]);
    for (int64_t d = threadIdx.x; d < kvlat; d += blockDim.x) {
        float l = 0.0f, acc = 0.0f;
        for (int64_t sp = 0; sp < S; ++sp) {
            const float lp = part_l[base + sp];
            if (lp <= 0.0f) continue;
            const float w = __expf(part_m[base + sp] - gm);
            l += lp * w;
            acc += part_acc[(base + sp) * kvlat + d] * w;
        }
        oh[d] = (l > 0.0f) ? acc / l : 0.0f;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    for (int64_t d = warp; d < vd; d += nwarps) {
        const __half* row = kvb + ((h * (nope + vd) + nope) + d) * kvlat;
        float acc = 0.0f;
        for (int64_t j = lane; j < kvlat; j += 32)
            acc += __half2float(row[j]) * oh[j];
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(qi * H + h) * vd + d] = acc;
    }
}

void mla_attention_decode(const float* qhat, const __half* latent, const int* indices,
                          int64_t index_topk, int64_t win, int64_t qpos0, int64_t nq,
                          int64_t n_heads, int64_t kvlat, int64_t rope, float scale,
                          const __half* kvb_f16, int64_t nope, int64_t vd, float* out,
                          float* part_acc, float* part_m, float* part_l,
                          int64_t max_splits, cudaStream_t s) {
    const int64_t dim = kvlat + rope;
    // Longest key range across the nq queries bounds the launch's split dim.
    int64_t max_keys;
    if (indices) max_keys = min(qpos0 + nq, index_topk);
    else if (win > 0) max_keys = min(qpos0 + nq, win);
    else max_keys = qpos0 + nq;
    // One warp per (head, split, query): short 16-key splits keep enough warps
    // in flight at nq == 1 (8 local heads alone would starve a 142-SM part).
    // Verify-sized chunks (nq <= kMlaParityMaxQ) keep the FULL per-query split
    // budget so their geometry matches plain decode bit-for-bit (spec-decode
    // parity); wider chunks (MTP absorb) divide the budget so the partial
    // buffers never outgrow [kMlaParityMaxQ, max_splits, heads, kvlat].
    const int64_t s_cap = nq <= kMlaParityMaxQ
                              ? max_splits
                              : (max_splits / nq > 0 ? max_splits / nq : 1);
    int64_t S = (max_keys + 15) / 16;
    if (S < 1) S = 1;
    if (S > s_cap) S = s_cap;
    dim3 g1((unsigned)n_heads, (unsigned)S, (unsigned)nq);
    mla_decode_p1_kernel<<<g1, 32, 0, s>>>(qhat, latent, indices, index_topk, win, qpos0,
                                           kvlat, dim, scale, s_cap,
                                           part_acc, part_m, part_l);
    dim3 g2((unsigned)n_heads, (unsigned)nq);
    mla_decode_p2_expand_kernel<<<g2, 256, kvlat * sizeof(float), s>>>(
        part_acc, part_m, part_l, kvb_f16, S, kvlat, nope, vd, out);
}

// fp32 -> fp16 device conversion (tiny-checkpoint kv_b is fp32-resident).
__global__ void convert_f32_f16_kernel(const float* __restrict__ src, int64_t n,
                                       __half* __restrict__ dst) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        dst[i] = __float2half(src[i]);
}

void convert_f32_f16(const float* src, int64_t n, __half* dst, cudaStream_t s) {
    convert_f32_f16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, n, dst);
}

}  // namespace cuda
}  // namespace glmserve
