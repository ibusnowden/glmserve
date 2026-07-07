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

// qhat[h] = [ W_UK[h]^T q_nope[h] | q_pe[h] ].  kvb is the fp16 kv_b shard
// [heads, nope+vd, kvlat]; W_UK[h] = its first `nope` rows. One block per
// head; thread j owns output latent dim j (coalesced row reads).
__global__ void mla_absorb_q_kernel(const float* __restrict__ q, const __half* __restrict__ kvb,
                                    int64_t qk, int64_t nope, int64_t vd,
                                    int64_t kvlat, int64_t rope,
                                    float* __restrict__ qhat) {
    const int64_t h = blockIdx.x;
    const float* qh = q + h * qk;
    const __half* Wk = kvb + h * (nope + vd) * kvlat;
    float* out = qhat + h * (kvlat + rope);
    for (int64_t j = threadIdx.x; j < kvlat; j += blockDim.x) {
        float acc = 0.0f;
        for (int64_t i = 0; i < nope; ++i)
            acc += qh[i] * __half2float(Wk[i * kvlat + j]);
        out[j] = acc;
    }
    for (int64_t r = threadIdx.x; r < rope; r += blockDim.x)
        out[kvlat + r] = qh[nope + r];
}

void mla_absorb_q(const float* q, const __half* kvb_f16, int64_t n_heads, int64_t qk,
                  int64_t nope, int64_t vd, int64_t kvlat, int64_t rope, float* qhat,
                  cudaStream_t s) {
    mla_absorb_q_kernel<<<(unsigned)n_heads, 256, 0, s>>>(q, kvb_f16, qk, nope, vd,
                                                          kvlat, rope, qhat);
}

// out[h] = W_UV[h] ohat[h].  W_UV[h] = kvb rows [nope, nope+vd) of head h.
// One warp per output row (lanes stride the kvlat dot, coalesced fp16 reads);
// 8 warps per block over rows, heads on grid.y.
__global__ void mla_expand_o_kernel(const float* __restrict__ ohat, const __half* __restrict__ kvb,
                                    int64_t nope, int64_t vd, int64_t kvlat,
                                    float* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t d = blockIdx.x * 8 + warp;
    const int64_t h = blockIdx.y;
    if (d >= vd) return;
    const float* oh = ohat + h * kvlat;
    const __half* row = kvb + ((h * (nope + vd) + nope) + d) * kvlat;
    float acc = 0.0f;
    for (int64_t j = lane; j < kvlat; j += 32)
        acc += __half2float(row[j]) * oh[j];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) out[h * vd + d] = acc;
}

void mla_expand_o(const float* ohat, const __half* kvb_f16, int64_t n_heads, int64_t nope,
                  int64_t vd, int64_t kvlat, float* out, cudaStream_t s) {
    dim3 grid((unsigned)((vd + 7) / 8), (unsigned)n_heads);
    mla_expand_o_kernel<<<grid, 256, 0, s>>>(ohat, kvb_f16, nope, vd, kvlat, out);
}

// Pass 1 (flash-decoding over latents): one WARP per (head, split). Each key
// row is read once and serves both the score dot (all dims) and the value
// accumulation (first kvlat dims). Key selection covers the three decode
// modes: full dense (indices == nullptr, j0 == 0), recent window (j0 > 0),
// and DSA top-k (indices != nullptr; negatives / out-of-range skipped).
// Lane l owns dim PAIRS (2l, 2l+1) + strides of 64: __half2 loads give the
// warp full 128-byte coalesced rows (a lane-strided scalar layout only moved
// 64 B per access). dim and kvlat must be even (256-multiples in practice).
__global__ void mla_decode_p1_kernel(const float* __restrict__ qhat, const __half* __restrict__ latent,
                                     const int* __restrict__ indices, int64_t j0, int64_t n_keys,
                                     int64_t qpos, int64_t kvlat, int64_t dim, float scale,
                                     int64_t keys_per_split,
                                     float* __restrict__ part_acc, float* __restrict__ part_m,
                                     float* __restrict__ part_l) {
    const int64_t h = blockIdx.x, sp = blockIdx.y, S = gridDim.y;
    const int lane = threadIdx.x & 31;
    const int nfrag = (int)((dim + 63) >> 6);   // pairs per lane

    float2 qf[kMlaMaxFrag];
    const float* qh = qhat + h * dim;
    for (int i = 0; i < nfrag; ++i) {
        const int64_t d = (lane << 1) + ((int64_t)i << 6);
        qf[i] = d + 1 < dim ? *(const float2*)(qh + d) : make_float2(0.0f, 0.0f);
    }
    float2 acc[kMlaMaxFrag];
    for (int i = 0; i < nfrag; ++i) acc[i] = make_float2(0.0f, 0.0f);
    float m = -1e30f, l = 0.0f;

    const int64_t k0 = sp * keys_per_split;
    const int64_t k1 = min(k0 + keys_per_split, n_keys);
    for (int64_t kk = k0; kk < k1; ++kk) {
        const int64_t pos = indices ? (int64_t)indices[kk] : j0 + kk;
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
    const int64_t idx = h * S + sp;
    for (int i = 0; i < nfrag; ++i) {
        const int64_t d = (lane << 1) + ((int64_t)i << 6);
        if (d + 1 < kvlat) *(float2*)(part_acc + idx * kvlat + d) = acc[i];
    }
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
}

// Pass 2: merge the S partial softmaxes into ohat[h][kvlat].
__global__ void mla_decode_p2_kernel(const float* __restrict__ part_acc,
                                     const float* __restrict__ part_m, const float* __restrict__ part_l,
                                     int64_t S, int64_t kvlat, float* __restrict__ ohat) {
    const int64_t h = blockIdx.x;
    float gm = -1e30f;
    for (int64_t sp = 0; sp < S; ++sp) {
        const int64_t idx = h * S + sp;
        if (part_l[idx] > 0.0f) gm = fmaxf(gm, part_m[idx]);
    }
    for (int64_t d = threadIdx.x; d < kvlat; d += blockDim.x) {
        float l = 0.0f, acc = 0.0f;
        for (int64_t sp = 0; sp < S; ++sp) {
            const int64_t idx = h * S + sp;
            const float lp = part_l[idx];
            if (lp <= 0.0f) continue;
            const float w = __expf(part_m[idx] - gm);
            l += lp * w;
            acc += part_acc[idx * kvlat + d] * w;
        }
        ohat[h * kvlat + d] = (l > 0.0f) ? acc / l : 0.0f;
    }
}

void mla_attention_decode(const float* qhat, const __half* latent, const int* indices,
                          int64_t j0, int64_t n_keys, int64_t qpos, int64_t n_heads,
                          int64_t kvlat, int64_t rope, float scale, float* ohat,
                          float* part_acc, float* part_m, float* part_l,
                          int64_t max_splits, cudaStream_t s) {
    const int64_t dim = kvlat + rope;
    // One warp per (head, split): short 16-key splits keep enough warps in
    // flight (8 local heads alone would starve a 142-SM part).
    int64_t S = (n_keys + 15) / 16;
    if (S < 1) S = 1;
    if (S > max_splits) S = max_splits;
    const int64_t keys_per_split = (n_keys + S - 1) / S;
    dim3 g1((unsigned)n_heads, (unsigned)S);
    mla_decode_p1_kernel<<<g1, 32, 0, s>>>(qhat, latent, indices, j0, n_keys, qpos,
                                           kvlat, dim, scale, keys_per_split,
                                           part_acc, part_m, part_l);
    mla_decode_p2_kernel<<<(unsigned)n_heads, 256, 0, s>>>(part_acc, part_m, part_l,
                                                           S, kvlat, ohat);
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
