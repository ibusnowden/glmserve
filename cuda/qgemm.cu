// glmserve — GGUF quant GEMM: dequantize-on-the-fly weight @ activation.
//
// The real GLM-5.2 3-bit GGUF (UD-Q3_K_XL) keeps its weights in GGML block
// formats (Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_XS, F16) in VRAM and
// dequantizes inside the GEMM, exactly like llama.cpp's MMVQ/MMAQ kernels.
// Weight layout is [out, in] row-major with `in` contiguous (the GGUF linear
// view): row o lives at qW + o * row_bytes and holds `in` quantized elements.
//
// Work partition (MMVQ-style): ONE WARP per output row. The row is walked in
// 256-element groups; lane l dequantizes only its own 8-element fragment
// [l*8, l*8+8) of the group directly into registers (dq8), multiplies by the
// activation fragment, and the warp shuffle-reduces the partial sums. No
// shared-memory staging, no redundant decode work, coalesced quant reads.
// Prefill additionally tiles tokens (T=8) so each decoded weight fragment is
// reused across 8 tokens before being re-read.
//
// The MoE expert FFN only reads the ACTIVE top-k experts (not all 256): a
// two-kernel fused path (gate_up -> h_act -> down) that is the key advantage
// over a dense all-experts GEMM. Fragment dequantizers mirror
// src/gguf_quant.cpp exactly; the IQ3_XXS/IQ4_XS tables are seeded from the
// validated host tables so the GPU path matches the CPU reference.
#include "common.cuh"
#include "kernels.cuh"
#include "gguf_quant.hpp"

#include <cstdint>
#include <cstring>

namespace glmserve {
namespace cuda {

enum : uint32_t {
    QTYPE_F32 = 0, QTYPE_F16 = 1, QTYPE_Q8_0 = 8, QTYPE_Q3_K = 11,
    QTYPE_Q4_K = 12, QTYPE_Q5_K = 13, QTYPE_Q6_K = 14, QTYPE_IQ3_XXS = 18,
    QTYPE_IQ4_XS = 23
};

// Device copies of the host dequant tables (seeded once from gguf_quant.cpp).
__constant__ uint8_t d_iq3xxs_grid[256 * 4];
__constant__ uint8_t d_ksigns_iq2xs[128];
__constant__ uint8_t d_kmask_iq2xs[8];
__constant__ int8_t  d_kvalues_iq4nl[16];
static bool g_tables_seeded = false;

static void seed_tables() {
    if (g_tables_seeded) return;
    cudaMemcpyToSymbol(d_iq3xxs_grid, gguf_iq3xxs_grid_table(), 256 * 4);
    cudaMemcpyToSymbol(d_ksigns_iq2xs, gguf_ksigns_iq2xs_table(), 128);
    cudaMemcpyToSymbol(d_kmask_iq2xs, gguf_kmask_iq2xs_table(), 8);
    cudaMemcpyToSymbol(d_kvalues_iq4nl, gguf_kvalues_iq4nl_table(), 16);
    g_tables_seeded = true;
}

// f16 -> f32 via the hardware converter (correct for subnormals — a bit-trick
// predecessor mis-decoded subnormal scales, corrupting every tensor whose
// f16 block scale underflows f16 normal range).
__device__ __forceinline__ float f16d(const uint8_t* p) {
    __half h;
    memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Bytes covering one 256-element group of a row (multiple whole blocks for
// sub-256-element block types).
__device__ __host__ __forceinline__ int qgroup_bytes(uint32_t type) {
    switch (type) {
        case QTYPE_F32: return 1024;      // 256 * 4
        case QTYPE_F16: return 512;       // 256 * 2
        case QTYPE_Q8_0: return 272;      // 8 blocks * 34
        case QTYPE_Q3_K: return 110;
        case QTYPE_Q4_K: return 144;
        case QTYPE_Q5_K: return 176;
        case QTYPE_Q6_K: return 210;
        case QTYPE_IQ3_XXS: return 98;
        case QTYPE_IQ4_XS: return 136;
        default: return 0;
    }
}

// Dequantize the 8-element fragment [frag*8, frag*8+8) of the 256-element
// group at `grp` into w[8]. frag is the lane id (0..31). Semantics mirror
// dequant_block()/gguf_quant.cpp element-for-element; only the work partition
// differs (each lane decodes just its own fragment, in registers).
__device__ __forceinline__ void dq8(uint32_t type, const uint8_t* grp, int frag, float* w) {
    const int idx0 = frag << 3;   // first element of this fragment in [0, 256)
    switch (type) {
        case QTYPE_F32: {
            const float* p = (const float*)(grp) + idx0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = p[j];
            return;
        }
        case QTYPE_F16: {
            const uint8_t* p = grp + idx0 * 2;
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = f16d(p + 2 * j);
            return;
        }
        case QTYPE_Q8_0: {
            // 8 blocks of 32 elems (34 B each): fragment sits inside block frag/4.
            const uint8_t* blk = grp + (frag >> 2) * 34;
            const float d = f16d(blk);
            const int8_t* qs = (const int8_t*)(blk + 2) + ((frag & 3) << 3);
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = d * (float)qs[j];
            return;
        }
        case QTYPE_Q3_K: {
            const uint8_t* hmask = grp;
            const uint8_t* sb = grp + 96;             // 32 hmask + 64 q
            const float d_all = f16d(sb + 12);
            const int hn = idx0 >> 7;                  // 0/1: which 128-half
            const int rem = idx0 & 127;
            const int j = rem >> 5;                    // 0..3
            const int sub16 = (rem & 31) >> 4;         // 0/1: which 16-run
            const int qi = rem & 31;                   // q/hmask byte base
            uint32_t a0, a1, a2;
            memcpy(&a0, sb + 0, 4); memcpy(&a1, sb + 4, 4); memcpy(&a2, sb + 8, 4);
            const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
            uint32_t aux[4];
            aux[0] = (a0 & km2) | (((a2 >> 0) & km1) << 4);
            aux[1] = (a1 & km2) | (((a2 >> 2) & km1) << 4);
            aux[2] = ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4);
            aux[3] = ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4);
            const int is = hn * 8 + j * 2 + sub16;
            const int8_t sc = ((const int8_t*)aux)[is];
            const float dl = d_all * (float)(sc - 32);
            const uint8_t* q = grp + 32 + hn * 32 + qi;
            const uint8_t* hm = hmask + qi;
            const int shift = 2 * j;
            const uint8_t m = (uint8_t)(1u << (hn * 4 + j));
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2)
                w[j2] = dl * (float)((int8_t)((q[j2] >> shift) & 3) - ((hm[j2] & m) ? 0 : 4));
            return;
        }
        case QTYPE_Q4_K: {
            const float d = f16d(grp), dmin = f16d(grp + 2);
            const uint8_t* scales = grp + 4;
            const int j64 = idx0 >> 6;                 // 0..3
            const int half = (idx0 & 63) >> 5;         // 0/1: low/high nibble
            const int l = idx0 & 31;
            uint8_t sc, mm;
            get_scale_min_k4(j64 * 2 + half, scales, &sc, &mm);
            const float d1 = d * (float)sc, m1 = dmin * (float)mm;
            const uint8_t* q = grp + 16 + j64 * 32 + l;
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                const uint8_t b = q[j2];
                w[j2] = d1 * (float)(half ? (b >> 4) : (b & 0x0F)) - m1;
            }
            return;
        }
        case QTYPE_Q5_K: {
            const float d = f16d(grp), dmin = f16d(grp + 2);
            const uint8_t* scales = grp + 4;
            const uint8_t* qh = grp + 16;
            const uint8_t* ql = grp + 48;
            const int j64 = idx0 >> 6;
            const int half = (idx0 & 63) >> 5;
            const int l = idx0 & 31;
            uint8_t sc, mm;
            get_scale_min_k4(j64 * 2 + half, scales, &sc, &mm);
            const float d1 = d * (float)sc, m1 = dmin * (float)mm;
            const uint8_t u = (uint8_t)(1u << (j64 * 2 + half));
            const uint8_t* qlp = ql + j64 * 32 + l;
            const uint8_t* qhp = qh + l;
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                const uint8_t b = qlp[j2];
                const int lo = half ? (b >> 4) : (b & 0x0F);
                w[j2] = d1 * (float)(lo + ((qhp[j2] & u) ? 16 : 0)) - m1;
            }
            return;
        }
        case QTYPE_Q6_K: {
            const int hn = idx0 >> 7;                  // 0/1
            const int rem = idx0 & 127;
            const int k = rem >> 5;                    // 0..3
            const int l = rem & 31;
            const uint8_t* ql = grp + hn * 64;
            const uint8_t* qh = grp + 128 + hn * 32;
            const int8_t* sc = (const int8_t*)(grp + 192) + hn * 8;
            const float d = f16d(grp + 208);
            const int is = l >> 4;                     // constant across the 8-run
            const float dscale = d * (float)sc[is + 2 * k];
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                const int ll = l + j2;
                int8_t q;
                switch (k) {
                    case 0:  q = (int8_t)(((ql[ll] & 0x0F) | (((qh[ll] >> 0) & 3) << 4)) - 32); break;
                    case 1:  q = (int8_t)(((ql[ll + 32] & 0x0F) | (((qh[ll] >> 2) & 3) << 4)) - 32); break;
                    case 2:  q = (int8_t)(((ql[ll] >> 4) | (((qh[ll] >> 4) & 3) << 4)) - 32); break;
                    default: q = (int8_t)(((ql[ll + 32] >> 4) | (((qh[ll] >> 6) & 3) << 4)) - 32); break;
                }
                w[j2] = dscale * (float)q;
            }
            return;
        }
        case QTYPE_IQ3_XXS: {
            const float d = f16d(grp);
            const uint8_t* qs = grp + 2;
            const uint8_t* ss = qs + 64;               // QK_K/4 = 64
            const int ib32 = idx0 >> 5;                // 0..7
            const int l = (idx0 & 31) >> 3;            // 0..3: exactly this fragment
            uint32_t aux32;
            memcpy(&aux32, ss + 4 * ib32, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t* g1 = &d_iq3xxs_grid[qs[ib32 * 8 + 2 * l + 0] * 4];
            const uint8_t* g2 = &d_iq3xxs_grid[qs[ib32 * 8 + 2 * l + 1] * 4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                w[j + 0] = db * (float)g1[j] * ((signs & d_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                w[j + 4] = db * (float)g2[j] * ((signs & d_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            return;
        }
        case QTYPE_IQ4_XS: {
            const float d = f16d(grp);
            const uint16_t scales_h = (uint16_t)grp[2] | ((uint16_t)grp[3] << 8);
            const uint8_t* scales_l = grp + 4;
            const uint8_t* qs = grp + 8;               // 4 scale_l bytes
            const int ib32 = idx0 >> 5;                // 0..7
            const int within = idx0 & 31;
            const int half = within >> 4;              // 0: low nibble, 1: high
            const int j0 = within & 15;                // 0 or 8
            const uint8_t lo4 = (ib32 & 1) ? (scales_l[ib32 / 2] >> 4) : (scales_l[ib32 / 2] & 0x0F);
            const uint8_t hi2 = (scales_h >> (2 * ib32)) & 0x03;
            const int8_t scale = (int8_t)(lo4 | (hi2 << 4)) - 32;
            const float dl = d * (float)scale;
            const uint8_t* q = qs + ib32 * 16 + j0;
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                const uint8_t b = q[j2];
                w[j2] = dl * (float)d_kvalues_iq4nl[half ? (b >> 4) : (b & 0x0F)];
            }
            return;
        }
        default: {
            #pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = 0.0f;
            return;
        }
    }
}

// Dot-product of lane fragment weights against the activation fragment at
// x[idx0..idx0+8) with a tail guard (rem = elements remaining in the row).
__device__ __forceinline__ float frag_dot(const float* w, const float* xt, int rem) {
    if (rem >= 8) {
        const float4 x0 = *(const float4*)(xt);
        const float4 x1 = *(const float4*)(xt + 4);
        return w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w +
               w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
    }
    float acc = 0.0f;
    for (int j = 0; j < rem; ++j) acc += w[j] * xt[j];
    return acc;
}

constexpr int kWarpsPerBlock = 8;

// y[t0+t, o] = x[t0+t, :] @ W[o, :] (+bias) for t in [0, T). One warp per
// output row; lanes own disjoint 8-element fragments of each 256-element
// group. T tokens share each decoded fragment (register reuse for prefill).
template <int T>
__global__ void gemv_q_kernel(const float* __restrict__ x, const uint8_t* __restrict__ qW,
                              const float* __restrict__ bias, float* __restrict__ y,
                              int t0, int in, int out, int64_t row_bytes, uint32_t type) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * kWarpsPerBlock + warp;
    if (o >= out) return;
    const uint8_t* wrow = qW + (size_t)o * row_bytes;
    const int gbytes = qgroup_bytes(type);
    const int ngroups = (in + 255) >> 8;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    const float* xbase = x + (size_t)t0 * in;
    for (int g = 0; g < ngroups; ++g) {
        const int idx0 = (g << 8) + (lane << 3);
        const int rem = in - idx0;
        if (rem <= 0) continue;
        float w8[8];
        dq8(type, wrow + (size_t)g * gbytes, lane, w8);
        #pragma unroll
        for (int t = 0; t < T; ++t)
            acc[t] += frag_dot(w8, xbase + (size_t)t * in + idx0, rem);
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const float r = warp_reduce_sum(acc[t]);
        if (lane == 0) y[(size_t)(t0 + t) * out + o] = bias ? (r + bias[o]) : r;
    }
}

void gemm_q(uint32_t qtype, const float* x, const uint8_t* qW, const float* bias,
            float* y, int64_t n, int64_t in, int64_t out, int64_t row_bytes,
            cudaStream_t s) {
    seed_tables();
    const dim3 grid((unsigned)((out + kWarpsPerBlock - 1) / kWarpsPerBlock));
    const int threads = kWarpsPerBlock * 32;
    int64_t t0 = 0;
    while (n - t0 >= 8) {
        gemv_q_kernel<8><<<grid, threads, 0, s>>>(x, qW, bias, y, (int)t0, (int)in,
                                                  (int)out, row_bytes, qtype);
        t0 += 8;
    }
    if (n - t0 >= 4) {
        gemv_q_kernel<4><<<grid, threads, 0, s>>>(x, qW, bias, y, (int)t0, (int)in,
                                                  (int)out, row_bytes, qtype);
        t0 += 4;
    }
    if (n - t0 >= 2) {
        gemv_q_kernel<2><<<grid, threads, 0, s>>>(x, qW, bias, y, (int)t0, (int)in,
                                                  (int)out, row_bytes, qtype);
        t0 += 2;
    }
    if (n - t0 >= 1) {
        gemv_q_kernel<1><<<grid, threads, 0, s>>>(x, qW, bias, y, (int)t0, (int)in,
                                                  (int)out, row_bytes, qtype);
        t0 += 1;
    }
}

// Gather embeddings from a quant [vocab, H] table: one block per token; the 8
// warps stride the row's 256-element groups, each lane writing its fragment.
__global__ void embed_gather_q_kernel(const uint8_t* __restrict__ qtable, const int* __restrict__ tokens,
                                      float* __restrict__ hidden, int H, int64_t row_bytes,
                                      uint32_t type) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x;
    const uint8_t* wrow = qtable + (int64_t)tokens[t] * row_bytes;
    float* dst = hidden + (int64_t)t * H;
    const int gbytes = qgroup_bytes(type);
    const int ngroups = (H + 255) >> 8;
    for (int g = warp; g < ngroups; g += kWarpsPerBlock) {
        const int idx0 = (g << 8) + (lane << 3);
        const int rem = H - idx0;
        if (rem <= 0) continue;
        float w8[8];
        dq8(type, wrow + (size_t)g * gbytes, lane, w8);
        const int cnt = rem >= 8 ? 8 : rem;
        for (int j = 0; j < cnt; ++j) dst[idx0 + j] = w8[j];
    }
}

void embed_gather_q(uint32_t qtype, const uint8_t* qtable, const int* tokens,
                    float* hidden, int64_t n, int64_t H, int64_t row_bytes,
                    cudaStream_t s) {
    seed_tables();
    embed_gather_q_kernel<<<(unsigned)n, kWarpsPerBlock * 32, 0, s>>>(
        qtable, tokens, hidden, (int)H, row_bytes, qtype);
}

// Dequantize a single row (in elements) of a quant weight into fp32 dst.
__global__ void dequant_row_q_kernel(const uint8_t* __restrict__ qW, float* __restrict__ dst,
                                     int in, uint32_t type) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gbytes = qgroup_bytes(type);
    const int ngroups = (in + 255) >> 8;
    for (int g = warp; g < ngroups; g += kWarpsPerBlock) {
        const int idx0 = (g << 8) + (lane << 3);
        const int rem = in - idx0;
        if (rem <= 0) continue;
        float w8[8];
        dq8(type, qW + (size_t)g * gbytes, lane, w8);
        const int cnt = rem >= 8 ? 8 : rem;
        for (int j = 0; j < cnt; ++j) dst[idx0 + j] = w8[j];
    }
}

void dequant_row_q(uint32_t qtype, const uint8_t* qW, float* dst, int64_t in,
                   int64_t row_bytes, cudaStream_t s) {
    seed_tables();
    (void)row_bytes;
    dequant_row_q_kernel<<<1, kWarpsPerBlock * 32, 0, s>>>(qW, dst, (int)in, qtype);
}

// ---- MoE expert FFN (only active top-k experts are read) ------------------
// gate/up are [E, out=moe_inter, in=hidden] (row o of expert e at
// base + (e*moe_inter + o)*row_bytes); down is [E, out=hidden, in=moe_inter].
// Phase 1 (gate_up): grid (moe_inter/8, n*topk); one warp per (t, slot, f):
//   h_act[t, slot, f] = silu(gate_e[f] . x) * (up_e[f] . x).
// Phase 2 (down):    grid (hidden/8, n*topk); one warp per (t, slot, o):
//   out[t, o] += weight * (down_e[o] . h_act[t, slot]).

__global__ void moe_gate_up_q_kernel(const float* __restrict__ x, const int* __restrict__ topk_ids,
                                     const uint8_t* __restrict__ gate_q, const uint8_t* __restrict__ up_q,
                                     int topk, int hidden, int moe_inter,
                                     int64_t gate_row_bytes, int64_t up_row_bytes,
                                     uint32_t gate_type, uint32_t up_type,
                                     float* __restrict__ h_act) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f = blockIdx.x * kWarpsPerBlock + warp;
    if (f >= moe_inter) return;
    const int ts = blockIdx.y;                  // t * topk + slot
    const int t = ts / topk;
    const int e = topk_ids[ts];
    const float* xt = x + (size_t)t * hidden;
    const uint8_t* grow = gate_q + ((size_t)e * moe_inter + f) * gate_row_bytes;
    const uint8_t* urow = up_q + ((size_t)e * moe_inter + f) * up_row_bytes;
    const int g_gbytes = qgroup_bytes(gate_type);
    const int u_gbytes = qgroup_bytes(up_type);
    const int ngroups = (hidden + 255) >> 8;
    float g = 0.0f, u = 0.0f;
    for (int gi = 0; gi < ngroups; ++gi) {
        const int idx0 = (gi << 8) + (lane << 3);
        const int rem = hidden - idx0;
        if (rem <= 0) continue;
        float w8[8];
        dq8(gate_type, grow + (size_t)gi * g_gbytes, lane, w8);
        g += frag_dot(w8, xt + idx0, rem);
        dq8(up_type, urow + (size_t)gi * u_gbytes, lane, w8);
        u += frag_dot(w8, xt + idx0, rem);
    }
    g = warp_reduce_sum(g);
    u = warp_reduce_sum(u);
    if (lane == 0)
        h_act[(size_t)ts * moe_inter + f] = silu(g) * u;
}

__global__ void moe_down_q_kernel(const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
                                  const uint8_t* __restrict__ down_q,
                                  const float* __restrict__ h_act, int topk, int hidden, int moe_inter,
                                  int64_t down_row_bytes, uint32_t down_type,
                                  float* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * kWarpsPerBlock + warp;
    if (o >= hidden) return;
    const int ts = blockIdx.y;
    const int t = ts / topk;
    const int e = topk_ids[ts];
    const float weight = topk_w[ts];
    const uint8_t* drow = down_q + ((size_t)e * hidden + o) * down_row_bytes;
    const float* ha = h_act + (size_t)ts * moe_inter;
    const int gbytes = qgroup_bytes(down_type);
    const int ngroups = (moe_inter + 255) >> 8;
    float acc = 0.0f;
    for (int gi = 0; gi < ngroups; ++gi) {
        const int idx0 = (gi << 8) + (lane << 3);
        const int rem = moe_inter - idx0;
        if (rem <= 0) continue;
        float w8[8];
        dq8(down_type, drow + (size_t)gi * gbytes, lane, w8);
        acc += frag_dot(w8, ha + idx0, rem);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0)
        atomicAdd(&out[(size_t)t * hidden + o], weight * acc);
}

// ---- expert-major prefill path ---------------------------------------------
// The token-major kernels above launch one warp per (token, slot, row): every
// (token, slot) re-dequantizes the full expert — at prefill (n=1024, topk=8,
// E=160) each expert's weights are decoded ~51x per layer, and the MoE FFN was
// 56% of the whole prefill. Expert-major instead groups the (t, slot) pairs by
// expert (counts -> offsets -> scatter) and tiles 8 tokens per weight fragment
// (same register-reuse trick as gemv_q_kernel<8>), cutting quant reads ~8x.
// Per-(ts, row) accumulation order is identical to the token-major kernels, so
// h_act matches bit-for-bit; the down accumulation uses the same atomicAdd.

__global__ void moe_dispatch_count_kernel(const int* __restrict__ topk_ids, int nts,
                                          int* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nts) atomicAdd(&counts[topk_ids[i]], 1);
}

// E <= 256: a serial scan on one thread is cheaper than its launch.
__global__ void moe_dispatch_scan_kernel(const int* __restrict__ counts, int E,
                                         int* __restrict__ offsets) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < E; ++e) { offsets[e] = acc; acc += counts[e]; }
    offsets[E] = acc;
}

__global__ void moe_dispatch_scatter_kernel(const int* __restrict__ topk_ids, int nts,
                                            const int* __restrict__ offsets,
                                            int* __restrict__ cursor,
                                            int* __restrict__ ts_sorted) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nts) return;
    const int e = topk_ids[i];
    ts_sorted[offsets[e] + atomicAdd(&cursor[e], 1)] = i;
}

__global__ void moe_gate_up_q_emajor_kernel(const float* __restrict__ x,
                                            const int* __restrict__ ts_sorted,
                                            const int* __restrict__ offsets,
                                            const uint8_t* __restrict__ gate_q,
                                            const uint8_t* __restrict__ up_q,
                                            int topk, int hidden, int moe_inter,
                                            int64_t gate_row_bytes, int64_t up_row_bytes,
                                            uint32_t gate_type, uint32_t up_type,
                                            float* __restrict__ h_act) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f = blockIdx.x * kWarpsPerBlock + warp;
    if (f >= moe_inter) return;
    const int e = blockIdx.y;
    const int beg = offsets[e], end = offsets[e + 1];
    if (beg == end) return;
    const uint8_t* grow = gate_q + ((size_t)e * moe_inter + f) * gate_row_bytes;
    const uint8_t* urow = up_q + ((size_t)e * moe_inter + f) * up_row_bytes;
    const int g_gbytes = qgroup_bytes(gate_type);
    const int u_gbytes = qgroup_bytes(up_type);
    const int ngroups = (hidden + 255) >> 8;
    for (int base = beg; base < end; base += 8) {
        const int B = min(8, end - base);
        int ts[8];
        const float* xt[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < B) {
                ts[j] = ts_sorted[base + j];
                xt[j] = x + (size_t)(ts[j] / topk) * hidden;
            }
        }
        float g8[8], u8[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) { g8[j] = 0.0f; u8[j] = 0.0f; }
        for (int gi = 0; gi < ngroups; ++gi) {
            const int idx0 = (gi << 8) + (lane << 3);
            const int rem = hidden - idx0;
            if (rem <= 0) continue;
            float w8[8];
            dq8(gate_type, grow + (size_t)gi * g_gbytes, lane, w8);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < B) g8[j] += frag_dot(w8, xt[j] + idx0, rem);
            dq8(up_type, urow + (size_t)gi * u_gbytes, lane, w8);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < B) u8[j] += frag_dot(w8, xt[j] + idx0, rem);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= B) continue;
            const float g = warp_reduce_sum(g8[j]);
            const float u = warp_reduce_sum(u8[j]);
            if (lane == 0) h_act[(size_t)ts[j] * moe_inter + f] = silu(g) * u;
        }
    }
}

__global__ void moe_down_q_emajor_kernel(const int* __restrict__ ts_sorted,
                                         const int* __restrict__ offsets,
                                         const float* __restrict__ topk_w,
                                         const uint8_t* __restrict__ down_q,
                                         const float* __restrict__ h_act,
                                         int hidden, int moe_inter,
                                         int64_t down_row_bytes, uint32_t down_type,
                                         float* __restrict__ out, int topk) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * kWarpsPerBlock + warp;
    if (o >= hidden) return;
    const int e = blockIdx.y;
    const int beg = offsets[e], end = offsets[e + 1];
    if (beg == end) return;
    const uint8_t* drow = down_q + ((size_t)e * hidden + o) * down_row_bytes;
    const int gbytes = qgroup_bytes(down_type);
    const int ngroups = (moe_inter + 255) >> 8;
    for (int base = beg; base < end; base += 8) {
        const int B = min(8, end - base);
        int ts[8];
        const float* ha[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < B) {
                ts[j] = ts_sorted[base + j];
                ha[j] = h_act + (size_t)ts[j] * moe_inter;
            }
        }
        float a8[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) a8[j] = 0.0f;
        for (int gi = 0; gi < ngroups; ++gi) {
            const int idx0 = (gi << 8) + (lane << 3);
            const int rem = moe_inter - idx0;
            if (rem <= 0) continue;
            float w8[8];
            dq8(down_type, drow + (size_t)gi * gbytes, lane, w8);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < B) a8[j] += frag_dot(w8, ha[j] + idx0, rem);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= B) continue;
            const float acc = warp_reduce_sum(a8[j]);
            if (lane == 0)
                atomicAdd(&out[(size_t)(ts[j] / topk) * hidden + o], topk_w[ts[j]] * acc);
        }
    }
}

void moe_expert_ffn_q(uint32_t gate_type, uint32_t up_type, uint32_t down_type,
                      const float* x, const int* topk_ids, const float* topk_w,
                      const uint8_t* gate_q, const uint8_t* up_q, const uint8_t* down_q,
                      int n, int topk, int hidden, int moe_inter, int E,
                      int64_t gate_row_bytes, int64_t up_row_bytes, int64_t down_row_bytes,
                      float* h_act, float* out, int* dispatch, cudaStream_t s) {
    seed_tables();
    cudaMemsetAsync(out, 0, (size_t)n * hidden * sizeof(float), s);
    const int threads = kWarpsPerBlock * 32;
    const int nts = n * topk;
    if (dispatch && nts >= 64) {
        // Expert-major: dispatch layout [counts E | offsets E+1 | cursor E | ts nts].
        int* counts = dispatch;
        int* offsets = dispatch + E;
        int* cursor = dispatch + 2 * E + 1;
        int* ts_sorted = dispatch + 3 * E + 1;
        cudaMemsetAsync(counts, 0, (size_t)E * sizeof(int), s);
        cudaMemsetAsync(cursor, 0, (size_t)E * sizeof(int), s);
        const unsigned db = (unsigned)((nts + 255) / 256);
        moe_dispatch_count_kernel<<<db, 256, 0, s>>>(topk_ids, nts, counts);
        moe_dispatch_scan_kernel<<<1, 32, 0, s>>>(counts, E, offsets);
        moe_dispatch_scatter_kernel<<<db, 256, 0, s>>>(topk_ids, nts, offsets, cursor,
                                                       ts_sorted);
        dim3 gu((unsigned)((moe_inter + kWarpsPerBlock - 1) / kWarpsPerBlock), (unsigned)E);
        moe_gate_up_q_emajor_kernel<<<gu, threads, 0, s>>>(
            x, ts_sorted, offsets, gate_q, up_q, topk, hidden, moe_inter,
            gate_row_bytes, up_row_bytes, gate_type, up_type, h_act);
        dim3 dn((unsigned)((hidden + kWarpsPerBlock - 1) / kWarpsPerBlock), (unsigned)E);
        moe_down_q_emajor_kernel<<<dn, threads, 0, s>>>(
            ts_sorted, offsets, topk_w, down_q, h_act, hidden, moe_inter,
            down_row_bytes, down_type, out, topk);
        return;
    }
    dim3 gu((unsigned)((moe_inter + kWarpsPerBlock - 1) / kWarpsPerBlock),
            (unsigned)(n * topk));
    moe_gate_up_q_kernel<<<gu, threads, 0, s>>>(
        x, topk_ids, gate_q, up_q, topk, hidden, moe_inter,
        gate_row_bytes, up_row_bytes, gate_type, up_type, h_act);
    dim3 dn((unsigned)((hidden + kWarpsPerBlock - 1) / kWarpsPerBlock),
            (unsigned)(n * topk));
    moe_down_q_kernel<<<dn, threads, 0, s>>>(
        topk_ids, topk_w, down_q, h_act, topk, hidden, moe_inter,
        down_row_bytes, down_type, out);
}

}  // namespace cuda
}  // namespace glmserve
