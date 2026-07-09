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
#include <cstdlib>
#include <cstring>

namespace glmserve {
namespace cuda {

enum : uint32_t {
    QTYPE_F32 = 0, QTYPE_F16 = 1, QTYPE_Q8_0 = 8, QTYPE_Q3_K = 11,
    QTYPE_Q4_K = 12, QTYPE_Q5_K = 13, QTYPE_Q6_K = 14, QTYPE_IQ3_XXS = 18,
    QTYPE_IQ4_XS = 23
};

// Device copies of the host dequant tables (seeded once from gguf_quant.cpp).
// These are indexed DIVERGENTLY across a warp (every lane decodes a different
// fragment), so they must live in __device__ global memory: __constant__
// serializes on distinct addresses (one broadcast per cycle) and was the
// dominant cost of the IQ3_XXS/IQ4_XS streaming paths. Only kmask (indexed by
// a compile-time-unrolled loop counter, uniform) stays in constant memory.
__device__ uint32_t d_iq3xxs_grid[256];
__device__ uint8_t  d_ksigns_iq2xs[128];
__constant__ uint8_t d_kmask_iq2xs[8];
__device__ int8_t   d_kvalues_iq4nl[16];
// Per-7-bit-sign-code byte masks (0x00/0xFF per element): x = (g ^ m) - m
// conditionally negates the four packed grid bytes in two instructions.
__device__ uint2    d_signmask_iq2xs[128];
static bool g_tables_seeded = false;

static void seed_tables() {
    if (g_tables_seeded) return;
    cudaMemcpyToSymbol(d_iq3xxs_grid, gguf_iq3xxs_grid_table(), 256 * 4);
    cudaMemcpyToSymbol(d_ksigns_iq2xs, gguf_ksigns_iq2xs_table(), 128);
    cudaMemcpyToSymbol(d_kmask_iq2xs, gguf_kmask_iq2xs_table(), 8);
    cudaMemcpyToSymbol(d_kvalues_iq4nl, gguf_kvalues_iq4nl_table(), 16);
    const uint8_t* ks = gguf_ksigns_iq2xs_table();
    uint32_t sm[128][2];
    for (int c = 0; c < 128; ++c) {
        uint32_t lo = 0, hi = 0;
        for (int j = 0; j < 4; ++j) {
            if ((ks[c] >> j) & 1)       lo |= 0xFFu << (8 * j);
            if ((ks[c] >> (4 + j)) & 1) hi |= 0xFFu << (8 * j);
        }
        sm[c][0] = lo;
        sm[c][1] = hi;
    }
    cudaMemcpyToSymbol(d_signmask_iq2xs, sm, sizeof(sm));
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

// ---- int8 MMQ (dp4a) path --------------------------------------------------
// The fp32 kernels are ALU-bound in dq8's int->float conversions and fp32 FMAs
// (decode moe.experts measured ~60 GB/s effective vs ~800 peak). The MMQ path
// quantizes activations to int8 per 32-element block (llama.cpp Q8_1-style)
// and keeps the weight fragment as INTEGERS + one affine scale pair, so the
// 8-element dot is two dp4a instructions:
//   weight[j] = ds * wq[j] - dm      (dm != 0 only for Q4_K/Q5_K minima)
//   x[j]      = sx * xq[j]
//   dot      += sx * (ds * dp4a(wq, xq) - dm * dp4a(1, xq))
// Every GGUF type in the real checkpoint has an integer fragment form; only
// F16/F32 tensors fall back to the fp32 kernels (qtype_has_i8 == false).

// Decoded int8 weight fragment: 8 lanes packed into two dp4a operands.
struct WI8 { int w0, w1; float ds, dm; };

__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Integer twin of dq8(): decode the 8-element fragment [frag*8, frag*8+8) of
// the 256-element group at `grp` into WI8. Element values and scales mirror
// dq8 exactly; only the (int, affine) factorization differs.
__device__ __forceinline__ void dq8i(uint32_t type, const uint8_t* grp, int frag, WI8& w) {
    const int idx0 = frag << 3;
    w.dm = 0.0f;
    switch (type) {
        case QTYPE_Q8_0: {
            const uint8_t* blk = grp + (frag >> 2) * 34;
            w.ds = f16d(blk);
            memcpy(&w.w0, blk + 2 + ((frag & 3) << 3), 4);
            memcpy(&w.w1, blk + 6 + ((frag & 3) << 3), 4);
            return;
        }
        case QTYPE_Q3_K: {
            const uint8_t* hmask = grp;
            const uint8_t* sb = grp + 96;
            const float d_all = f16d(sb + 12);
            const int hn = idx0 >> 7;
            const int rem = idx0 & 127;
            const int j = rem >> 5;
            const int sub16 = (rem & 31) >> 4;
            const int qi = rem & 31;
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
            w.ds = d_all * (float)(sc - 32);
            const uint8_t* q = grp + 32 + hn * 32 + qi;
            const uint8_t* hm = hmask + qi;
            const int shift = 2 * j;
            const uint8_t m = (uint8_t)(1u << (hn * 4 + j));
            int v[8];
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2)
                v[j2] = (int)((q[j2] >> shift) & 3) - ((hm[j2] & m) ? 0 : 4);
            w.w0 = pack4(v[0], v[1], v[2], v[3]);
            w.w1 = pack4(v[4], v[5], v[6], v[7]);
            return;
        }
        case QTYPE_Q4_K: {
            const float d = f16d(grp), dmin = f16d(grp + 2);
            const uint8_t* scales = grp + 4;
            const int j64 = idx0 >> 6;
            const int half = (idx0 & 63) >> 5;
            const int l = idx0 & 31;
            uint8_t sc, mm;
            get_scale_min_k4(j64 * 2 + half, scales, &sc, &mm);
            w.ds = d * (float)sc;
            w.dm = dmin * (float)mm;
            const uint8_t* q = grp + 16 + j64 * 32 + l;
            int v[8];
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2)
                v[j2] = half ? (q[j2] >> 4) : (q[j2] & 0x0F);
            w.w0 = pack4(v[0], v[1], v[2], v[3]);
            w.w1 = pack4(v[4], v[5], v[6], v[7]);
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
            w.ds = d * (float)sc;
            w.dm = dmin * (float)mm;
            const uint8_t u = (uint8_t)(1u << (j64 * 2 + half));
            const uint8_t* qlp = ql + j64 * 32 + l;
            const uint8_t* qhp = qh + l;
            int v[8];
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                const uint8_t b = qlp[j2];
                v[j2] = (half ? (b >> 4) : (b & 0x0F)) + ((qhp[j2] & u) ? 16 : 0);
            }
            w.w0 = pack4(v[0], v[1], v[2], v[3]);
            w.w1 = pack4(v[4], v[5], v[6], v[7]);
            return;
        }
        case QTYPE_Q6_K: {
            const int hn = idx0 >> 7;
            const int rem = idx0 & 127;
            const int k = rem >> 5;
            const int l = rem & 31;
            const uint8_t* ql = grp + hn * 64;
            const uint8_t* qh = grp + 128 + hn * 32;
            const int8_t* sc = (const int8_t*)(grp + 192) + hn * 8;
            const float d = f16d(grp + 208);
            const int is = l >> 4;
            w.ds = d * (float)sc[is + 2 * k];
            int v[8];
            #pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                const int ll = l + j2;
                switch (k) {
                    case 0:  v[j2] = (int)((ql[ll] & 0x0F) | (((qh[ll] >> 0) & 3) << 4)) - 32; break;
                    case 1:  v[j2] = (int)((ql[ll + 32] & 0x0F) | (((qh[ll] >> 2) & 3) << 4)) - 32; break;
                    case 2:  v[j2] = (int)((ql[ll] >> 4) | (((qh[ll] >> 4) & 3) << 4)) - 32; break;
                    default: v[j2] = (int)((ql[ll + 32] >> 4) | (((qh[ll] >> 6) & 3) << 4)) - 32; break;
                }
            }
            w.w0 = pack4(v[0], v[1], v[2], v[3]);
            w.w1 = pack4(v[4], v[5], v[6], v[7]);
            return;
        }
        case QTYPE_IQ3_XXS: {
            const float d = f16d(grp);
            const uint8_t* qs = grp + 2;
            const uint8_t* ss = qs + 64;
            const int ib32 = idx0 >> 5;
            const int l = (idx0 & 31) >> 3;
            uint32_t aux32;
            memcpy(&aux32, ss + 4 * ib32, 4);
            w.ds = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            // (g ^ m) - m negates exactly the bytes whose sign bit is set
            // (grid values are <= 62, so the byte negate cannot wrap).
            const uint2 m = d_signmask_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint32_t g1 = d_iq3xxs_grid[qs[ib32 * 8 + 2 * l + 0]];
            const uint32_t g2 = d_iq3xxs_grid[qs[ib32 * 8 + 2 * l + 1]];
            w.w0 = (int)__vsub4(g1 ^ m.x, m.x);
            w.w1 = (int)__vsub4(g2 ^ m.y, m.y);
            return;
        }
        case QTYPE_IQ4_XS: {
            const float d = f16d(grp);
            const uint16_t scales_h = (uint16_t)grp[2] | ((uint16_t)grp[3] << 8);
            const uint8_t* scales_l = grp + 4;
            const uint8_t* qs = grp + 8;
            const int ib32 = idx0 >> 5;
            const int within = idx0 & 31;
            const int half = within >> 4;
            const int j0 = within & 15;
            const uint8_t lo4 = (ib32 & 1) ? (scales_l[ib32 / 2] >> 4) : (scales_l[ib32 / 2] & 0x0F);
            const uint8_t hi2 = (scales_h >> (2 * ib32)) & 0x03;
            const int8_t scale = (int8_t)(lo4 | (hi2 << 4)) - 32;
            w.ds = d * (float)scale;
            const uint8_t* q = qs + ib32 * 16 + j0;
            uint32_t q0, q1;
            memcpy(&q0, q, 4);
            memcpy(&q1, q + 4, 4);
            const int sh = half * 4;
            const uint32_t n0 = (q0 >> sh) & 0x0F0F0F0Fu;
            const uint32_t n1 = (q1 >> sh) & 0x0F0F0F0Fu;
            w.w0 = pack4(d_kvalues_iq4nl[n0 & 0xFF], d_kvalues_iq4nl[(n0 >> 8) & 0xFF],
                         d_kvalues_iq4nl[(n0 >> 16) & 0xFF], d_kvalues_iq4nl[n0 >> 24]);
            w.w1 = pack4(d_kvalues_iq4nl[n1 & 0xFF], d_kvalues_iq4nl[(n1 >> 8) & 0xFF],
                         d_kvalues_iq4nl[(n1 >> 16) & 0xFF], d_kvalues_iq4nl[n1 >> 24]);
            return;
        }
        default: {
            w.w0 = w.w1 = 0; w.ds = 0.0f;
            return;
        }
    }
}

// dot(weight fragment, activation fragment): two dp4a's, one fp32 FMA (plus a
// correction pair for Q4_K/Q5_K minima). xq must be 8-byte aligned (idx0 is a
// multiple of 8 and rows are 32-multiples).
__device__ __forceinline__ float frag_dot_i8(const WI8& w, const int8_t* xq, float sx) {
    int2 xi;
    memcpy(&xi, xq, 8);
    const int sumi = __dp4a(w.w1, xi.y, __dp4a(w.w0, xi.x, 0));
    float r = w.ds * (float)sumi;
    if (w.dm != 0.0f) {
        const int sumx = __dp4a(0x01010101, xi.y, __dp4a(0x01010101, xi.x, 0));
        r -= w.dm * (float)sumx;
    }
    return sx * r;
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
            const uint8_t* g1 =
                reinterpret_cast<const uint8_t*>(&d_iq3xxs_grid[qs[ib32 * 8 + 2 * l + 0]]);
            const uint8_t* g2 =
                reinterpret_cast<const uint8_t*>(&d_iq3xxs_grid[qs[ib32 * 8 + 2 * l + 1]]);
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

// Expert-major shared-decode tiles engage above this many token-slots: below
// it experts see too few tokens for the 64-token tile to amortize anything and
// the (rows x E) grid only adds launch overhead.
constexpr int kMoeSmemMinTs = 2048;

// GLMSERVE_SMEM=1 enables the shared-decode (non-tensor-core) tile kernels.
// Default OFF: the mma path supersedes them for every prefill-hot case, and
// the stub A/B showed the smem MoE variant REGRESSES vs the register-tile
// kernel (135 -> 227 ms — prefill MoE is instruction-bound, not decode-bound).
static bool smem_kernels_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("GLMSERVE_SMEM");
        return v && *v && *v == '1';
    }();
    return on;
}

bool qtype_has_i8(uint32_t qtype) {
    switch (qtype) {
        case QTYPE_Q8_0: case QTYPE_Q3_K: case QTYPE_Q4_K: case QTYPE_Q5_K:
        case QTYPE_Q6_K: case QTYPE_IQ3_XXS: case QTYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// Quantize activations to int8, one scale per 32-element block (Q8_1-style):
// xq[t, i] = round(x[t, i] / sx),  xs[t, b] = sx = absmax(block b) / 127.
// One warp per block; `in` must be a multiple of 32 (every GGUF quant row is).
__global__ void quantize_act_q8_kernel(const float* __restrict__ x, int64_t nblocks,
                                       int8_t* __restrict__ xq, float* __restrict__ xs) {
    const int64_t b = (int64_t)blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (b >= nblocks) return;
    const int lane = threadIdx.x & 31;
    const float v = x[b * 32 + lane];
    const float amax = warp_reduce_max(fabsf(v));
    const float sx = amax / 127.0f;
    const float id = sx > 0.0f ? 1.0f / sx : 0.0f;
    xq[b * 32 + lane] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) xs[b] = sx;
}

void quantize_act_q8(const float* x, int64_t n, int64_t in, int8_t* xq, float* xs,
                     cudaStream_t s) {
    const int64_t nblocks = n * (in >> 5);
    const unsigned grid = (unsigned)((nblocks + kWarpsPerBlock - 1) / kWarpsPerBlock);
    quantize_act_q8_kernel<<<grid, kWarpsPerBlock * 32, 0, s>>>(x, nblocks, xq, xs);
}

// Per-token activation view, templated fp32 / int8 (the kernels below carry
// both instantiations; the int8 one replaces dq8+fp32 FMAs with dq8i+dp4a).
template <bool I8>
struct ActView;
template <>
struct ActView<false> {
    const float* row;
    __device__ ActView(const float* x, const int8_t*, const float*, size_t t, int in)
        : row(x + t * (size_t)in) {}
};
template <>
struct ActView<true> {
    const int8_t* row;
    const float* srow;
    __device__ ActView(const float*, const int8_t* xq, const float* xs, size_t t, int in)
        : row(xq + t * (size_t)in), srow(xs + t * (size_t)(in >> 5)) {}
};

template <bool I8>
struct WFrag;
template <>
struct WFrag<false> {
    float w[8];
    __device__ void decode(uint32_t type, const uint8_t* grp, int lane) {
        dq8(type, grp, lane, w);
    }
    __device__ float dot(const ActView<false>& a, int idx0, int rem) const {
        return frag_dot(w, a.row + idx0, rem);
    }
};
template <>
struct WFrag<true> {
    WI8 w;
    __device__ void decode(uint32_t type, const uint8_t* grp, int lane) {
        dq8i(type, grp, lane, w);
    }
    __device__ float dot(const ActView<true>& a, int idx0, int) const {
        return frag_dot_i8(w, a.row + idx0, a.srow[idx0 >> 5]);
    }
};

// y[t0+t, o] = x[t0+t, :] @ W[o, :] (+bias) for t in [0, T). One warp per
// output row; lanes own disjoint 8-element fragments of each 256-element
// group. T tokens share each decoded fragment (register reuse for prefill).
template <int T, bool I8>
__global__ void gemv_q_kernel(const float* __restrict__ x, const int8_t* __restrict__ xq,
                              const float* __restrict__ xs, const uint8_t* __restrict__ qW,
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
    for (int g = 0; g < ngroups; ++g) {
        const int idx0 = (g << 8) + (lane << 3);
        const int rem = in - idx0;
        if (rem <= 0) continue;
        WFrag<I8> wf;
        wf.decode(type, wrow + (size_t)g * gbytes, lane);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            ActView<I8> a(x, xq, xs, (size_t)(t0 + t), in);
            acc[t] += wf.dot(a, idx0, rem);
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const float r = warp_reduce_sum(acc[t]);
        if (lane == 0) y[(size_t)(t0 + t) * out + o] = bias ? (r + bias[o]) : r;
    }
}

// Shared-decode prefill GEMM: one output row x one 64-token tile per block.
// Each of the 8 warps decodes one 256-element group of the row into shared
// WFrags, then every warp dots the whole decoded chunk against its own 8
// tokens — the decode (the expensive half of dq8/dq8i) runs once per 64
// tokens instead of once per 8. Per token the groups still accumulate in
// ascending order into the same per-lane partials, so results are bitwise
// identical to gemv_q_kernel.
template <bool I8>
__global__ void gemm_q_smem_kernel(const float* __restrict__ x, const int8_t* __restrict__ xq,
                                   const float* __restrict__ xs, const uint8_t* __restrict__ qW,
                                   const float* __restrict__ bias, float* __restrict__ y,
                                   int n, int in, int out, int64_t row_bytes, uint32_t type) {
    constexpr int kChunk = kWarpsPerBlock;
    constexpr int kTok = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x;
    const uint8_t* wrow = qW + (size_t)o * row_bytes;
    const int gbytes = qgroup_bytes(type);
    const int ngroups = (in + 255) >> 8;
    __shared__ WFrag<I8> sh[kChunk][32];
    const int tb = blockIdx.y * (kChunk * kTok) + warp * kTok;
    const int B = min(kTok, n - tb);  // <= 0 marks an idle warp (still syncs)
    float acc[kTok];
    #pragma unroll
    for (int t = 0; t < kTok; ++t) acc[t] = 0.0f;
    for (int gc = 0; gc < ngroups; gc += kChunk) {
        const int gd = gc + warp;
        if (gd < ngroups && ((gd << 8) + (lane << 3)) < in)
            sh[warp][lane].decode(type, wrow + (size_t)gd * gbytes, lane);
        __syncthreads();
        const int gn = min(kChunk, ngroups - gc);
        for (int gi = 0; gi < gn; ++gi) {
            const int idx0 = ((gc + gi) << 8) + (lane << 3);
            const int rem = in - idx0;
            if (rem <= 0) continue;
            #pragma unroll
            for (int t = 0; t < kTok; ++t)
                if (t < B) acc[t] += sh[gi][lane].dot(
                    ActView<I8>(x, xq, xs, (size_t)(tb + t), in), idx0, rem);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < kTok; ++t) {
        if (t >= B) continue;
        const float r = warp_reduce_sum(acc[t]);
        if (lane == 0) y[(size_t)(tb + t) * out + o] = bias ? (r + bias[o]) : r;
    }
}

// Defined with the other mma kernels below.
static bool mma_type_ok(uint32_t type);
static bool mma_kernels_enabled();
__global__ void gemm_q_mma_s8_kernel(const int8_t* xq, const float* xs,
                                     const uint8_t* qW, const float* bias, float* y,
                                     int n, int K, int out, int64_t row_bytes,
                                     uint32_t type);

template <bool I8>
static void gemm_q_launch(uint32_t qtype, const float* x, const int8_t* xq, const float* xs,
                          const uint8_t* qW, const float* bias, float* y, int64_t n,
                          int64_t in, int64_t out, int64_t row_bytes, cudaStream_t s) {
    seed_tables();
    const dim3 grid((unsigned)((out + kWarpsPerBlock - 1) / kWarpsPerBlock));
    const int threads = kWarpsPerBlock * 32;
    if (I8 && n >= 64 && mma_kernels_enabled() && mma_type_ok(qtype) &&
        (out & 15) == 0 && (in & 255) == 0) {
        const dim3 gm((unsigned)(out / 16), (unsigned)((n + 63) / 64));
        gemm_q_mma_s8_kernel<<<gm, 256, 0, s>>>(xq, xs, qW, bias, y, (int)n,
                                                (int)in, (int)out, row_bytes, qtype);
        return;
    }
    if (n >= 64 && smem_kernels_enabled()) {
        // Shared-decode tile kernel: covers all n in one launch (last tile
        // partial; idle warps keep the barriers uniform).
        const dim3 g2((unsigned)out, (unsigned)((n + 63) / 64));
        gemm_q_smem_kernel<I8><<<g2, threads, 0, s>>>(x, xq, xs, qW, bias, y, (int)n,
                                                      (int)in, (int)out, row_bytes, qtype);
        return;
    }
    int64_t t0 = 0;
    while (n - t0 >= 8) {
        gemv_q_kernel<8, I8><<<grid, threads, 0, s>>>(x, xq, xs, qW, bias, y, (int)t0,
                                                      (int)in, (int)out, row_bytes, qtype);
        t0 += 8;
    }
    if (n - t0 >= 4) {
        gemv_q_kernel<4, I8><<<grid, threads, 0, s>>>(x, xq, xs, qW, bias, y, (int)t0,
                                                      (int)in, (int)out, row_bytes, qtype);
        t0 += 4;
    }
    if (n - t0 >= 2) {
        gemv_q_kernel<2, I8><<<grid, threads, 0, s>>>(x, xq, xs, qW, bias, y, (int)t0,
                                                      (int)in, (int)out, row_bytes, qtype);
        t0 += 2;
    }
    if (n - t0 >= 1) {
        gemv_q_kernel<1, I8><<<grid, threads, 0, s>>>(x, xq, xs, qW, bias, y, (int)t0,
                                                      (int)in, (int)out, row_bytes, qtype);
        t0 += 1;
    }
}

void gemm_q(uint32_t qtype, const float* x, const uint8_t* qW, const float* bias,
            float* y, int64_t n, int64_t in, int64_t out, int64_t row_bytes,
            cudaStream_t s) {
    gemm_q_launch<false>(qtype, x, nullptr, nullptr, qW, bias, y, n, in, out, row_bytes, s);
}

void gemm_q_i8(uint32_t qtype, const int8_t* xq, const float* xs, const uint8_t* qW,
               const float* bias, float* y, int64_t n, int64_t in, int64_t out,
               int64_t row_bytes, cudaStream_t s) {
    gemm_q_launch<true>(qtype, nullptr, xq, xs, qW, bias, y, n, in, out, row_bytes, s);
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

// Batch-dequantize a quant weight [rows, in] into fp16 (one block per row):
// used to keep kv_b resident in fp16 for the absorbed-MLA decode path.
__global__ void dequant_rows_f16_kernel(const uint8_t* __restrict__ qW, int64_t in,
                                        int64_t row_bytes, uint32_t type,
                                        __half* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t r = blockIdx.x;
    const uint8_t* wrow = qW + (size_t)r * row_bytes;
    const int gbytes = qgroup_bytes(type);
    const int ngroups = (int)((in + 255) >> 8);
    for (int g = warp; g < ngroups; g += kWarpsPerBlock) {
        const int idx0 = (g << 8) + (lane << 3);
        const int64_t rem = in - idx0;
        if (rem <= 0) continue;
        float w8[8];
        dq8(type, wrow + (size_t)g * gbytes, lane, w8);
        const int cnt = rem >= 8 ? 8 : (int)rem;
        for (int j = 0; j < cnt; ++j)
            out[(size_t)r * in + idx0 + j] = __float2half(w8[j]);
    }
}

void dequant_rows_f16(uint32_t qtype, const uint8_t* qW, int64_t rows, int64_t in,
                      int64_t row_bytes, __half* out, cudaStream_t s) {
    seed_tables();
    dequant_rows_f16_kernel<<<(unsigned)rows, kWarpsPerBlock * 32, 0, s>>>(
        qW, in, row_bytes, qtype, out);
}

// ---- MoE expert FFN (only active top-k experts are read) ------------------
// gate/up are [E, out=moe_inter, in=hidden] (row o of expert e at
// base + (e*moe_inter + o)*row_bytes); down is [E, out=hidden, in=moe_inter].
// Phase 1 (gate_up): grid (moe_inter/8, n*topk); one warp per (t, slot, f):
//   h_act[t, slot, f] = silu(gate_e[f] . x) * (up_e[f] . x).
// Phase 2 (down):    grid (hidden/8, n*topk); one warp per (t, slot, o):
//   out[t, o] += weight * (down_e[o] . h_act[t, slot]).

template <bool I8>
__global__ void moe_gate_up_q_kernel(const float* __restrict__ x, const int8_t* __restrict__ xq,
                                     const float* __restrict__ xs, const int* __restrict__ topk_ids,
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
    const ActView<I8> xt(x, xq, xs, (size_t)t, hidden);
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
        WFrag<I8> wf;
        wf.decode(gate_type, grow + (size_t)gi * g_gbytes, lane);
        g += wf.dot(xt, idx0, rem);
        wf.decode(up_type, urow + (size_t)gi * u_gbytes, lane);
        u += wf.dot(xt, idx0, rem);
    }
    g = warp_reduce_sum(g);
    u = warp_reduce_sum(u);
    if (lane == 0)
        h_act[(size_t)ts * moe_inter + f] = silu(g) * u;
}

// Down projections write UNWEIGHTED per-slot partial rows dpart[ts, hidden];
// a fixed-slot-order reduce applies the gate weights. (An atomicAdd
// accumulation was grid-shape-dependent, so a verify chunk's logits could
// drift off the decode steps it re-checks — breaking speculative parity.)
template <bool I8>
__global__ void moe_down_q_kernel(const int* __restrict__ topk_ids,
                                  const uint8_t* __restrict__ down_q,
                                  const float* __restrict__ h_act, const int8_t* __restrict__ hq,
                                  const float* __restrict__ hs, int topk, int hidden, int moe_inter,
                                  int64_t down_row_bytes, uint32_t down_type,
                                  float* __restrict__ dpart) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * kWarpsPerBlock + warp;
    if (o >= hidden) return;
    const int ts = blockIdx.y;
    const int e = topk_ids[ts];
    const uint8_t* drow = down_q + ((size_t)e * hidden + o) * down_row_bytes;
    const ActView<I8> ha(h_act, hq, hs, (size_t)ts, moe_inter);
    const int gbytes = qgroup_bytes(down_type);
    const int ngroups = (moe_inter + 255) >> 8;
    float acc = 0.0f;
    for (int gi = 0; gi < ngroups; ++gi) {
        const int idx0 = (gi << 8) + (lane << 3);
        const int rem = moe_inter - idx0;
        if (rem <= 0) continue;
        WFrag<I8> wf;
        wf.decode(down_type, drow + (size_t)gi * gbytes, lane);
        acc += wf.dot(ha, idx0, rem);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0)
        dpart[(size_t)ts * hidden + o] = acc;
}

__global__ void moe_down_reduce_kernel(const float* __restrict__ dpart,
                                       const float* __restrict__ topk_w,
                                       int n, int topk, int hidden,
                                       float* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * hidden) return;
    const int64_t t = i / hidden, o = i % hidden;
    float acc = 0.0f;
    for (int s = 0; s < topk; ++s)
        acc += topk_w[t * topk + s] * dpart[(size_t)(t * topk + s) * hidden + o];
    out[i] = acc;
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

template <int T, bool I8>
__global__ void moe_gate_up_q_emajor_kernel(const float* __restrict__ x,
                                            const int8_t* __restrict__ xq,
                                            const float* __restrict__ xs,
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
    for (int base = beg; base < end; base += T) {
        const int B = min(T, end - base);
        int ts[T];
        #pragma unroll
        for (int j = 0; j < T; ++j)
            if (j < B) ts[j] = ts_sorted[base + j];
        float g8[T], u8[T];
        #pragma unroll
        for (int j = 0; j < T; ++j) { g8[j] = 0.0f; u8[j] = 0.0f; }
        for (int gi = 0; gi < ngroups; ++gi) {
            const int idx0 = (gi << 8) + (lane << 3);
            const int rem = hidden - idx0;
            if (rem <= 0) continue;
            WFrag<I8> wf;
            wf.decode(gate_type, grow + (size_t)gi * g_gbytes, lane);
            #pragma unroll
            for (int j = 0; j < T; ++j)
                if (j < B) g8[j] += wf.dot(ActView<I8>(x, xq, xs, (size_t)(ts[j] / topk), hidden),
                                           idx0, rem);
            wf.decode(up_type, urow + (size_t)gi * u_gbytes, lane);
            #pragma unroll
            for (int j = 0; j < T; ++j)
                if (j < B) u8[j] += wf.dot(ActView<I8>(x, xq, xs, (size_t)(ts[j] / topk), hidden),
                                           idx0, rem);
        }
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            if (j >= B) continue;
            const float g = warp_reduce_sum(g8[j]);
            const float u = warp_reduce_sum(u8[j]);
            if (lane == 0) h_act[(size_t)ts[j] * moe_inter + f] = silu(g) * u;
        }
    }
}

template <int T, bool I8>
__global__ void moe_down_q_emajor_kernel(const int* __restrict__ ts_sorted,
                                         const int* __restrict__ offsets,
                                         const uint8_t* __restrict__ down_q,
                                         const float* __restrict__ h_act,
                                         const int8_t* __restrict__ hq,
                                         const float* __restrict__ hs,
                                         int hidden, int moe_inter,
                                         int64_t down_row_bytes, uint32_t down_type,
                                         float* __restrict__ dpart) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * kWarpsPerBlock + warp;
    if (o >= hidden) return;
    const int e = blockIdx.y;
    const int beg = offsets[e], end = offsets[e + 1];
    if (beg == end) return;
    const uint8_t* drow = down_q + ((size_t)e * hidden + o) * down_row_bytes;
    const int gbytes = qgroup_bytes(down_type);
    const int ngroups = (moe_inter + 255) >> 8;
    for (int base = beg; base < end; base += T) {
        const int B = min(T, end - base);
        int ts[T];
        #pragma unroll
        for (int j = 0; j < T; ++j)
            if (j < B) ts[j] = ts_sorted[base + j];
        float a8[T];
        #pragma unroll
        for (int j = 0; j < T; ++j) a8[j] = 0.0f;
        for (int gi = 0; gi < ngroups; ++gi) {
            const int idx0 = (gi << 8) + (lane << 3);
            const int rem = moe_inter - idx0;
            if (rem <= 0) continue;
            WFrag<I8> wf;
            wf.decode(down_type, drow + (size_t)gi * gbytes, lane);
            #pragma unroll
            for (int j = 0; j < T; ++j)
                if (j < B) a8[j] += wf.dot(ActView<I8>(h_act, hq, hs, (size_t)ts[j], moe_inter),
                                           idx0, rem);
        }
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            if (j >= B) continue;
            const float acc = warp_reduce_sum(a8[j]);
            if (lane == 0)
                dpart[(size_t)ts[j] * hidden + o] = acc;
        }
    }
}

// Shared-decode expert-major gate/up: one (expert, output row) per block, the
// 8 warps split each 64-token tile and share the decoded weight chunk (see
// gemm_q_smem_kernel). Bitwise-identical accumulation order to the T=8
// register-tile kernel above.
template <bool I8>
__global__ void moe_gate_up_q_emajor_smem_kernel(const float* __restrict__ x,
                                                 const int8_t* __restrict__ xq,
                                                 const float* __restrict__ xs,
                                                 const int* __restrict__ ts_sorted,
                                                 const int* __restrict__ offsets,
                                                 const uint8_t* __restrict__ gate_q,
                                                 const uint8_t* __restrict__ up_q,
                                                 int topk, int hidden, int moe_inter,
                                                 int64_t gate_row_bytes, int64_t up_row_bytes,
                                                 uint32_t gate_type, uint32_t up_type,
                                                 float* __restrict__ h_act) {
    constexpr int kChunk = kWarpsPerBlock;
    constexpr int kTok = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f = blockIdx.x;
    const int e = blockIdx.y;
    const int beg = offsets[e], end = offsets[e + 1];
    if (beg == end) return;  // block-uniform: safe before any barrier
    const uint8_t* grow = gate_q + ((size_t)e * moe_inter + f) * gate_row_bytes;
    const uint8_t* urow = up_q + ((size_t)e * moe_inter + f) * up_row_bytes;
    const int g_gbytes = qgroup_bytes(gate_type);
    const int u_gbytes = qgroup_bytes(up_type);
    const int ngroups = (hidden + 255) >> 8;
    __shared__ WFrag<I8> shg[kChunk][32];
    __shared__ WFrag<I8> shu[kChunk][32];
    for (int base = beg; base < end; base += kChunk * kTok) {
        const int tb = base + warp * kTok;
        const int B = min(kTok, end - tb);
        int ts[kTok];
        #pragma unroll
        for (int j = 0; j < kTok; ++j)
            if (j < B) ts[j] = ts_sorted[tb + j];
        float g8[kTok], u8[kTok];
        #pragma unroll
        for (int j = 0; j < kTok; ++j) { g8[j] = 0.0f; u8[j] = 0.0f; }
        for (int gc = 0; gc < ngroups; gc += kChunk) {
            const int gd = gc + warp;
            if (gd < ngroups && ((gd << 8) + (lane << 3)) < hidden) {
                shg[warp][lane].decode(gate_type, grow + (size_t)gd * g_gbytes, lane);
                shu[warp][lane].decode(up_type, urow + (size_t)gd * u_gbytes, lane);
            }
            __syncthreads();
            const int gn = min(kChunk, ngroups - gc);
            for (int gi = 0; gi < gn; ++gi) {
                const int idx0 = ((gc + gi) << 8) + (lane << 3);
                const int rem = hidden - idx0;
                if (rem <= 0) continue;
                #pragma unroll
                for (int j = 0; j < kTok; ++j) {
                    if (j >= B) continue;
                    const ActView<I8> a(x, xq, xs, (size_t)(ts[j] / topk), hidden);
                    g8[j] += shg[gi][lane].dot(a, idx0, rem);
                    u8[j] += shu[gi][lane].dot(a, idx0, rem);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < kTok; ++j) {
            if (j >= B) continue;
            const float g = warp_reduce_sum(g8[j]);
            const float u = warp_reduce_sum(u8[j]);
            if (lane == 0) h_act[(size_t)ts[j] * moe_inter + f] = silu(g) * u;
        }
    }
}

// Shared-decode expert-major down projection (same tiling as gate/up).
template <bool I8>
__global__ void moe_down_q_emajor_smem_kernel(const int* __restrict__ ts_sorted,
                                              const int* __restrict__ offsets,
                                              const uint8_t* __restrict__ down_q,
                                              const float* __restrict__ h_act,
                                              const int8_t* __restrict__ hq,
                                              const float* __restrict__ hs,
                                              int hidden, int moe_inter,
                                              int64_t down_row_bytes, uint32_t down_type,
                                              float* __restrict__ dpart) {
    constexpr int kChunk = kWarpsPerBlock;
    constexpr int kTok = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x;
    const int e = blockIdx.y;
    const int beg = offsets[e], end = offsets[e + 1];
    if (beg == end) return;  // block-uniform
    const uint8_t* drow = down_q + ((size_t)e * hidden + o) * down_row_bytes;
    const int gbytes = qgroup_bytes(down_type);
    const int ngroups = (moe_inter + 255) >> 8;
    __shared__ WFrag<I8> sh[kChunk][32];
    for (int base = beg; base < end; base += kChunk * kTok) {
        const int tb = base + warp * kTok;
        const int B = min(kTok, end - tb);
        int ts[kTok];
        #pragma unroll
        for (int j = 0; j < kTok; ++j)
            if (j < B) ts[j] = ts_sorted[tb + j];
        float a8[kTok];
        #pragma unroll
        for (int j = 0; j < kTok; ++j) a8[j] = 0.0f;
        for (int gc = 0; gc < ngroups; gc += kChunk) {
            const int gd = gc + warp;
            if (gd < ngroups && ((gd << 8) + (lane << 3)) < moe_inter)
                sh[warp][lane].decode(down_type, drow + (size_t)gd * gbytes, lane);
            __syncthreads();
            const int gn = min(kChunk, ngroups - gc);
            for (int gi = 0; gi < gn; ++gi) {
                const int idx0 = ((gc + gi) << 8) + (lane << 3);
                const int rem = moe_inter - idx0;
                if (rem <= 0) continue;
                #pragma unroll
                for (int j = 0; j < kTok; ++j)
                    if (j < B) a8[j] += sh[gi][lane].dot(
                        ActView<I8>(h_act, hq, hs, (size_t)ts[j], moe_inter), idx0, rem);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < kTok; ++j) {
            if (j >= B) continue;
            const float acc = warp_reduce_sum(a8[j]);
            if (lane == 0)
                dpart[(size_t)ts[j] * hidden + o] = acc;
        }
    }
}

// ---- int8 tensor-core (mma.m16n8k32) expert-major prefill path -------------
// The dp4a kernels are instruction-issue-bound at prefill (~6 MACs/instr incl.
// decode + addressing). One mma.sync.m16n8k32 does 4096 MACs/warp-instr; with
// the decoded weight chunk staged in shared and reused across a 64-token tile
// this cuts the instruction budget ~20x. Requirements: int8 activations
// (quantize_act_q8 layout: per-32 scales) and weight types whose scale is
// uniform per 32 elements with no affine min — Q8_0 / IQ3_XXS / IQ4_XS, which
// covers every prefill-hot tensor of the UD-Q3_K_XL GGUF. Falls back to the
// dp4a kernels per-type otherwise. GLMSERVE_MMA=0 disables.

static bool mma_type_ok(uint32_t type) {
    switch (type) {
        case QTYPE_Q8_0: case QTYPE_IQ3_XXS: case QTYPE_IQ4_XS: return true;
        default: return false;
    }
}

static bool mma_kernels_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("GLMSERVE_MMA");
        return !(v && *v && *v == '0');
    }();
    return on;
}

__device__ __forceinline__ void mma_s8_16x8x32(int* d, const int* a, const int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(0), "r"(0), "r"(0), "r"(0));
}

// dst[slot * dst_stride + f0 + row] = W[e][f0+row] . act(token(slot)) for the
// expert's dispatched slots. One block = (16-row tile, expert); inside, 64-slot
// token tiles: 8 warps each own one 8-token mma column subtile. Per 256-elem
// K-chunk the block decodes the 16 rows into shared once (int8 + per-32
// scales), then each warp runs 8 mma k-blocks scaled by ws x as.
// `div` maps slot -> activation row (topk for token-level acts, 1 for h_act).
__global__ void emajor_mma_s8_kernel(const int8_t* __restrict__ xq,
                                     const float* __restrict__ xs,
                                     const int* __restrict__ ts_sorted,
                                     const int* __restrict__ offsets, int div,
                                     const uint8_t* __restrict__ Wq,
                                     int64_t row_bytes, uint32_t type,
                                     int rows, int K,
                                     float* __restrict__ dst, int dst_stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f0 = blockIdx.x * 16;
    const int e = blockIdx.y;
    const int beg = offsets[e], end = offsets[e + 1];
    if (beg == end) return;  // block-uniform
    const int gbytes = qgroup_bytes(type);
    const int nchunks = K >> 8;              // K is a multiple of 256
    const int kb_scales = K >> 5;            // act scale row stride
    __shared__ int   shA[16][64];            // decoded chunk: 16 rows x 256 int8
    __shared__ float shS[16][8];             // per-32 weight scales
    __shared__ int   shTok[64];              // slot -> activation row
    __shared__ float shAS[64][8];            // per-32 act scales, tile x chunk
    const int r = lane >> 2, q = lane & 3;   // mma row group / thread-in-group

    for (int base = beg; base < end; base += 64) {
        const int nvalid = min(64, end - base);
        // Stage the tile's activation rows (clamped: lanes past the end
        // recompute the last slot and never write).
        if (threadIdx.x < 64)
            shTok[threadIdx.x] = ts_sorted[min(base + (int)threadIdx.x, end - 1)] / div;
        __syncthreads();
        // B loads use this lane's column (lane/4 of the warp subtile)...
        const int8_t* arow = xq + (size_t)shTok[warp * 8 + r] * K;
        // ...but the accumulated D fragment holds columns 2q and 2q+1, whose
        // activation scales differ — they are staged per chunk in shAS.
        float cf0 = 0.0f, cf1 = 0.0f, cf2 = 0.0f, cf3 = 0.0f;
        for (int ch = 0; ch < nchunks; ++ch) {
            __syncthreads();  // previous mma reads done before re-decode
            // 16 rows x 32 fragments; 512 frags over 256 threads = 2 each.
            #pragma unroll
            for (int rep = 0; rep < 2; ++rep) {
                const int fid = (int)threadIdx.x + rep * 256;
                const int row = fid >> 5, f = fid & 31;
                WFrag<true> wf;
                wf.decode(type, Wq + ((size_t)e * rows + f0 + row) * row_bytes +
                                    (size_t)ch * gbytes, f);
                shA[row][f * 2] = wf.w.w0;
                shA[row][f * 2 + 1] = wf.w.w1;
                if ((f & 3) == 0) shS[row][f >> 2] = wf.w.ds;
                // Act scales for the tile: 64 cols x 8 k-blocks = 512 floats.
                const int col = fid >> 3, kb = fid & 7;
                shAS[col][kb] = xs[(size_t)shTok[col] * kb_scales + (ch << 3) + kb];
            }
            __syncthreads();
            const int kbase = ch << 8;
            #pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                int a[4], b[2], d[4];
                a[0] = shA[r][kb * 8 + q];
                a[1] = shA[r + 8][kb * 8 + q];
                a[2] = shA[r][kb * 8 + q + 4];
                a[3] = shA[r + 8][kb * 8 + q + 4];
                memcpy(&b[0], arow + kbase + kb * 32 + q * 4, 4);
                memcpy(&b[1], arow + kbase + kb * 32 + q * 4 + 16, 4);
                mma_s8_16x8x32(d, a, b);
                const float ws0 = shS[r][kb], ws1 = shS[r + 8][kb];
                const float as0 = shAS[warp * 8 + q * 2][kb];
                const float as1 = shAS[warp * 8 + q * 2 + 1][kb];
                cf0 += (float)d[0] * ws0 * as0;
                cf1 += (float)d[1] * ws0 * as1;
                cf2 += (float)d[2] * ws1 * as0;
                cf3 += (float)d[3] * ws1 * as1;
            }
        }
        // Write back: c0/c1 -> row r, cols 2q/2q+1; c2/c3 -> row r+8.
        const int c0 = warp * 8 + q * 2, c1 = c0 + 1;
        if (c0 < nvalid) {
            const int slot = ts_sorted[base + c0];
            dst[(size_t)slot * dst_stride + f0 + r] = cf0;
            dst[(size_t)slot * dst_stride + f0 + r + 8] = cf2;
        }
        if (c1 < nvalid) {
            const int slot = ts_sorted[base + c1];
            dst[(size_t)slot * dst_stride + f0 + r] = cf1;
            dst[(size_t)slot * dst_stride + f0 + r + 8] = cf3;
        }
        __syncthreads();
    }
}

// h_act[i] = silu(g[i]) * u[i] over the dispatched slot rows.
__global__ void moe_silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                    float* __restrict__ h, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) h[i] = silu(g[i]) * u[i];
}

// Dense sibling of emajor_mma_s8_kernel: y[t, o] = x[t] . W[o] with one block
// per (16-row tile, 64-token tile). Same decode-to-shared + mma structure,
// token rows are just contiguous (no dispatch indirection).
__global__ void gemm_q_mma_s8_kernel(const int8_t* __restrict__ xq,
                                     const float* __restrict__ xs,
                                     const uint8_t* __restrict__ qW,
                                     const float* __restrict__ bias,
                                     float* __restrict__ y,
                                     int n, int K, int out, int64_t row_bytes,
                                     uint32_t type) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f0 = blockIdx.x * 16;
    const int tbase = blockIdx.y * 64;
    const int gbytes = qgroup_bytes(type);
    const int nchunks = K >> 8;
    const int kb_scales = K >> 5;
    __shared__ int   shA[16][64];
    __shared__ float shS[16][8];
    __shared__ float shAS[64][8];
    const int r = lane >> 2, q = lane & 3;
    const int nvalid = min(64, n - tbase);
    const int btok = min(tbase + warp * 8 + r, n - 1);  // B-load token (clamped)
    const int8_t* arow = xq + (size_t)btok * K;
    float cf0 = 0.0f, cf1 = 0.0f, cf2 = 0.0f, cf3 = 0.0f;
    for (int ch = 0; ch < nchunks; ++ch) {
        __syncthreads();
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int fid = (int)threadIdx.x + rep * 256;
            const int row = fid >> 5, f = fid & 31;
            WFrag<true> wf;
            wf.decode(type, qW + (size_t)(f0 + row) * row_bytes + (size_t)ch * gbytes, f);
            shA[row][f * 2] = wf.w.w0;
            shA[row][f * 2 + 1] = wf.w.w1;
            if ((f & 3) == 0) shS[row][f >> 2] = wf.w.ds;
            const int col = fid >> 3, kb = fid & 7;
            shAS[col][kb] = xs[(size_t)min(tbase + col, n - 1) * kb_scales + (ch << 3) + kb];
        }
        __syncthreads();
        const int kbase = ch << 8;
        #pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            int a[4], b[2], d[4];
            a[0] = shA[r][kb * 8 + q];
            a[1] = shA[r + 8][kb * 8 + q];
            a[2] = shA[r][kb * 8 + q + 4];
            a[3] = shA[r + 8][kb * 8 + q + 4];
            memcpy(&b[0], arow + kbase + kb * 32 + q * 4, 4);
            memcpy(&b[1], arow + kbase + kb * 32 + q * 4 + 16, 4);
            mma_s8_16x8x32(d, a, b);
            const float ws0 = shS[r][kb], ws1 = shS[r + 8][kb];
            const float as0 = shAS[warp * 8 + q * 2][kb];
            const float as1 = shAS[warp * 8 + q * 2 + 1][kb];
            cf0 += (float)d[0] * ws0 * as0;
            cf1 += (float)d[1] * ws0 * as1;
            cf2 += (float)d[2] * ws1 * as0;
            cf3 += (float)d[3] * ws1 * as1;
        }
    }
    const int c0 = warp * 8 + q * 2, c1 = c0 + 1;
    const float bs0 = bias ? bias[f0 + r] : 0.0f;
    const float bs8 = bias ? bias[f0 + r + 8] : 0.0f;
    if (c0 < nvalid) {
        y[(size_t)(tbase + c0) * out + f0 + r] = cf0 + bs0;
        y[(size_t)(tbase + c0) * out + f0 + r + 8] = cf2 + bs8;
    }
    if (c1 < nvalid) {
        y[(size_t)(tbase + c1) * out + f0 + r] = cf1 + bs0;
        y[(size_t)(tbase + c1) * out + f0 + r + 8] = cf3 + bs8;
    }
}

// ---- split-K single/few-token path -----------------------------------------
// At decode (n=1, topk=8) the token-major gate/up grid is (moe_inter_local/8,
// 8) blocks — ~192 blocks on a 142-SM part, each warp walking the whole
// hidden-dim group loop serially. Splitting the input dim kMoeSplitK ways
// multiplies the block count and shortens each warp's loop; partials land in
// gu_part [2, S, nts, moe_inter] (one slot per (split, ts, row) — no atomics)
// and a small epilogue reduces splits in fixed order (deterministic) and
// applies silu(g)*u.

template <bool I8>
__global__ void moe_gate_up_q_splitk_kernel(const float* __restrict__ x,
                                            const int8_t* __restrict__ xq,
                                            const float* __restrict__ xs,
                                            const int* __restrict__ topk_ids,
                                            const uint8_t* __restrict__ gate_q,
                                            const uint8_t* __restrict__ up_q,
                                            int nts, int topk, int hidden, int moe_inter,
                                            int64_t gate_row_bytes, int64_t up_row_bytes,
                                            uint32_t gate_type, uint32_t up_type,
                                            float* __restrict__ gu_part) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f = blockIdx.x * kWarpsPerBlock + warp;
    if (f >= moe_inter) return;
    const int ts = blockIdx.y;
    const int sp = blockIdx.z;
    const int e = topk_ids[ts];
    const ActView<I8> xt(x, xq, xs, (size_t)(ts / topk), hidden);
    const uint8_t* grow = gate_q + ((size_t)e * moe_inter + f) * gate_row_bytes;
    const uint8_t* urow = up_q + ((size_t)e * moe_inter + f) * up_row_bytes;
    const int g_gbytes = qgroup_bytes(gate_type);
    const int u_gbytes = qgroup_bytes(up_type);
    const int ngroups = (hidden + 255) >> 8;
    const int chunk = (ngroups + kMoeSplitK - 1) / kMoeSplitK;
    const int g0 = sp * chunk, g1 = min(ngroups, g0 + chunk);
    float g = 0.0f, u = 0.0f;
    for (int gi = g0; gi < g1; ++gi) {
        const int idx0 = (gi << 8) + (lane << 3);
        const int rem = hidden - idx0;
        if (rem <= 0) continue;
        WFrag<I8> wf;
        wf.decode(gate_type, grow + (size_t)gi * g_gbytes, lane);
        g += wf.dot(xt, idx0, rem);
        wf.decode(up_type, urow + (size_t)gi * u_gbytes, lane);
        u += wf.dot(xt, idx0, rem);
    }
    g = warp_reduce_sum(g);
    u = warp_reduce_sum(u);
    if (lane == 0) {
        const size_t slab = (size_t)kMoeSplitK * nts * moe_inter;
        const size_t slot = ((size_t)sp * nts + ts) * moe_inter + f;
        gu_part[slot] = g;
        gu_part[slab + slot] = u;
    }
}

__global__ void moe_gate_up_splitk_reduce_kernel(const float* __restrict__ gu_part,
                                                 int nts, int moe_inter,
                                                 float* __restrict__ h_act) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nts * moe_inter) return;
    const size_t slab = (size_t)kMoeSplitK * nts * moe_inter;
    float g = 0.0f, u = 0.0f;
    for (int sp = 0; sp < kMoeSplitK; ++sp) {
        g += gu_part[(size_t)sp * nts * moe_inter + i];
        u += gu_part[slab + (size_t)sp * nts * moe_inter + i];
    }
    h_act[i] = silu(g) * u;
}

template <bool I8>
static void moe_expert_ffn_q_launch(uint32_t gate_type, uint32_t up_type, uint32_t down_type,
                                    const float* x, const int8_t* xq, const float* xs,
                                    const int* topk_ids, const float* topk_w,
                                    const uint8_t* gate_q, const uint8_t* up_q,
                                    const uint8_t* down_q, int n, int topk, int hidden,
                                    int moe_inter, int E, int64_t gate_row_bytes,
                                    int64_t up_row_bytes, int64_t down_row_bytes,
                                    float* h_act, int8_t* hq, float* hs, float* dpart,
                                    float* out, int* dispatch, float* gu_part,
                                    cudaStream_t s) {
    seed_tables();
    const int threads = kWarpsPerBlock * 32;
    const int nts = n * topk;
    // The i8 down kernels read h_act re-quantized to int8 (per-32 blocks).
    auto quantize_h = [&] {
        if (I8) quantize_act_q8(h_act, nts, moe_inter, hq, hs, s);
    };
    // Fixed-slot-order weighted reduce of the down partials (deterministic —
    // an atomicAdd accumulation was grid-shape-dependent).
    auto reduce_down = [&] {
        const int64_t total = (int64_t)n * hidden;
        moe_down_reduce_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(
            dpart, topk_w, n, topk, hidden, out);
    };
    if (gu_part && nts < kMoeSplitKMaxTs) {
        // Split-K gate/up + reduce epilogue, then the token-major down (its
        // grid spans `hidden` rows — no occupancy problem at n=1).
        dim3 gs((unsigned)((moe_inter + kWarpsPerBlock - 1) / kWarpsPerBlock),
                (unsigned)nts, (unsigned)kMoeSplitK);
        moe_gate_up_q_splitk_kernel<I8><<<gs, threads, 0, s>>>(
            x, xq, xs, topk_ids, gate_q, up_q, nts, topk, hidden, moe_inter,
            gate_row_bytes, up_row_bytes, gate_type, up_type, gu_part);
        const int hcount = nts * moe_inter;
        moe_gate_up_splitk_reduce_kernel<<<(unsigned)((hcount + 255) / 256), 256, 0, s>>>(
            gu_part, nts, moe_inter, h_act);
        quantize_h();
        dim3 dn0((unsigned)((hidden + kWarpsPerBlock - 1) / kWarpsPerBlock),
                 (unsigned)nts);
        moe_down_q_kernel<I8><<<dn0, threads, 0, s>>>(
            topk_ids, down_q, h_act, hq, hs, topk, hidden, moe_inter,
            down_row_bytes, down_type, dpart);
        reduce_down();
        return;
    }
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
        if (I8 && nts >= kMoeSmemMinTs && mma_kernels_enabled() &&
            mma_type_ok(gate_type) && mma_type_ok(up_type) && mma_type_ok(down_type) &&
            (moe_inter & 255) == 0 && (hidden & 255) == 0 && 2 * moe_inter <= hidden) {
            // Tensor-core path: gate and up land in the (otherwise still
            // unused) dpart scratch — 2*moe_inter <= hidden so both fit.
            float* gbuf = dpart;
            float* ubuf = dpart + (size_t)nts * moe_inter;
            dim3 g1((unsigned)(moe_inter / 16), (unsigned)E);
            emajor_mma_s8_kernel<<<g1, 256, 0, s>>>(
                xq, xs, ts_sorted, offsets, topk, gate_q, gate_row_bytes, gate_type,
                moe_inter, hidden, gbuf, moe_inter);
            emajor_mma_s8_kernel<<<g1, 256, 0, s>>>(
                xq, xs, ts_sorted, offsets, topk, up_q, up_row_bytes, up_type,
                moe_inter, hidden, ubuf, moe_inter);
            const int64_t hcount = (int64_t)nts * moe_inter;
            moe_silu_mul_kernel<<<(unsigned)((hcount + 255) / 256), 256, 0, s>>>(
                gbuf, ubuf, h_act, hcount);
            quantize_h();
            dim3 g2((unsigned)(hidden / 16), (unsigned)E);
            emajor_mma_s8_kernel<<<g2, 256, 0, s>>>(
                hq, hs, ts_sorted, offsets, 1, down_q, down_row_bytes, down_type,
                hidden, moe_inter, dpart, hidden);
            reduce_down();
            return;
        }
        if (nts >= kMoeSmemMinTs && smem_kernels_enabled()) {
            // Shared-decode tiles: at prefill token counts each expert sees
            // ~nts/E tokens per pass, and the register-tile kernel re-decodes
            // every weight group per 8 tokens; the smem variants decode per
            // 64 (bitwise-identical accumulation).
            dim3 gu((unsigned)moe_inter, (unsigned)E);
            moe_gate_up_q_emajor_smem_kernel<I8><<<gu, threads, 0, s>>>(
                x, xq, xs, ts_sorted, offsets, gate_q, up_q, topk, hidden, moe_inter,
                gate_row_bytes, up_row_bytes, gate_type, up_type, h_act);
            quantize_h();
            dim3 dn((unsigned)hidden, (unsigned)E);
            moe_down_q_emajor_smem_kernel<I8><<<dn, threads, 0, s>>>(
                ts_sorted, offsets, down_q, h_act, hq, hs, hidden, moe_inter,
                down_row_bytes, down_type, dpart);
            reduce_down();
            return;
        }
        dim3 gu((unsigned)((moe_inter + kWarpsPerBlock - 1) / kWarpsPerBlock), (unsigned)E);
        moe_gate_up_q_emajor_kernel<8, I8><<<gu, threads, 0, s>>>(
            x, xq, xs, ts_sorted, offsets, gate_q, up_q, topk, hidden, moe_inter,
            gate_row_bytes, up_row_bytes, gate_type, up_type, h_act);
        quantize_h();
        dim3 dn((unsigned)((hidden + kWarpsPerBlock - 1) / kWarpsPerBlock), (unsigned)E);
        moe_down_q_emajor_kernel<8, I8><<<dn, threads, 0, s>>>(
            ts_sorted, offsets, down_q, h_act, hq, hs, hidden, moe_inter,
            down_row_bytes, down_type, dpart);
        reduce_down();
        return;
    }
    dim3 gu((unsigned)((moe_inter + kWarpsPerBlock - 1) / kWarpsPerBlock),
            (unsigned)(n * topk));
    moe_gate_up_q_kernel<I8><<<gu, threads, 0, s>>>(
        x, xq, xs, topk_ids, gate_q, up_q, topk, hidden, moe_inter,
        gate_row_bytes, up_row_bytes, gate_type, up_type, h_act);
    quantize_h();
    dim3 dn((unsigned)((hidden + kWarpsPerBlock - 1) / kWarpsPerBlock),
            (unsigned)(n * topk));
    moe_down_q_kernel<I8><<<dn, threads, 0, s>>>(
        topk_ids, down_q, h_act, hq, hs, topk, hidden, moe_inter,
        down_row_bytes, down_type, dpart);
    reduce_down();
}

void moe_expert_ffn_q(uint32_t gate_type, uint32_t up_type, uint32_t down_type,
                      const float* x, const int* topk_ids, const float* topk_w,
                      const uint8_t* gate_q, const uint8_t* up_q, const uint8_t* down_q,
                      int n, int topk, int hidden, int moe_inter, int E,
                      int64_t gate_row_bytes, int64_t up_row_bytes, int64_t down_row_bytes,
                      float* h_act, float* dpart, float* out, int* dispatch, float* gu_part,
                      cudaStream_t s) {
    moe_expert_ffn_q_launch<false>(gate_type, up_type, down_type, x, nullptr, nullptr,
                                   topk_ids, topk_w, gate_q, up_q, down_q, n, topk, hidden,
                                   moe_inter, E, gate_row_bytes, up_row_bytes, down_row_bytes,
                                   h_act, nullptr, nullptr, dpart, out, dispatch, gu_part, s);
}

// int8-activation MoE FFN: caller pre-quantizes x (shared with the other
// consumers of the same normed activations); h_act is re-quantized into
// (hq, hs) between the two phases. All three types must be qtype_has_i8 and
// hidden/moe_inter multiples of 32.
void moe_expert_ffn_q_i8(uint32_t gate_type, uint32_t up_type, uint32_t down_type,
                         const int8_t* xq, const float* xs,
                         const int* topk_ids, const float* topk_w,
                         const uint8_t* gate_q, const uint8_t* up_q, const uint8_t* down_q,
                         int n, int topk, int hidden, int moe_inter, int E,
                         int64_t gate_row_bytes, int64_t up_row_bytes, int64_t down_row_bytes,
                         float* h_act, int8_t* hq, float* hs, float* dpart, float* out,
                         int* dispatch, float* gu_part, cudaStream_t s) {
    moe_expert_ffn_q_launch<true>(gate_type, up_type, down_type, nullptr, xq, xs,
                                  topk_ids, topk_w, gate_q, up_q, down_q, n, topk, hidden,
                                  moe_inter, E, gate_row_bytes, up_row_bytes, down_row_bytes,
                                  h_act, hq, hs, dpart, out, dispatch, gu_part, s);
}

}  // namespace cuda
}  // namespace glmserve
