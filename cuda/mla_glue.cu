// glmserve — glue kernels for the GPU forward path: residual add, silu-gate,
// embedding gather, decoupled-RoPE on MLA sub-vectors, and per-head K/V assembly.
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

// y += x  (elementwise, length n)
__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ x, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
void add_inplace(float* y, const float* x, int64_t n, cudaStream_t s) {
    add_inplace_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(y, x, n);
}

// out = silu(g) * u  (length n)
__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                float* __restrict__ out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu(g[i]) * u[i];
}
void silu_mul(const float* g, const float* u, float* out, int64_t n, cudaStream_t s) {
    silu_mul_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(g, u, out, n);
}

// gather embedding rows: hidden[t] = table[tokens[t]]   (n rows of width H)
__global__ void embed_gather_kernel(const float* __restrict__ table, const int* __restrict__ tok,
                                    float* __restrict__ hidden, int64_t n, int64_t H) {
    int64_t t = blockIdx.x;
    const float* row = table + (int64_t)tok[t] * H;
    float* dst = hidden + t * H;
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) dst[i] = row[i];
}
void embed_gather(const float* table, const int* tokens, float* hidden, int64_t n, int64_t H,
                  cudaStream_t s) {
    embed_gather_kernel<<<(unsigned)n, 256, 0, s>>>(table, tokens, hidden, n, H);
}

// copy columns [offset, offset+len) of each row of src[n, src_stride] into dst[n, len]
__global__ void slice_rows_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                  int64_t n, int64_t src_stride, int64_t offset, int64_t len) {
    int64_t r = blockIdx.x;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
        dst[r * len + i] = src[r * src_stride + offset + i];
}
void slice_rows(const float* src, float* dst, int64_t n, int64_t src_stride, int64_t offset,
                int64_t len, cudaStream_t s) {
    slice_rows_kernel<<<(unsigned)n, 128, 0, s>>>(src, dst, n, src_stride, offset, len);
}

__device__ __forceinline__ void rope_pair(float* v, int64_t dim, int64_t pos, double theta,
                                           bool interleave, int64_t i) {
    int64_t half = dim / 2;
    double freq = 1.0 / pow(theta, (2.0 * i) / dim);
    double ang = pos * freq;
    float c = (float)cos(ang), s = (float)sin(ang);
    int64_t a = interleave ? 2 * i : i;
    int64_t b = interleave ? 2 * i + 1 : i + half;
    float x0 = v[a], x1 = v[b];
    v[a] = x0 * c - x1 * s;
    v[b] = x1 * c + x0 * s;
}

// RoPE the pe sub-vector of each head of q[n, H, qk] (offset `nope`, length `rope`).
__global__ void rope_q_kernel(float* __restrict__ q, int64_t n_heads, int64_t qk, int64_t nope,
                              int64_t rope, int64_t start_pos, double theta, bool interleave) {
    int64_t row = blockIdx.x;                  // token
    int64_t h   = blockIdx.y;                  // head
    int64_t pos = start_pos + row;
    float* v = q + (row * n_heads + h) * qk + nope;
    for (int64_t i = threadIdx.x; i < rope / 2; i += blockDim.x)
        rope_pair(v, rope, pos, theta, interleave, i);
}
void rope_q(float* q, int64_t n, int64_t n_heads, int64_t qk, int64_t nope, int64_t rope,
            int64_t start_pos, double theta, bool interleave, cudaStream_t s) {
    dim3 grid((unsigned)n, (unsigned)n_heads);
    rope_q_kernel<<<grid, 32, 0, s>>>(q, n_heads, qk, nope, rope, start_pos, theta, interleave);
}

// RoPE the shared k_pe[n, rope].
__global__ void rope_k_kernel(float* __restrict__ kpe, int64_t rope, int64_t start_pos,
                             double theta, bool interleave) {
    int64_t row = blockIdx.x;
    int64_t pos = start_pos + row;
    float* v = kpe + row * rope;
    for (int64_t i = threadIdx.x; i < rope / 2; i += blockDim.x)
        rope_pair(v, rope, pos, theta, interleave, i);
}
void rope_k(float* kpe, int64_t n, int64_t rope, int64_t start_pos, double theta,
            bool interleave, cudaStream_t s) {
    rope_k_kernel<<<(unsigned)n, 32, 0, s>>>(kpe, rope, start_pos, theta, interleave);
}

// Assemble per-head K=[k_nope|k_pe] (width hc) and V=v (width vd, in hc slot) from
// kvb[n, H, nope+vd] and the shared kpe[n, rope].
__global__ void assemble_kv_kernel(const float* __restrict__ kvb, const float* __restrict__ kpe,
                                   float* __restrict__ K, float* __restrict__ V, int64_t n_heads,
                                   int64_t nope, int64_t rope, int64_t vd, int64_t hc) {
    int64_t t = blockIdx.x, h = blockIdx.y;
    const float* src = kvb + (t * n_heads + h) * (nope + vd);
    const float* kp = kpe + t * rope;
    float* kdst = K + (t * n_heads + h) * hc;
    float* vdst = V + (t * n_heads + h) * hc;
    for (int64_t i = threadIdx.x; i < nope; i += blockDim.x) kdst[i] = src[i];
    for (int64_t i = threadIdx.x; i < rope; i += blockDim.x) kdst[nope + i] = kp[i];
    for (int64_t i = threadIdx.x; i < vd; i += blockDim.x) vdst[i] = src[nope + i];
}
void assemble_kv(const float* kvb, const float* kpe, float* K, float* V, int64_t n,
                 int64_t n_heads, int64_t nope, int64_t rope, int64_t vd, int64_t hc,
                 cudaStream_t s) {
    dim3 grid((unsigned)n, (unsigned)n_heads);
    assemble_kv_kernel<<<grid, 64, 0, s>>>(kvb, kpe, K, V, n_heads, nope, rope, vd, hc);
}

// Add the shared-expert output (already computed) scaled by 1.0 — convenience:
// out += shared  (same as add_inplace; provided for readability of the MoE path).

}  // namespace cuda
}  // namespace glmserve
