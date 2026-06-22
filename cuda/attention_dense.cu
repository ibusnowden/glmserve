// glmserve — dense causal attention over the paged KV cache (flash-style online
// softmax). One block per (query, head); one thread per head-dim lane.
//
// K/V pools for a single layer are laid out [num_phys_blocks, block_size,
// n_kv_heads, head_dim]; block_table maps a sequence's logical block -> physical.
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

__device__ __forceinline__ const float* kv_at(const float* base, const int* block_table,
                                               int64_t pos, int64_t block_size,
                                               int64_t n_kv_heads, int64_t head_dim,
                                               int64_t kvh) {
    int phys = block_table[pos / block_size];
    int64_t off = pos % block_size;
    return base + (((int64_t)phys * block_size + off) * n_kv_heads + kvh) * head_dim;
}

// blockDim.x == head_dim (one lane per dim). Shared buffer reduces the q.k dot.
__global__ void attn_dense_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                  const float* __restrict__ vc, const int* __restrict__ block_table,
                                  int64_t start_pos, int64_t n_heads, int64_t n_kv_heads,
                                  int64_t head_dim, int64_t block_size, float scale,
                                  float* __restrict__ out) {
    extern __shared__ float red[];           // size = head_dim
    int64_t qi = blockIdx.x;
    int64_t h  = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int64_t group = n_heads / n_kv_heads;
    int64_t kvh = h / group;
    int64_t qpos = start_pos + qi;

    float qd = q[(qi * n_heads + h) * head_dim + d];
    float m = -1e30f, l = 0.0f, acc = 0.0f;

    for (int64_t j = 0; j <= qpos; ++j) {
        const float* kj = kv_at(kc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        red[d] = qd * kj[d];
        __syncthreads();
        // tree reduction over head_dim lanes
        for (int stride = (int)head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();

        const float* vj = kv_at(vc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        float new_m = fmaxf(m, score);
        float corr = __expf(m - new_m);
        float p = __expf(score - new_m);
        l = l * corr + p;
        acc = acc * corr + p * vj[d];
        m = new_m;
    }
    out[(qi * n_heads + h) * head_dim + d] = (l > 0.0f) ? acc / l : 0.0f;
}

void attention_dense_paged(const float* q, const float* k_cache, const float* v_cache,
                           const int* block_table, int64_t n_query, int64_t start_pos,
                           int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                           int64_t block_size, float scale, float* out, cudaStream_t s) {
    dim3 grid((unsigned)n_query, (unsigned)n_heads);
    int threads = (int)head_dim;             // assumes head_dim is a power of two <= 1024
    size_t shmem = (size_t)head_dim * sizeof(float);
    attn_dense_kernel<<<grid, threads, shmem, s>>>(q, k_cache, v_cache, block_table,
                                                   start_pos, n_heads, n_kv_heads,
                                                   head_dim, block_size, scale, out);
}

// ---- flash-decoding (split-K) for the single-query decode step --------------
// The dense kernel above launches only n_heads blocks and walks every key
// serially — fine for prefill (n_query supplies parallelism) but it starves the
// GPU at decode (n_query==1) and grows linearly with context. Split the key
// range into S chunks processed by n_heads*S blocks in parallel (pass 1), then
// merge the S partial softmaxes per head (pass 2). Each block's serial key walk
// is only ctx/S long, so decode latency drops roughly S-fold at long context.

// Pass 1: each (head, split) block computes a partial online softmax over its
// key sub-range -> (acc[head_dim], m, l). Layout of partials: [n_heads, S, *].
__global__ void attn_decode_p1_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                      const float* __restrict__ vc, const int* __restrict__ block_table,
                                      int64_t qpos, int64_t n_heads, int64_t n_kv_heads,
                                      int64_t head_dim, int64_t block_size, float scale,
                                      int64_t keys_per_split, float* __restrict__ part_acc,
                                      float* __restrict__ part_m, float* __restrict__ part_l) {
    extern __shared__ float red[];           // size = head_dim
    int64_t h  = blockIdx.x;
    int64_t sp = blockIdx.y;
    int64_t S  = gridDim.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int64_t group = n_heads / n_kv_heads;
    int64_t kvh = h / group;
    int64_t j0 = sp * keys_per_split;
    int64_t j1 = j0 + keys_per_split;
    if (j1 > qpos + 1) j1 = qpos + 1;

    float qd = q[h * head_dim + d];
    float m = -1e30f, l = 0.0f, acc = 0.0f;
    for (int64_t j = j0; j < j1; ++j) {
        const float* kj = kv_at(kc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        red[d] = qd * kj[d];
        __syncthreads();
        for (int stride = (int)head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float score = red[0] * scale;
        __syncthreads();
        const float* vj = kv_at(vc, block_table, j, block_size, n_kv_heads, head_dim, kvh);
        float new_m = fmaxf(m, score);
        float corr = __expf(m - new_m);
        float p = __expf(score - new_m);
        l = l * corr + p;
        acc = acc * corr + p * vj[d];
        m = new_m;
    }
    int64_t idx = h * S + sp;
    part_acc[idx * head_dim + d] = acc;
    if (d == 0) { part_m[idx] = m; part_l[idx] = l; }
}

// Pass 2: merge the S partial softmaxes for each head into the final output.
__global__ void attn_decode_p2_kernel(const float* __restrict__ part_acc,
                                      const float* __restrict__ part_m, const float* __restrict__ part_l,
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

void attention_decode_paged(const float* q, const float* k_cache, const float* v_cache,
                            const int* block_table, int64_t qpos, int64_t n_heads,
                            int64_t n_kv_heads, int64_t head_dim, int64_t block_size,
                            float scale, float* out, float* part_acc, float* part_m,
                            float* part_l, int64_t max_splits, cudaStream_t s) {
    int64_t ctx = qpos + 1;
    // ~128 keys per split, but never more splits than the scratch holds.
    int64_t S = (ctx + 127) / 128;
    if (S < 1) S = 1;
    if (S > max_splits) S = max_splits;
    int64_t keys_per_split = (ctx + S - 1) / S;

    int threads = (int)head_dim;
    size_t shmem = (size_t)head_dim * sizeof(float);
    dim3 g1((unsigned)n_heads, (unsigned)S);
    attn_decode_p1_kernel<<<g1, threads, shmem, s>>>(q, k_cache, v_cache, block_table, qpos,
                                                     n_heads, n_kv_heads, head_dim, block_size,
                                                     scale, keys_per_split, part_acc, part_m, part_l);
    attn_decode_p2_kernel<<<(unsigned)n_heads, threads, 0, s>>>(part_acc, part_m, part_l,
                                                               n_heads, head_dim, S, out);
}

}  // namespace cuda
}  // namespace glmserve
