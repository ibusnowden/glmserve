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
#include "common.cuh"
#include "kernels.cuh"

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

__global__ void dsa_select_topk_kernel(const float* __restrict__ index_q,
                                       const float* __restrict__ index_k_cache,
                                       const float* __restrict__ index_w,
                                       int64_t start_pos, int64_t index_heads,
                                       int64_t index_dim, int64_t index_topk,
                                       float score_scale, float weight_scale,
                                       int* __restrict__ topk_indices,
                                       float* __restrict__ topk_scores) {
    int64_t t = blockIdx.x;
    if (threadIdx.x != 0) return;
    int64_t qpos = start_pos + t;
    int64_t topk = (qpos + 1 < index_topk) ? (qpos + 1) : index_topk;
    int* ids = topk_indices + t * index_topk;
    float* best = topk_scores + t * index_topk;
    for (int64_t k = 0; k < index_topk; ++k) {
        ids[k] = 0;
        best[k] = -1e30f;
    }
    int64_t filled = 0;
    const float* q = index_q + t * index_heads * index_dim;
    const float* wt = index_w + t * index_heads;
    for (int64_t j = 0; j <= qpos; ++j) {
        const float* kj = index_k_cache + j * index_dim;
        float score = 0.0f;
        for (int64_t h = 0; h < index_heads; ++h) {
            const float* qh = q + h * index_dim;
            float dot = 0.0f;
            for (int64_t d = 0; d < index_dim; ++d) dot += qh[d] * kj[d];
            float relu_score = fmaxf(0.0f, dot * score_scale);
            score += wt[h] * weight_scale * relu_score;
        }
        if (filled < topk) {
            ids[filled] = (int)j;
            best[filled] = score;
            ++filled;
            continue;
        }
        int64_t min_i = 0;
        float min_v = best[0];
        for (int64_t k = 1; k < topk; ++k) {
            if (best[k] < min_v) {
                min_v = best[k];
                min_i = k;
            }
        }
        if (score > min_v) {
            ids[min_i] = (int)j;
            best[min_i] = score;
        }
    }
    // Match the CPU reference's deterministic attention accumulation order.
    for (int64_t a = 1; a < topk; ++a) {
        int id = ids[a];
        float sc = best[a];
        int64_t b = a - 1;
        while (b >= 0 && ids[b] > id) {
            ids[b + 1] = ids[b];
            best[b + 1] = best[b];
            --b;
        }
        ids[b + 1] = id;
        best[b + 1] = sc;
    }
}

void dsa_select_topk(const float* index_q, const float* index_k_cache,
                     const float* index_w, int64_t n_query, int64_t start_pos,
                     int64_t index_heads, int64_t index_dim, int64_t index_topk,
                     float score_scale, float weight_scale, int* topk_indices,
                     float* topk_scores, cudaStream_t s) {
    dsa_select_topk_kernel<<<(unsigned)n_query, 32, 0, s>>>(
        index_q, index_k_cache, index_w, start_pos, index_heads, index_dim, index_topk,
        score_scale, weight_scale, topk_indices, topk_scores);
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
