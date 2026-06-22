// glmserve — GLM DSA (sparse) attention, V1 baseline.
//
// GLM-5.2 uses a "lightning indexer" that scores keys and attends only to the
// top-`index_topk` per query (spec §3, §9.4). The roadmap is explicit: don't
// start with the perfect DSA kernel — first make logits match, then specialize.
// So this V1:
//   * ctx <= index_topk  -> exact dense attention (indexer is a no-op)
//   * ctx >  index_topk  -> attend over the most-recent index_topk keys
//     (a well-defined sparse pattern; the true top-k-by-indexer-score selection
//      is the V2 upgrade, swapped in behind this same wrapper).
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

void attention_dsa_paged(const float* q, const float* k_cache, const float* v_cache,
                         const int* block_table, int64_t n_query, int64_t start_pos,
                         int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                         int64_t block_size, int64_t index_topk, float scale,
                         float* out, cudaStream_t s) {
    int64_t ctx = start_pos + n_query;
    if (ctx <= index_topk) {
        // exact dense path
        attention_dense_paged(q, k_cache, v_cache, block_table, n_query, start_pos,
                              n_heads, n_kv_heads, head_dim, block_size, scale, out, s);
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
