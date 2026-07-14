// glmserve — public CUDA kernel launch API (host-callable wrappers).
//
// All buffers are device pointers. The GPU forward path keeps activations in
// float32 (or half where noted); weights may be float32/half (fp16_gemm) or
// quantized (int4/fp8 gemm). These wrappers launch the kernels in cuda/*.cu and
// are declared here so the C++ control plane can call them without including
// CUDA headers everywhere. Only compiled when GLMSERVE_CUDA is set.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>   // half (used by gemm_fp16)

namespace glmserve {
namespace cuda {

// ---- normalization / position -------------------------------------------
// out[n,d] = rmsnorm(x[n,d]) * w[d]
void rmsnorm(const float* x, const float* w, float* out,
             int64_t n, int64_t d, float eps, cudaStream_t s = 0);

// Standard LayerNorm over rows: out = ((x - mean) / sqrt(var + eps)) * w + b.
void layernorm(const float* x, const float* w, const float* b, float* out,
               int64_t n, int64_t d, float eps, cudaStream_t s = 0);

// Per-head RMSNorm over head_dim (qk-norm).
void per_head_rmsnorm(float* x, const float* w, int64_t n, int64_t n_heads,
                      int64_t head_dim, float eps, cudaStream_t s = 0);

// Apply partial-rotary RoPE in place to x[n, n_heads, head_dim] at positions
// pos[n] (rotates first `rot` dims of each head).
void rope(float* x, const int64_t* pos, int64_t n, int64_t n_heads,
          int64_t head_dim, int64_t rot, double theta, cudaStream_t s = 0);

// ---- GEMM ----------------------------------------------------------------
// y[n,out] = x[n,in] @ W[out,in]^T (+ bias). cuBLASLt under the hood.
void gemm_fp32(const float* x, const float* W, const float* bias, float* y,
               int64_t n, int64_t in, int64_t out, cudaStream_t s = 0);
// Same GEMM with TF32 disabled (CUBLAS_PEDANTIC_MATH): true fp32 products, so
// the result matches the scalar fp32 kernel's domain (only summation order
// differs). Used by the DSA selector, whose top-k must agree with the scalar
// reference; TF32's 10-bit mantissa rotates scores enough to flip selections.
void gemm_fp32_pedantic(const float* x, const float* W, const float* bias, float* y,
                        int64_t n, int64_t in, int64_t out, cudaStream_t s = 0);
void gemm_fp16(const half* x, const half* W, const float* bias, half* y,
               int64_t n, int64_t in, int64_t out, cudaStream_t s = 0);

// W4A16: packed int4 weight (two nibbles/byte, symmetric, group scales) x fp16/fp32
// activations -> fp32 output. scales[out * (in/group_size)].
void gemm_w4a16(const float* x, const uint8_t* qW, const float* scales,
                const float* bias, float* y, int64_t n, int64_t in, int64_t out,
                int64_t group_size, cudaStream_t s = 0);

// FP8 (e4m3) weight x fp32 activations -> fp32 output (per-tensor or per-row scale).
void gemm_fp8(const float* x, const uint8_t* fp8W, const float* w_scale,
              const float* bias, float* y, int64_t n, int64_t in, int64_t out,
              cudaStream_t s = 0);

// GGUF quant GEMM: y[n,out] = x[n,in] @ W[out,in]^T (+bias) where W is a GGML
// block-quantized weight ([out,in] row-major, in contiguous, row_bytes stride).
// qtype is a GGML type id (8=Q8_0, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K,
// 18=IQ3_XXS, 23=IQ4_XS, 0=F32, 1=F16). Dequantizes on the fly inside the kernel
// (bandwidth-bound, like llama.cpp's MMVQ). One block per output column.
void gemm_q(uint32_t qtype, const float* x, const uint8_t* qW, const float* bias,
            float* y, int64_t n, int64_t in, int64_t out, int64_t row_bytes,
            cudaStream_t s = 0);

// ---- int8 MMQ (dp4a) activations ------------------------------------------
// The fp32 gemm_q/moe kernels are ALU-bound in dequant + fp32 FMAs. The MMQ
// twin quantizes activations to int8 per 32-element block and dots INTEGER
// weight fragments with two dp4a's per 8 elements (llama.cpp MMVQ-style).
// True for every GGUF type in the real checkpoint (Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/
// IQ3_XXS/IQ4_XS); F16/F32 fall back to the fp32 path.
bool qtype_has_i8(uint32_t qtype);

// xq[n, in] int8 + xs[n, in/32] scales; `in` must be a multiple of 32.
void quantize_act_q8(const float* x, int64_t n, int64_t in, int8_t* xq, float* xs,
                     cudaStream_t s = 0);

// gemm_q with pre-quantized activations.
void gemm_q_i8(uint32_t qtype, const int8_t* xq, const float* xs, const uint8_t* qW,
               const float* bias, float* y, int64_t n, int64_t in, int64_t out,
               int64_t row_bytes, cudaStream_t s = 0);

// Dequantize a single row (in elements) of a quant weight into fp32 dst (used
// for the embedding gather, where one row per token is needed).
void dequant_row_q(uint32_t qtype, const uint8_t* qW, float* dst, int64_t in,
                   int64_t row_bytes, cudaStream_t s = 0);

// Gather embeddings from a quant [vocab, H] table: for each token t, dequantize
// row tok[t] (H elements) into hidden[t*H]. qtable row o is at qtable + o*row_bytes.
void embed_gather_q(uint32_t qtype, const uint8_t* qtable, const int* tokens,
                    float* hidden, int64_t n, int64_t H, int64_t row_bytes,
                    cudaStream_t s = 0);

// MoE expert FFN reading merged per-expert GGUF quant tensors (only the active
// top-k experts are read). gate/up are [E, moe_inter, hidden] (in=hidden),
// down is [E, hidden, moe_inter] (in=moe_inter); row o of expert e lives at
// base + (e*out + o)*row_bytes. h_act is caller scratch [n, topk, moe_inter].
// out is zeroed and accumulated with topk weights. `dispatch` is optional int
// scratch [3E+1 + n*topk]: when given and n*topk is large enough, the FFN runs
// expert-major (tokens grouped by expert, 8-token weight-fragment reuse) —
// ~8x fewer quant decodes at prefill; pass nullptr for the token-major path.
// Split-K decode path constants: with a single token the gate/up grid is only
// (moe_inter_local/8 x topk) blocks — far below SM count — so the input-dim
// loop is split kMoeSplitK ways into `gu_part` partials [2, kMoeSplitK, nts,
// moe_inter] and reduced (in fixed split order — deterministic) by a small
// epilogue. Used when nts < kMoeSplitKMaxTs and `gu_part` is non-null.
constexpr int kMoeSplitK = 8;
constexpr int kMoeSplitKMaxTs = 64;

// `dpart` is caller scratch [n*topk, hidden]: the down phase writes unweighted
// per-slot partial rows there and a fixed-slot-order reduce applies the gate
// weights into `out` — deterministic across grid shapes (an atomicAdd
// accumulation broke speculative verify/decode parity).
void moe_expert_ffn_q(uint32_t gate_type, uint32_t up_type, uint32_t down_type,
                      const float* x, const int* topk_ids, const float* topk_w,
                      const uint8_t* gate_q, const uint8_t* up_q, const uint8_t* down_q,
                      int n, int topk, int hidden, int moe_inter, int E,
                      int64_t gate_row_bytes, int64_t up_row_bytes, int64_t down_row_bytes,
                      float* h_act, float* dpart, float* out, int* dispatch = nullptr,
                      float* gu_part = nullptr, cudaStream_t s = 0);

// int8-activation MoE FFN: x pre-quantized by the caller (shared with other
// consumers of the same normed activations); h_act is re-quantized into
// (hq [n*topk, moe_inter], hs) between the gate_up and down phases. Requires
// qtype_has_i8 on all three types and hidden/moe_inter multiples of 32.
void moe_expert_ffn_q_i8(uint32_t gate_type, uint32_t up_type, uint32_t down_type,
                         const int8_t* xq, const float* xs,
                         const int* topk_ids, const float* topk_w,
                         const uint8_t* gate_q, const uint8_t* up_q, const uint8_t* down_q,
                         int n, int topk, int hidden, int moe_inter, int E,
                         int64_t gate_row_bytes, int64_t up_row_bytes, int64_t down_row_bytes,
                         float* h_act, int8_t* hq, float* hs, float* dpart, float* out,
                         int* dispatch = nullptr, float* gu_part = nullptr,
                         cudaStream_t s = 0);

// ---- absorbed-MLA latent KV (cuda/mla_absorb.cu, fp16 dequant in qgemm.cu) --
// Latent cache row = [normed c_kv (kvlat) | roped k_pe (rope)] in fp16, shared
// by every head; decode attention runs absorbed (MQA over latents).
void latent_store(const float* ckv, const float* kpe, __half* latent, int64_t start_pos,
                  int64_t n, int64_t kvlat, int64_t rope, cudaStream_t s = 0);
// Absorb nq queries' q_nope through W_UK: qhat[nq, heads, kvlat+rope].
void mla_absorb_q(const float* q, const __half* kvb_f16, int64_t nq, int64_t n_heads,
                  int64_t qk, int64_t nope, int64_t vd, int64_t kvlat, int64_t rope,
                  float* qhat, cudaStream_t s = 0);
// Absorbed MQA over the latent cache for nq queries at positions
// [qpos0, qpos0+nq), each attending its own causal prefix. Key modes:
// indices != nullptr -> per-query DSA top-k rows indices[qi*index_topk ..]
// (count clamped to the prefix); win > 0 -> recent window; else dense [0,qpos].
// Pass 2 is fused with the W_UV expansion: out[nq, heads, vd] directly.
// Split geometry is per query and, for nq <= kMlaParityMaxQ, identical to the
// nq == 1 launch — a verify chunk merges bit-identically to the decode steps
// it re-checks (speculative parity). part_acc must hold
// kMlaParityMaxQ * max_splits * n_heads * kvlat floats.
constexpr int64_t kMlaParityMaxQ = 64;
void mla_attention_decode(const float* qhat, const __half* latent, const int* indices,
                          int64_t index_topk, int64_t win, int64_t qpos0, int64_t nq,
                          int64_t n_heads, int64_t kvlat, int64_t rope, float scale,
                          const __half* kvb_f16, int64_t nope, int64_t vd, float* out,
                          float* part_acc, float* part_m, float* part_l,
                          int64_t max_splits, cudaStream_t s = 0);
void convert_f32_f16(const float* src, int64_t n, __half* dst, cudaStream_t s = 0);
// Batch-dequantize a device-resident GGUF quant weight [rows, in] to fp16.
void dequant_rows_f16(uint32_t qtype, const uint8_t* qW, int64_t rows, int64_t in,
                      int64_t row_bytes, __half* out, cudaStream_t s = 0);

// ---- attention -----------------------------------------------------------
// Dense causal attention reading K/V from a paged cache. q[n,H,hd]; the cache
// stores K/V for absolute positions [0, ctx). block_table maps logical->phys.
void attention_dense_paged(const float* q, const float* k_cache, const float* v_cache,
                           const int* block_table, int64_t n_query, int64_t start_pos,
                           int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                           int64_t block_size, float scale, float* out,
                           cudaStream_t s = 0);

// Flash-decoding (split-K) for the single-query decode step. Parallelizes the
// key range across `n_heads * S` blocks (S chosen from ctx) and merges the
// partial softmaxes — far more parallelism than the dense kernel at n_query==1.
// part_acc[n_heads, max_splits, head_dim], part_m/part_l[n_heads, max_splits]
// are caller-provided device scratch. qpos is the query's absolute position.
void attention_decode_paged(const float* q, const float* k_cache, const float* v_cache,
                            const int* block_table, int64_t qpos, int64_t n_heads,
                            int64_t n_kv_heads, int64_t head_dim, int64_t block_size,
                            float scale, float* out, float* part_acc, float* part_m,
                            float* part_l, int64_t max_splits, cudaStream_t s = 0);

// DSA sparse attention: a lightning indexer scores keys, top-k are attended.
// Degrades to dense when ctx <= index_topk.
void attention_dsa_paged(const float* q, const float* k_cache, const float* v_cache,
                         const int* block_table, int64_t n_query, int64_t start_pos,
                         int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                         int64_t block_size, int64_t index_topk, float scale,
                         float* out, cudaStream_t s = 0,
                         float* part_acc = nullptr, float* part_m = nullptr,
                         float* part_l = nullptr, int64_t max_splits = 0);

// Learned GLM DSA selector (parallel scores + exact radix top-k) and indexed
// sparse attention. index_q[n,index_heads,index_dim], index_k_cache[ctx,
// index_dim] fp16, index_w[n,index_heads]. topk_indices[n,index_topk], sorted
// ascending; ties resolve to the smaller index (CPU reference parity).
// Queries are processed in chunks of kDsaScoreChunk; score_scratch must hold
// kDsaScoreChunk * (start_pos + n_query) floats.
constexpr int64_t kDsaScoreChunk = 64;
// Key-tile width of the SGEMM scoring path: per-head dot GEMM into gemm_dbuf
// [kDsaScoreChunk * index_heads, kDsaKeyTile] over an fp32-converted key tile
// gemm_kf32 [kDsaKeyTile, index_dim], then a ReLU-weighted head-sum epilogue.
// Null gemm buffers (or GLMSERVE_DSA_GEMM=0) fall back to the scalar kernel.
constexpr int64_t kDsaKeyTile = 4096;
// cub::DeviceRadixSort scratch for the decode top-k path (nc == 1): multi-block
// GPU-wide sort replaces the single-SM radix select. Allocated once at
// upload_to_gpu and sized for max_ctx. Null (or GLMSERVE_DSA_CUB=0) falls back
// to the single-block radix select.
struct DsaCubScratch {
    void* temp = nullptr;
    size_t temp_bytes = 0;
    uint64_t* keys_in = nullptr;
    int* idx_in = nullptr;
    uint64_t* keys_out = nullptr;
    int* idx_out = nullptr;
};
// Query the cub temp-storage bytes needed for sorting `max_ctx` uint64 keys.
// Call once at allocation; the result is stable for any count <= max_ctx.
// Defined in attention_dsa.cu (uses cub::DeviceRadixSort, CUDA-only).
size_t dsa_cub_temp_bytes(int64_t max_ctx);
void dsa_select_topk(const float* index_q, const __half* index_k_cache,
                     const float* index_w, int64_t n_query, int64_t start_pos,
                     int64_t index_heads, int64_t index_dim, int64_t index_topk,
                     float score_scale, float weight_scale, float* score_scratch,
                     int* topk_indices, float* topk_scores,
                     float* gemm_dbuf = nullptr, float* gemm_kf32 = nullptr,
                     DsaCubScratch* cub_scratch = nullptr, cudaStream_t s = 0);
void attention_dsa_indexed_paged(const float* q, const float* k_cache, const float* v_cache,
                                 const int* block_table, const int* topk_indices,
                                 int64_t n_query, int64_t start_pos, int64_t n_heads,
                                 int64_t n_kv_heads, int64_t head_dim, int64_t block_size,
                                 int64_t index_topk, float scale, float* out,
                                 cudaStream_t s = 0);

// ---- MoE -----------------------------------------------------------------
// Router: logits[n, E] -> per-token top-k expert ids and gate weights.
// sigmoid scoring + optional score-correction bias + norm + routed scaling.
void moe_router(const float* logits, const float* e_bias, int n, int E, int topk,
                bool norm_topk, float routed_scale, int* topk_ids,
                float* topk_weights, cudaStream_t s = 0);

// Build a permutation that groups tokens by expert (for grouped GEMM):
// fills expert_counts[E], and sorted_token/sorted_slot arrays of length n*topk.
void moe_dispatch(const int* topk_ids, int n, int topk, int E, int* expert_offsets,
                  int* sorted_token_ids, int* sorted_row, cudaStream_t s = 0);

// Expert FFN over dispatched rows: out += weight * down(silu(gate(x))*up(x)),
// accumulated into per-token output. Reference grouped implementation.
void moe_expert_ffn(const float* x, const int* topk_ids, const float* topk_weights,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int n, int topk, int hidden, int moe_inter, int E,
                    float* out, cudaStream_t s = 0);

// Same routed expert FFN with packed W4A16 resident weights. gate/up layout:
// qweight[E, moe_inter, ceil(hidden/2)], scales[E, moe_inter, ceil(hidden/group)].
// down layout: qweight[E, hidden, ceil(moe_inter/2)],
// scales[E, hidden, ceil(moe_inter/group)].
void moe_expert_ffn_w4a16(const float* x, const int* topk_ids, const float* topk_weights,
                          const uint8_t* gate_q, const float* gate_sc,
                          const uint8_t* up_q, const float* up_sc,
                          const uint8_t* down_q, const float* down_sc,
                          int n, int topk, int hidden, int moe_inter, int E,
                          int group_size, float* out, cudaStream_t s = 0);

// ---- sampling ------------------------------------------------------------
void argmax(const float* logits, int vocab, int* out_id, cudaStream_t s = 0);
// Per-row first-max argmax over y[nrows, ld] (ncols valid); out float2{val, idx}.
void argmax_rows(const float* y, int nrows, int ncols, int64_t ld, float2* out_pairs,
                 cudaStream_t s = 0);
// Per-row logsumexp over y[nrows, ld] (ncols valid); out[row] = max + log(sumexp).
void row_logsumexp(const float* y, int nrows, int ncols, int64_t ld, float* out_lse,
                   cudaStream_t s = 0);
void softmax_inplace(float* logits, int vocab, float temperature, cudaStream_t s = 0);

// ---- glue (GPU forward path) ---------------------------------------------
void add_inplace(float* y, const float* x, int64_t n, cudaStream_t s = 0);
void silu_mul(const float* g, const float* u, float* out, int64_t n, cudaStream_t s = 0);
void embed_gather(const float* table, const int* tokens, float* hidden, int64_t n,
                  int64_t H, cudaStream_t s = 0);
void slice_rows(const float* src, float* dst, int64_t n, int64_t src_stride, int64_t offset,
                int64_t len, cudaStream_t s = 0);
void rope_q(float* q, int64_t n, int64_t n_heads, int64_t qk, int64_t nope, int64_t rope,
            int64_t start_pos, double theta, bool interleave, cudaStream_t s = 0);
void rope_k(float* kpe, int64_t n, int64_t rope, int64_t start_pos, double theta,
            bool interleave, cudaStream_t s = 0);
void rope_index_q(float* q, int64_t n, int64_t n_heads, int64_t head_dim, int64_t rope,
                  int64_t start_pos, double theta, bool interleave, cudaStream_t s = 0);
void assemble_kv(const float* kvb, const float* kpe, float* K, float* V, int64_t n,
                 int64_t n_heads, int64_t nope, int64_t rope, int64_t vd, int64_t hc,
                 cudaStream_t s = 0);

}  // namespace cuda
}  // namespace glmserve
