// glmserve - GPU forward path (device-resident weights + CUDA kernels).
//
// Mirrors the CPU reference in src/model_glm52.cpp exactly, executing the full
// GLM-5.2 MLA block on the device: embed gather -> RMSNorm -> MLA Q/KV latent
// projections (cuBLAS) -> decoupled RoPE -> per-head K/V assembly -> dense paged
// attention -> o_proj -> dense MLP or MoE (router + grouped expert kernels +
// shared expert) -> residuals -> final norm -> lm_head.
//
// Both prefill and decode share one block-stack runner and a single persistent
// device state:
//   * forward_gpu_prefill() resets the cache, fills positions [0, n) in one shot.
//   * forward_gpu_decode()  appends one token's K/V and attends over [0, pos],
//     reusing persistent scratch (no per-token cudaMalloc) and the persistent
//     per-layer KV cache (O(ctx) per token instead of re-prefilling, O(ctx^2)).
// Only compiled in a GLMSERVE_CUDA build.
#include "model.hpp"
#include "common.hpp"
#include "gguf_quant.hpp"
#include "model_gguf.hpp"

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>
#include "kernels.cuh"

#include <algorithm>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace glmserve {

namespace {
// Cumulative H2D bytes this process has uploaded (progress logging).
uint64_t g_upload_bytes = 0;
// Per-component upload timing (µs), reset per layer by the progress logger:
// where does a degrading upload spend its time — allocation, DMA, or repack?
uint64_t g_malloc_us = 0, g_memcpy_us = 0, g_stage_us = 0;
struct UsTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    uint64_t us() const {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count());
    }
};
inline cudaError_t timed_malloc(void** p, size_t n) {
    UsTimer t; cudaError_t e = cudaMalloc(p, n); g_malloc_us += t.us(); return e;
}
inline cudaError_t timed_memcpy_h2d(void* dst, const void* src, size_t n) {
    UsTimer t; cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
    g_memcpy_us += t.us(); return e;
}
inline float* dmalloc(size_t n_floats) {
    float* p = nullptr;
    if (timed_malloc(reinterpret_cast<void**>(&p), n_floats * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    return p;
}
inline float* upload(const std::vector<float>& v) {
    float* p = dmalloc(v.size());
    timed_memcpy_h2d(p, v.data(), v.size() * sizeof(float));
    g_upload_bytes += v.size() * sizeof(float);
    return p;
}
inline uint8_t* upload_u8(const std::vector<uint8_t>& v) {
    uint8_t* p = nullptr;
    if (timed_malloc(reinterpret_cast<void**>(&p), v.size()) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    timed_memcpy_h2d(p, v.data(), v.size());
    g_upload_bytes += v.size();
    return p;
}
// Upload a raw host byte range (mmap'd GGUF quant payload) to the device.
inline uint8_t* upload_bytes(const void* host, size_t nbytes) {
    uint8_t* p = nullptr;
    if (timed_malloc(reinterpret_cast<void**>(&p), nbytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    timed_memcpy_h2d(p, host, nbytes);
    g_upload_bytes += nbytes;
    return p;
}
// Reusable host staging buffer for strided-row repacks. cudaMemcpy2D from
// pageable (mmap'd) memory stages row-by-row (~136 B DMAs — hours for the
// expert set); repacking on the CPU and issuing ONE contiguous H2D is ~1000x
// faster. Upload is single-threaded per rank, so a static buffer is fine.
inline uint8_t* stage_rows(const void* host, size_t rows, size_t row_bytes,
                           size_t src_stride) {
    UsTimer t;
    static std::vector<uint8_t> staging;
    if (staging.size() < rows * row_bytes) staging.resize(rows * row_bytes);
    const uint8_t* src = static_cast<const uint8_t*>(host);
    for (size_t r = 0; r < rows; ++r)
        std::memcpy(staging.data() + r * row_bytes, src + r * src_stride, row_bytes);
    g_stage_us += t.us();
    return staging.data();
}
// Upload a strided host row view (row-parallel TP shard of a mmap'd GGUF quant
// payload: `rows` rows of `row_bytes` useful bytes every `src_stride` bytes)
// into a contiguous device buffer.
inline uint8_t* upload_bytes_2d(const void* host, size_t rows, size_t row_bytes,
                                size_t src_stride) {
    if (src_stride == row_bytes) return upload_bytes(host, rows * row_bytes);
    return upload_bytes(stage_rows(host, rows, row_bytes, src_stride),
                        rows * row_bytes);
}
// Copy a (possibly strided) host quant payload into a contiguous device buffer
// at byte offset `dst_off` (used to pack per-expert tensors [E, out, in]).
inline void upload_bytes_2d_into(uint8_t* dst, size_t dst_off, const void* host,
                                 size_t rows, size_t row_bytes, size_t src_stride) {
    const void* src = host;
    if (src_stride != row_bytes) src = stage_rows(host, rows, row_bytes, src_stride);
    if (timed_memcpy_h2d(dst + dst_off, src, rows * row_bytes) != cudaSuccess)
        throw std::runtime_error("expert upload failed");
    g_upload_bytes += rows * row_bytes;
}
// --- Upload-source prefetch -------------------------------------------------
// The upload's host-side reads (stage_rows + pageable cudaMemcpy) source from
// the mmap'd GGUF payload. On a node near its page-cache ceiling the kernel
// evicts not-yet-uploaded ranges while earlier layers upload; every refetch
// then evicts another upcoming page and the upload degrades to random-fault
// NFS reads (observed 2.7 GB/s -> <10 MB/s). A prefetch thread touching a few
// layers ahead keeps source pages seconds-old at copy time: sequential touches
// get kernel readahead, and the eviction victims become already-uploaded
// ranges, which cost nothing.
struct PfRange { const uint8_t* p; size_t n; };
inline void collect_range(const Linear& L, std::vector<PfRange>& out) {
    // Only mmap-backed quant views (qbuf-resident tensors are already in RAM).
    if (!L.has_q() || !L.qbuf.empty() || L.out_features <= 0) return;
    const size_t stride = static_cast<size_t>(L.qsrc_stride());
    const size_t rows = static_cast<size_t>(L.out_features);
    out.push_back({L.qdata, (rows - 1) * stride + static_cast<size_t>(L.row_bytes)});
}
inline std::vector<PfRange> layer_prefetch_ranges(const Layer& L) {
    std::vector<PfRange> rs;
    collect_range(L.q_a_proj, rs); collect_range(L.q_b_proj, rs);
    collect_range(L.kv_a_proj, rs); collect_range(L.kv_b_proj, rs);
    collect_range(L.o_proj, rs);
    if (L.indexer.valid()) {
        collect_range(L.indexer.wq_b, rs); collect_range(L.indexer.wk, rs);
        collect_range(L.indexer.weights_proj, rs);
    }
    if (L.is_dense) {
        collect_range(L.dense_mlp.gate_proj, rs);
        collect_range(L.dense_mlp.up_proj, rs);
        collect_range(L.dense_mlp.down_proj, rs);
    } else {
        collect_range(L.moe.router, rs);
        for (const Expert& e : L.moe.experts) {
            collect_range(e.gate_proj, rs);
            collect_range(e.up_proj, rs);
            collect_range(e.down_proj, rs);
        }
        if (L.moe.has_shared) {
            collect_range(L.moe.shared.gate_proj, rs);
            collect_range(L.moe.shared.up_proj, rs);
            collect_range(L.moe.shared.down_proj, rs);
        }
    }
    return rs;
}
inline void touch_ranges(const std::vector<PfRange>& rs) {
    // Kick off async readahead for every range first, then touch to pace.
    for (const PfRange& r : rs) {
        if (!r.p || !r.n) continue;
        uintptr_t a = reinterpret_cast<uintptr_t>(r.p) & ~static_cast<uintptr_t>(4095);
        size_t n = r.n + (reinterpret_cast<uintptr_t>(r.p) - a);
        madvise(reinterpret_cast<void*>(a), n, MADV_WILLNEED);
    }
    volatile uint64_t sink = 0;
    for (const PfRange& r : rs) {
        if (!r.p || !r.n) continue;
        const uint8_t* a = r.p;
        const uint8_t* e = r.p + r.n;
        for (; a < e; a += 4096) sink += *a;
        sink += e[-1];
    }
    (void)sink;
}
// --- Upload-source eviction ---------------------------------------------
// Prefetch alone loses on a 515 GB node loading a 343 GB model: once the
// page cache hits the ceiling, kernel reclaim evicts not-yet-uploaded mmap
// pages and the upload collapses to random-fault NFS reads (jobs 143730..
// 143840 all died there). Deterministic fix: after a layer's shard views are
// on the device, drop them from our page tables AND the shared page cache
// with a lag of GLMSERVE_EVICT_LAG layers (default 8, covers rank skew plus
// the prefetch window). The resident window is then ~(lag + prefetch) layers
// (~50 GiB), so reclaim never runs and the source stream stays sequential.
inline void evict_ranges(const GGUFModel* gg, std::vector<PfRange> rs) {
    if (!gg || rs.empty()) return;
    std::sort(rs.begin(), rs.end(),
              [](const PfRange& a, const PfRange& b) { return a.p < b.p; });
    // Coalesce ranges separated by small gaps (tensor alignment padding and
    // same-layer TP stride interleave) into single evict calls.
    const size_t MAX_GAP = 1 << 20;
    const uint8_t* cur = rs[0].p;
    const uint8_t* end = rs[0].p + rs[0].n;
    for (size_t i = 1; i < rs.size(); ++i) {
        if (!rs[i].p || !rs[i].n) continue;
        if (rs[i].p <= end + MAX_GAP) {
            if (rs[i].p + rs[i].n > end) end = rs[i].p + rs[i].n;
        } else {
            gg->evict_range(cur, static_cast<size_t>(end - cur));
            cur = rs[i].p;
            end = rs[i].p + rs[i].n;
        }
    }
    gg->evict_range(cur, static_cast<size_t>(end - cur));
}
// Surface any pending CUDA error (async kernel launch failures included) so a
// bad GPU forward fails loudly instead of returning garbage logits to the gate.
inline void cuda_check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error after ") + what + ": " +
                                 cudaGetErrorString(e));
}
struct DLinear {
    float* w = nullptr;
    uint8_t* qweight = nullptr;
    float* scales = nullptr;
    int64_t out = 0, in = 0, group_size = 0;
    bool quantized_int4 = false;
    // GGUF block-quantized weight (Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS/F16).
    uint32_t qtype = 0;
    uint8_t* qdata = nullptr;
    int64_t row_bytes = 0;
    bool has_q() const { return qtype != 0 && qdata != nullptr; }
};
inline DLinear up_lin(const Linear& L) {
    DLinear d; d.out = L.out_features; d.in = L.in_features;
    d.group_size = L.group_size;
    d.quantized_int4 = L.quantized_int4;
    if (L.has_q()) {
        d.qtype = L.qtype;
        d.row_bytes = L.row_bytes;
        const uint8_t* host = L.qbuf.empty() ? L.qdata : L.qbuf.data();
        d.qdata = upload_bytes_2d(host, static_cast<size_t>(L.out_features),
                                  static_cast<size_t>(L.row_bytes),
                                  static_cast<size_t>(L.qsrc_stride()));
    } else {
        if (!L.w.empty()) d.w = upload(L.w);
        if (!L.qweight.empty()) d.qweight = upload_u8(L.qweight);
        if (!L.scales.empty()) d.scales = upload(L.scales);
    }
    return d;
}
inline void free_lin(DLinear& d) {
    if (d.w) cudaFree(d.w);
    if (d.qweight) cudaFree(d.qweight);
    if (d.scales) cudaFree(d.scales);
    if (d.qdata) cudaFree(d.qdata);
    d = DLinear{};
}
inline void gemm_linear(const float* x, const DLinear& d, const float* bias,
                        float* y, int64_t n, cudaStream_t s = 0) {
    if (d.has_q()) {
        glmserve::cuda::gemm_q(d.qtype, x, d.qdata, bias, y,
                               n, d.in, d.out, d.row_bytes, s);
    } else if (d.quantized_int4) {
        GLM_CHECK(d.qweight && d.scales && d.group_size > 0,
                  "quantized linear missing resident qweight/scales");
        glmserve::cuda::gemm_w4a16(x, d.qweight, d.scales, bias, y,
                                   n, d.in, d.out, d.group_size, s);
    } else {
        GLM_CHECK(d.w, "fp32 linear missing resident weight");
        glmserve::cuda::gemm_fp32(x, d.w, bias, y, n, d.in, d.out, s);
    }
}
}  // namespace

struct DLayer {
    float *input_norm = nullptr, *post_norm = nullptr;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    DLinear q_a, q_b, kv_a, kv_b, o;
    DLinear index_wq_b, index_wk, index_weights;
    float* index_k_norm = nullptr;
    float* index_k_bias = nullptr;
    bool is_dense = false;
    DLinear gate, up, down;                 // dense MLP
    DLinear router;                         // MoE
    float* e_bias = nullptr;
    int64_t expert_inter = 0;
    float *exp_gate = nullptr, *exp_up = nullptr, *exp_down = nullptr;  // packed [E,...]
    uint8_t *qexp_gate = nullptr, *qexp_up = nullptr, *qexp_down = nullptr;
    float *sexp_gate = nullptr, *sexp_up = nullptr, *sexp_down = nullptr;
    int64_t exp_group_size = 0;
    bool quantized_experts = false;
    // GGUF block-quantized experts (merged [E, out_l, in] on device).
    uint32_t qexp_gate_type = 0, qexp_up_type = 0, qexp_down_type = 0;
    uint8_t *qexp_gate_q = nullptr, *qexp_up_q = nullptr, *qexp_down_q = nullptr;
    int64_t qexp_gate_rb = 0, qexp_up_rb = 0, qexp_down_rb = 0;
    int64_t qexp_out_l = 0, qexp_in_l = 0;  // sharded expert out/in (gate/up out, down in)
    bool quant_experts_gguf = false;
    DLinear sh_gate, sh_up, sh_down;        // shared expert
};

// Per-forward activation buffers, allocated once and grown on demand. Decode
// (n=1) reuses the same arena every token, so there is no per-token cudaMalloc.
struct Scratch {
    int64_t cap = 0;     // max n this arena is sized for
    int*   d_tokens = nullptr;
    float *hidden_d = nullptr, *normed = nullptr, *qa = nullptr, *q = nullptr;
    float *kva = nullptr, *ckv = nullptr, *kpe = nullptr, *kvb = nullptr;
    float *attn = nullptr, *attn_proj = nullptr, *mlp = nullptr;
    float *gbuf = nullptr, *ubuf = nullptr, *gu = nullptr, *rlogits = nullptr, *shbuf = nullptr;
    float *index_q = nullptr, *index_k = nullptr, *index_w = nullptr, *dsa_scores = nullptr;
    int*   topk_ids = nullptr;
    int*   dsa_indices = nullptr;
    float* topk_w = nullptr;
    float* moe_h_act = nullptr;   // [n, topk, moe_inter] scratch for quant MoE

    void free_all() {
        for (float* p : {hidden_d, normed, qa, q, kva, ckv, kpe, kvb, attn,
                         attn_proj, mlp, gbuf, ubuf, gu, rlogits, shbuf,
                         index_q, index_k, index_w, dsa_scores, topk_w, moe_h_act})
            if (p) cudaFree(p);
        if (d_tokens) cudaFree(d_tokens);
        if (topk_ids) cudaFree(topk_ids);
        if (dsa_indices) cudaFree(dsa_indices);
        *this = Scratch{};
    }
    void ensure(int64_t n, const GLM52Config& c, int64_t local_heads,
                int64_t max_dense_inter, int64_t max_moe_inter) {
        if (n <= cap) return;
        free_all();
        const int64_t hidden = c.hidden_size, nheads = local_heads;
        const int64_t qlat = c.q_lora_rank, kvlat = c.kv_lora_rank;
        const int64_t rope = c.qk_rope_head_dim, nope = c.qk_nope_head_dim;
        const int64_t qk = c.qk_head_dim, vd = c.v_head_dim;
        const int64_t E = c.n_routed_experts, topk = c.num_experts_per_tok;
        const int64_t iH = c.index_n_heads, iD = c.index_head_dim;
        const int64_t ffmax = std::max<int64_t>(1, std::max(max_dense_inter, max_moe_inter));
        if (cudaMalloc(&d_tokens, n * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        hidden_d  = dmalloc(n * hidden);
        normed    = dmalloc(n * hidden);
        qa        = dmalloc(n * qlat);
        q         = dmalloc(n * nheads * qk);
        kva       = dmalloc(n * (kvlat + rope));
        ckv       = dmalloc(n * kvlat);
        kpe       = dmalloc(n * rope);
        kvb       = dmalloc(n * nheads * (nope + vd));
        attn      = dmalloc(n * nheads * vd);
        attn_proj = dmalloc(n * hidden);
        mlp       = dmalloc(n * hidden);
        gbuf      = dmalloc(n * ffmax);
        ubuf      = dmalloc(n * ffmax);
        gu        = dmalloc(n * ffmax);
        rlogits   = dmalloc(n * E);
        shbuf     = dmalloc(n * hidden);
        index_q   = dmalloc(n * iH * iD);
        index_k   = dmalloc(n * iD);
        index_w   = dmalloc(n * iH);
        dsa_scores = dmalloc(n * c.index_topk);
        if (cudaMalloc(&topk_ids, n * topk * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        if (cudaMalloc(&dsa_indices, n * c.index_topk * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        topk_w    = dmalloc(n * topk);
        moe_h_act = dmalloc(n * topk * ffmax);
        cap = n;
    }
};

// Device-resident MTP/NextN block (single-rank draft path): the NextN glue
// (enorm/hnorm/eh_proj/shared_head_norm), the full decoder layer, and a
// dedicated single-layer draft KV cache with the same g->max_ctx block size
// the attention kernels expect.
struct GpuMTP {
    bool ready = false;
    DLayer layer;
    DLinear eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *shared_head_norm = nullptr;
    float *Kc = nullptr, *Vc = nullptr, *Ic = nullptr;
    float *emb = nullptr;    // [H] draft-token embedding
    float *prev = nullptr;   // [H] previous pre-norm hidden state
    float *fused = nullptr;  // [2H] concat(enorm(emb), hnorm(prev))
};

struct GpuState {
    float* embed = nullptr;          // [vocab, H] (fp32, when embed is dequantized)
    uint8_t* embed_q = nullptr;       // [vocab, H] quant table (GGUF embed)
    uint32_t embed_type = 0;          // GGML type of embed_q (0 => use fp32 embed)
    int64_t embed_row_bytes = 0;
    // TP mode keeps the ~1 GiB quant embed table on the HOST (one 24 KiB row
    // dequant + upload per token) instead of replicating it on every rank.
    bool embed_host = false;
    float* final_norm = nullptr;
    DLinear lm_head;
    int64_t lm_head_off = 0;         // first vocab row of this rank's TP shard
    std::vector<DLayer> layers;
    GpuMTP mtp;
    int64_t local_heads = 0;
    int64_t max_dense_inter = 0;
    int64_t max_moe_inter = 0;

    // Persistent KV cache: one contiguous block per layer of capacity max_ctx,
    // laid out [max_ctx, local_heads, hc]. A single-entry block_table (=0) maps the
    // sequence's one logical block to it, so the paged-attention kernel runs
    // unchanged with block_size = max_ctx.
    std::vector<float*> Kc, Vc, Ic;
    int64_t max_ctx = 0;
    int*   block_table = nullptr;    // single entry, value 0
    float* logits_d = nullptr;       // [vocab]
    Scratch sc;
    int64_t cur_len = 0;             // positions currently valid in the cache

    // Flash-decoding partials: per (head, split) accumulator + softmax stats.
    float *part_acc = nullptr, *part_m = nullptr, *part_l = nullptr;
};

// Key-range splits for flash-decoding (decode-step attention parallelism).
static constexpr int64_t kMaxSplits = 128;

// Debug: print a checksum of a device buffer at each forward stage when
// GLMSERVE_DEBUG_CSUM is set (used to localize single-vs-TP divergence).
static void dbg_csum(const char* tag, const float* dptr, int64_t n) {
    static const bool on = std::getenv("GLMSERVE_DEBUG_CSUM") != nullptr;
    if (!on) return;
    std::vector<float> h(static_cast<size_t>(n));
    cudaMemcpy(h.data(), dptr, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0, sa = 0.0;
    for (float v : h) { s += v; sa += std::fabs(v); }
    std::fprintf(stderr, "[csum] %-14s n=%lld sum=%+.6e abs=%.6e\n", tag, (long long)n, s, sa);
}

static bool is_shared_dsa_layer(const GLM52Config& c, int64_t global_layer) {
    return c.use_dsa &&
           global_layer >= 0 &&
           global_layer < static_cast<int64_t>(c.indexer_types.size()) &&
           c.indexer_types[static_cast<size_t>(global_layer)] != "full";
}

bool GLM52Model::upload_to_gpu(int64_t max_ctx) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        GLM_WARN("no CUDA device available; staying on CPU path");
        return false;
    }
    Timer t;
    auto* g = new GpuState();
    g->local_heads = tp_heads_;
    GLM_CHECK(g->local_heads > 0, "GPU upload requires at least one local attention head");

    if (dist_.is_first_stage()) {
        if (embed_lin_.has_q() &&
            (dist_.world_size > 1 || std::getenv("GLMSERVE_HOST_EMBED") != nullptr)) {
            // TP: keep the quant embed table host-resident; the forward
            // dequantizes one row per token on the CPU and uploads it. Copy it
            // out of the mmap into anon memory so neither the upload eviction
            // below nor kernel reclaim can turn decode-time embed lookups into
            // per-token NFS page faults.
            if (embed_lin_.qbuf.empty() && embed_lin_.qdata) {
                GLM_CHECK(embed_lin_.qsrc_stride() == embed_lin_.row_bytes,
                          "host embed table must be contiguous");
                const size_t nb = static_cast<size_t>(embed_lin_.out_features) *
                                  static_cast<size_t>(embed_lin_.row_bytes);
                embed_lin_.qbuf.assign(embed_lin_.qdata, embed_lin_.qdata + nb);
                embed_lin_.qdata = embed_lin_.qbuf.data();
            }
            g->embed_host = true;
        } else if (embed_lin_.has_q()) {
            // GGUF quant embedding: keep on device as a quant table, gather via dequant.
            g->embed_type = embed_lin_.qtype;
            g->embed_row_bytes = embed_lin_.row_bytes;
            touch_ranges({{embed_lin_.qdata,
                static_cast<size_t>(embed_lin_.out_features * embed_lin_.row_bytes)}});
            g->embed_q = upload_bytes(embed_lin_.qdata,
                static_cast<size_t>(embed_lin_.out_features * embed_lin_.row_bytes));
            if (std::getenv("GLMSERVE_DEBUG_CSUM")) {
                // Self-check on the real table: device gather of one row vs the
                // CPU reference dequant of the same row.
                const int probe_tok = 151331;
                const int64_t H = cfg_.hidden_size;
                std::vector<float> cpu_row = gguf_dequantize_row(
                    embed_lin_.qtype, embed_lin_.qdata, static_cast<uint64_t>(H),
                    static_cast<uint64_t>(probe_tok));
                int* dtok = nullptr; float* drow = nullptr;
                cudaMalloc(&dtok, sizeof(int));
                cudaMalloc(&drow, H * sizeof(float));
                cudaMemcpy(dtok, &probe_tok, sizeof(int), cudaMemcpyHostToDevice);
                glmserve::cuda::embed_gather_q(g->embed_type, g->embed_q, dtok, drow,
                                               1, H, g->embed_row_bytes);
                std::vector<float> gpu_row(static_cast<size_t>(H));
                cudaMemcpy(gpu_row.data(), drow, H * sizeof(float), cudaMemcpyDeviceToHost);
                double md = 0; int64_t mdi = -1;
                for (int64_t i = 0; i < H; ++i) {
                    double dd = std::fabs(cpu_row[i] - gpu_row[i]);
                    if (dd > md) { md = dd; mdi = i; }
                }
                std::fprintf(stderr, "[csum] embed self-check tok=%d max_abs_diff=%.6e at %lld "
                             "(cpu=%.6f gpu=%.6f)\n", probe_tok, md, (long long)mdi,
                             mdi >= 0 ? cpu_row[mdi] : 0.0f, mdi >= 0 ? gpu_row[mdi] : 0.0f);
                cudaFree(dtok); cudaFree(drow);
            }
        } else {
            GLM_CHECK(!embed_tokens_.empty(), "first GPU pipeline stage is missing embeddings");
            g->embed = upload(embed_tokens_);
        }
    }
    if (dist_.is_last_stage()) {
        GLM_CHECK(!final_norm_.w.empty(), "last GPU pipeline stage is missing final norm");
        GLM_CHECK(lm_head_.valid(), "last GPU pipeline stage is missing lm_head");
        g->final_norm = upload(final_norm_.w);
        {
            std::vector<PfRange> rs;
            collect_range(lm_head_, rs);
            touch_ranges(rs);
        }
        g->lm_head = up_lin(lm_head_);
        g->lm_head_off = lm_head_shard_off_;
    }
    GLM_INFO("gpu upload: rank %d embed+head done (%.2f GiB, %.1f s)",
             dist_.rank, g_upload_bytes / 1073741824.0, t.ms() / 1000.0);

    const auto& c = cfg_;
    const int64_t hidden = c.hidden_size;
    g->layers.resize(layers_.size());

    // Touch-ahead prefetch of the mmap'd upload source (see PfRange above).
    // GLMSERVE_UPLOAD_PREFETCH = layers ahead (default 4, 0 disables).
    int pf_ahead = 4;
    if (const char* e = std::getenv("GLMSERVE_UPLOAD_PREFETCH")) pf_ahead = std::atoi(e);
    // Consumed-source eviction (see evict_ranges above).
    // GLMSERVE_EVICT_LAG = layers behind the upload (default 8, negative disables).
    int evict_lag = 8;
    if (const char* e = std::getenv("GLMSERVE_EVICT_LAG")) evict_lag = std::atoi(e);
    const GGUFModel* evict_gg =
        (evict_lag >= 0 && gguf_weights_) ? &gguf_weights_->gguf() : nullptr;
    if (evict_gg && dist_.is_last_stage()) {
        // lm_head was uploaded above and is never read from the mmap again.
        std::vector<PfRange> rs;
        collect_range(lm_head_, rs);
        evict_ranges(evict_gg, std::move(rs));
    }
    std::atomic<size_t> pf_uploaded{0};
    std::atomic<bool> pf_stop{false};
    std::thread pf_thread;
    if (pf_ahead > 0) {
        pf_thread = std::thread([&] {
            for (size_t k = 0; k < layers_.size() && !pf_stop.load(); ++k) {
                while (!pf_stop.load() &&
                       k > pf_uploaded.load() + static_cast<size_t>(pf_ahead))
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (pf_stop.load()) break;
                touch_ranges(layer_prefetch_ranges(layers_[k]));
            }
        });
    }
    struct PfJoin {
        std::atomic<bool>& stop; std::thread& t;
        ~PfJoin() { stop.store(true); if (t.joinable()) t.join(); }
    } pf_join{pf_stop, pf_thread};

    for (size_t i = 0; i < layers_.size(); ++i) {
        const Layer& L = layers_[i];
        DLayer& d = g->layers[i];
        d.input_norm = upload(L.input_norm.w);
        d.post_norm  = upload(L.post_attn_norm.w);
        d.q_a_norm   = upload(L.q_a_norm.w);
        d.kv_a_norm  = upload(L.kv_a_norm.w);
        d.q_a = up_lin(L.q_a_proj); d.q_b = up_lin(L.q_b_proj);
        d.kv_a = up_lin(L.kv_a_proj); d.kv_b = up_lin(L.kv_b_proj);
        d.o = up_lin(L.o_proj);
        if (L.indexer.valid()) {
            d.index_wq_b = up_lin(L.indexer.wq_b);
            d.index_wk = up_lin(L.indexer.wk);
            d.index_weights = up_lin(L.indexer.weights_proj);
            d.index_k_norm = upload(L.indexer.k_norm.w);
            if (!L.indexer.k_norm_bias.empty()) d.index_k_bias = upload(L.indexer.k_norm_bias);
        }
        d.is_dense = L.is_dense;
        if (L.is_dense) {
            d.gate = up_lin(L.dense_mlp.gate_proj);
            d.up   = up_lin(L.dense_mlp.up_proj);
            d.down = up_lin(L.dense_mlp.down_proj);
            g->max_dense_inter = std::max(g->max_dense_inter, d.gate.out);
        } else {
            d.router = up_lin(L.moe.router);
            if (!L.moe.e_bias.empty()) d.e_bias = upload(L.moe.e_bias);
            // pack experts contiguously: [E, moei, hidden] (gate/up), [E, hidden, moei] (down)
            int64_t E = static_cast<int64_t>(L.moe.experts.size());
            d.expert_inter = E > 0 ? L.moe.experts[0].gate_proj.out_features : 0;
            GLM_CHECK(d.expert_inter > 0, "MoE layer has no expert intermediate dimension");
            bool any_int4 = false, all_int4 = true;
            bool any_gguf = false, all_gguf = true;
            for (const Expert& e : L.moe.experts) {
                const bool qi = e.gate_proj.quantized_int4 || e.up_proj.quantized_int4 ||
                                 e.down_proj.quantized_int4;
                const bool qg = e.gate_proj.has_q() && e.up_proj.has_q() && e.down_proj.has_q();
                any_int4 = any_int4 || qi;  all_int4 = all_int4 && (qi && !qg);
                any_gguf = any_gguf || qg;   all_gguf  = all_gguf  && qg;
            }
            GLM_CHECK(!(any_int4 && any_gguf),
                      "mixed int4/GGUF-quant routed expert weights are not supported");
            d.quantized_experts = all_int4;
            d.quant_experts_gguf = all_gguf;
            std::vector<float> pg, pu, pd, sg, su, sd;
            std::vector<uint8_t> qg, qu, qd;
            if (d.quant_experts_gguf) {
                d.qexp_gate_type = L.moe.experts[0].gate_proj.qtype;
                d.qexp_up_type   = L.moe.experts[0].up_proj.qtype;
                d.qexp_down_type = L.moe.experts[0].down_proj.qtype;
                d.qexp_gate_rb = L.moe.experts[0].gate_proj.row_bytes;
                d.qexp_up_rb   = L.moe.experts[0].up_proj.row_bytes;
                d.qexp_down_rb = L.moe.experts[0].down_proj.row_bytes;
                d.qexp_out_l = d.expert_inter;
                d.qexp_in_l  = d.expert_inter;
                // Allocate packed device buffers up front; each expert is
                // 2D-copied straight from the mmap'd payload (no host merge).
                if (timed_malloc(reinterpret_cast<void**>(&d.qexp_gate_q), static_cast<size_t>(E * d.expert_inter * d.qexp_gate_rb)) != cudaSuccess ||
                    timed_malloc(reinterpret_cast<void**>(&d.qexp_up_q), static_cast<size_t>(E * d.expert_inter * d.qexp_up_rb)) != cudaSuccess ||
                    timed_malloc(reinterpret_cast<void**>(&d.qexp_down_q), static_cast<size_t>(E * hidden * d.qexp_down_rb)) != cudaSuccess)
                    throw std::runtime_error("cudaMalloc for GGUF expert buffers failed");
            } else if (d.quantized_experts) {
                d.exp_group_size = L.moe.experts[0].gate_proj.group_size;
                const int64_t gate_packed = (hidden + 1) / 2;
                const int64_t gate_groups = (hidden + d.exp_group_size - 1) / d.exp_group_size;
                const int64_t down_packed = (d.expert_inter + 1) / 2;
                const int64_t down_groups = (d.expert_inter + d.exp_group_size - 1) / d.exp_group_size;
                qg.reserve(E * d.expert_inter * gate_packed);
                qu.reserve(E * d.expert_inter * gate_packed);
                qd.reserve(E * hidden * down_packed);
                sg.reserve(E * d.expert_inter * gate_groups);
                su.reserve(E * d.expert_inter * gate_groups);
                sd.reserve(E * hidden * down_groups);
            } else {
                pg.reserve(E * d.expert_inter * hidden);
                pu.reserve(E * d.expert_inter * hidden);
                pd.reserve(E * hidden * d.expert_inter);
            }
            int64_t ei = 0;
            for (const Expert& e : L.moe.experts) {
                GLM_CHECK(e.gate_proj.out_features == d.expert_inter &&
                          e.up_proj.out_features == d.expert_inter &&
                          e.down_proj.in_features == d.expert_inter,
                          "inconsistent sharded expert dimensions");
                if (d.quant_experts_gguf) {
                    GLM_CHECK(e.gate_proj.has_q() && e.up_proj.has_q() && e.down_proj.has_q(),
                              "GGUF-quant expert missing quant weight");
                    const uint8_t* gp = e.gate_proj.qbuf.empty() ? e.gate_proj.qdata : e.gate_proj.qbuf.data();
                    const uint8_t* up = e.up_proj.qbuf.empty() ? e.up_proj.qdata : e.up_proj.qbuf.data();
                    const uint8_t* dp = e.down_proj.qbuf.empty() ? e.down_proj.qdata : e.down_proj.qbuf.data();
                    upload_bytes_2d_into(d.qexp_gate_q,
                        static_cast<size_t>(ei * d.expert_inter * d.qexp_gate_rb), gp,
                        static_cast<size_t>(d.expert_inter), static_cast<size_t>(d.qexp_gate_rb),
                        static_cast<size_t>(e.gate_proj.qsrc_stride()));
                    upload_bytes_2d_into(d.qexp_up_q,
                        static_cast<size_t>(ei * d.expert_inter * d.qexp_up_rb), up,
                        static_cast<size_t>(d.expert_inter), static_cast<size_t>(d.qexp_up_rb),
                        static_cast<size_t>(e.up_proj.qsrc_stride()));
                    upload_bytes_2d_into(d.qexp_down_q,
                        static_cast<size_t>(ei * hidden * d.qexp_down_rb), dp,
                        static_cast<size_t>(hidden), static_cast<size_t>(d.qexp_down_rb),
                        static_cast<size_t>(e.down_proj.qsrc_stride()));
                } else if (d.quantized_experts) {
                    GLM_CHECK(e.gate_proj.quantized_int4 && e.up_proj.quantized_int4 &&
                              e.down_proj.quantized_int4 &&
                              e.gate_proj.group_size == d.exp_group_size &&
                              e.up_proj.group_size == d.exp_group_size &&
                              e.down_proj.group_size == d.exp_group_size,
                              "mixed or inconsistent quantized expert weights");
                    qg.insert(qg.end(), e.gate_proj.qweight.begin(), e.gate_proj.qweight.end());
                    qu.insert(qu.end(), e.up_proj.qweight.begin(), e.up_proj.qweight.end());
                    qd.insert(qd.end(), e.down_proj.qweight.begin(), e.down_proj.qweight.end());
                    sg.insert(sg.end(), e.gate_proj.scales.begin(), e.gate_proj.scales.end());
                    su.insert(su.end(), e.up_proj.scales.begin(), e.up_proj.scales.end());
                    sd.insert(sd.end(), e.down_proj.scales.begin(), e.down_proj.scales.end());
                } else {
                    GLM_CHECK(e.gate_proj.has_f32() && e.up_proj.has_f32() && e.down_proj.has_f32(),
                              "routed expert missing fp32 resident weights");
                    pg.insert(pg.end(), e.gate_proj.w.begin(), e.gate_proj.w.end());
                    pu.insert(pu.end(), e.up_proj.w.begin(), e.up_proj.w.end());
                    pd.insert(pd.end(), e.down_proj.w.begin(), e.down_proj.w.end());
                }
                ++ei;
            }
            if (d.quant_experts_gguf) {
                // experts already packed on device via upload_bytes_2d_into
            } else if (d.quantized_experts) {
                d.qexp_gate = upload_u8(qg); d.qexp_up = upload_u8(qu); d.qexp_down = upload_u8(qd);
                d.sexp_gate = upload(sg); d.sexp_up = upload(su); d.sexp_down = upload(sd);
            } else {
                d.exp_gate = upload(pg); d.exp_up = upload(pu); d.exp_down = upload(pd);
            }
            g->max_moe_inter = std::max(g->max_moe_inter, d.expert_inter);
            if (L.moe.has_shared) {
                d.sh_gate = up_lin(L.moe.shared.gate_proj);
                d.sh_up   = up_lin(L.moe.shared.up_proj);
                d.sh_down = up_lin(L.moe.shared.down_proj);
                g->max_moe_inter = std::max(g->max_moe_inter, d.sh_gate.out);
            }
        }
        pf_uploaded.store(i + 1);
        if (evict_gg && i >= static_cast<size_t>(evict_lag))
            evict_ranges(evict_gg,
                         layer_prefetch_ranges(layers_[i - static_cast<size_t>(evict_lag)]));
        const double el = t.ms() / 1000.0;
        GLM_INFO("gpu upload: rank %d layer %zu/%zu (%.2f GiB, %.1f s, %.0f MB/s) "
                 "[layer: malloc %.2fs memcpy %.2fs stage %.2fs]",
                 dist_.rank, i + 1, layers_.size(), g_upload_bytes / 1073741824.0,
                 el, el > 0 ? g_upload_bytes / 1048576.0 / el : 0.0,
                 g_malloc_us / 1e6, g_memcpy_us / 1e6, g_stage_us / 1e6);
        g_malloc_us = g_memcpy_us = g_stage_us = 0;
    }

    // Persistent KV cache + small persistent buffers.
    const int64_t nheads = g->local_heads, hc = c.kv_cache_head_dim();
    g->max_ctx = max_ctx > 0 ? max_ctx : 4096;
    g->Kc.resize(layers_.size());
    g->Vc.resize(layers_.size());
    g->Ic.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        g->Kc[i] = dmalloc(g->max_ctx * nheads * hc);
        g->Vc[i] = dmalloc(g->max_ctx * nheads * hc);
        g->Ic[i] = dmalloc(g->max_ctx * c.index_head_dim);
    }
    cudaMalloc(&g->block_table, sizeof(int));
    int zero = 0;
    cudaMemcpy(g->block_table, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (dist_.is_last_stage())
        g->logits_d = dmalloc(c.vocab_size);

    // Flash-decoding partials: [n_heads, kMaxSplits, hc] + [n_heads, kMaxSplits].
    g->part_acc = dmalloc(nheads * kMaxSplits * hc);
    g->part_m   = dmalloc(nheads * kMaxSplits);
    g->part_l   = dmalloc(nheads * kMaxSplits);

    // MTP/NextN draft block (single-rank only: the draft path needs embeddings
    // and the lm_head co-resident, and mtp_draft_logits_gpu mirrors the CPU
    // reference which is single-rank).
    if (mtp_ready() && dist_.world_size == 1) {
        const MTPBlock& m = mtp_blocks_[0];
        GpuMTP& gm = g->mtp;
        DLayer& d = gm.layer;
        const Layer& L = m.layer;
        d.input_norm = upload(L.input_norm.w);
        d.post_norm  = upload(L.post_attn_norm.w);
        d.q_a_norm   = upload(L.q_a_norm.w);
        d.kv_a_norm  = upload(L.kv_a_norm.w);
        d.q_a = up_lin(L.q_a_proj); d.q_b = up_lin(L.q_b_proj);
        d.kv_a = up_lin(L.kv_a_proj); d.kv_b = up_lin(L.kv_b_proj);
        d.o = up_lin(L.o_proj);
        if (L.indexer.valid()) {
            d.index_wq_b = up_lin(L.indexer.wq_b);
            d.index_wk = up_lin(L.indexer.wk);
            d.index_weights = up_lin(L.indexer.weights_proj);
            d.index_k_norm = upload(L.indexer.k_norm.w);
            if (!L.indexer.k_norm_bias.empty()) d.index_k_bias = upload(L.indexer.k_norm_bias);
        }
        d.is_dense = L.is_dense;
        if (L.is_dense) {
            d.gate = up_lin(L.dense_mlp.gate_proj);
            d.up   = up_lin(L.dense_mlp.up_proj);
            d.down = up_lin(L.dense_mlp.down_proj);
            g->max_dense_inter = std::max(g->max_dense_inter, d.gate.out);
        } else {
            d.router = up_lin(L.moe.router);
            if (!L.moe.e_bias.empty()) d.e_bias = upload(L.moe.e_bias);
            int64_t E = static_cast<int64_t>(L.moe.experts.size());
            d.expert_inter = E > 0 ? L.moe.experts[0].gate_proj.out_features : 0;
            GLM_CHECK(d.expert_inter > 0, "MTP MoE layer has no expert intermediate dimension");
            bool all_gguf = !L.moe.experts.empty();
            for (const Expert& e : L.moe.experts)
                all_gguf = all_gguf && e.gate_proj.has_q() && e.up_proj.has_q() && e.down_proj.has_q();
            d.quant_experts_gguf = all_gguf;
            if (d.quant_experts_gguf) {
                d.qexp_gate_type = L.moe.experts[0].gate_proj.qtype;
                d.qexp_up_type   = L.moe.experts[0].up_proj.qtype;
                d.qexp_down_type = L.moe.experts[0].down_proj.qtype;
                d.qexp_gate_rb = L.moe.experts[0].gate_proj.row_bytes;
                d.qexp_up_rb   = L.moe.experts[0].up_proj.row_bytes;
                d.qexp_down_rb = L.moe.experts[0].down_proj.row_bytes;
                d.qexp_out_l = d.expert_inter; d.qexp_in_l = d.expert_inter;
                if (cudaMalloc(&d.qexp_gate_q, static_cast<size_t>(E * d.expert_inter * d.qexp_gate_rb)) != cudaSuccess ||
                    cudaMalloc(&d.qexp_up_q, static_cast<size_t>(E * d.expert_inter * d.qexp_up_rb)) != cudaSuccess ||
                    cudaMalloc(&d.qexp_down_q, static_cast<size_t>(E * hidden * d.qexp_down_rb)) != cudaSuccess)
                    throw std::runtime_error("cudaMalloc for MTP GGUF expert buffers failed");
                int64_t ei = 0;
                for (const Expert& e : L.moe.experts) {
                    const uint8_t* gp = e.gate_proj.qbuf.empty() ? e.gate_proj.qdata : e.gate_proj.qbuf.data();
                    const uint8_t* up = e.up_proj.qbuf.empty() ? e.up_proj.qdata : e.up_proj.qbuf.data();
                    const uint8_t* dp = e.down_proj.qbuf.empty() ? e.down_proj.qdata : e.down_proj.qbuf.data();
                    upload_bytes_2d_into(d.qexp_gate_q,
                        static_cast<size_t>(ei * d.expert_inter * d.qexp_gate_rb), gp,
                        static_cast<size_t>(d.expert_inter), static_cast<size_t>(d.qexp_gate_rb),
                        static_cast<size_t>(e.gate_proj.qsrc_stride()));
                    upload_bytes_2d_into(d.qexp_up_q,
                        static_cast<size_t>(ei * d.expert_inter * d.qexp_up_rb), up,
                        static_cast<size_t>(d.expert_inter), static_cast<size_t>(d.qexp_up_rb),
                        static_cast<size_t>(e.up_proj.qsrc_stride()));
                    upload_bytes_2d_into(d.qexp_down_q,
                        static_cast<size_t>(ei * hidden * d.qexp_down_rb), dp,
                        static_cast<size_t>(hidden), static_cast<size_t>(d.qexp_down_rb),
                        static_cast<size_t>(e.down_proj.qsrc_stride()));
                    ++ei;
                }
            } else {
                std::vector<float> pg, pu, pd;
                pg.reserve(E * d.expert_inter * hidden);
                pu.reserve(E * d.expert_inter * hidden);
                pd.reserve(E * hidden * d.expert_inter);
                for (const Expert& e : L.moe.experts) {
                    GLM_CHECK(e.gate_proj.has_f32() && e.up_proj.has_f32() && e.down_proj.has_f32(),
                              "MTP routed expert missing resident weights");
                    pg.insert(pg.end(), e.gate_proj.w.begin(), e.gate_proj.w.end());
                    pu.insert(pu.end(), e.up_proj.w.begin(), e.up_proj.w.end());
                    pd.insert(pd.end(), e.down_proj.w.begin(), e.down_proj.w.end());
                }
                d.exp_gate = upload(pg); d.exp_up = upload(pu); d.exp_down = upload(pd);
            }
            g->max_moe_inter = std::max(g->max_moe_inter, d.expert_inter);
            if (L.moe.has_shared) {
                d.sh_gate = up_lin(L.moe.shared.gate_proj);
                d.sh_up   = up_lin(L.moe.shared.up_proj);
                d.sh_down = up_lin(L.moe.shared.down_proj);
                g->max_moe_inter = std::max(g->max_moe_inter, d.sh_gate.out);
            }
        }
        gm.eh_proj = up_lin(m.eh_proj);
        gm.enorm = upload(m.enorm.w);
        gm.hnorm = upload(m.hnorm.w);
        gm.shared_head_norm = upload(m.shared_head_norm.w);
        gm.Kc = dmalloc(g->max_ctx * nheads * hc);
        gm.Vc = dmalloc(g->max_ctx * nheads * hc);
        gm.Ic = dmalloc(g->max_ctx * c.index_head_dim);
        gm.emb   = dmalloc(hidden);
        gm.prev  = dmalloc(hidden);
        gm.fused = dmalloc(2 * hidden);
        gm.ready = true;
    }

    gpu_state_ = g;
    size_t freeb = 0, total = 0; cudaMemGetInfo(&freeb, &total);
    GLM_INFO("uploaded weights to GPU in %.1f s; KV cache ctx=%lld (%.2f GiB free / %.2f GiB total)",
             t.ms() / 1000.0, (long long)g->max_ctx, freeb / 1073741824.0, total / 1073741824.0);
    return true;
}

int64_t GLM52Model::gpu_kv_ctx() const {
    return gpu_state_ ? static_cast<GpuState*>(gpu_state_)->max_ctx : 0;
}

namespace {
// One decoder layer on `n` tokens whose activations are in g->sc.hidden_d.
// Appends the layer's K/V to (Kc, Vc, Ic) at absolute offset `start_pos` and
// attends over [0, start_pos+n). The cache buffers are always g->max_ctx
// tokens deep (the attention kernels use g->max_ctx as the block size).
// Returns the updated have_shared_dsa_indices flag.
bool run_layer_step(GpuState* g, const GLM52Config& c, Communicator* comm,
                    const DistConfig& dist, const DLayer& d,
                    float* Kc, float* Vc, float* Ic,
                    int64_t n, int64_t start_pos, bool have_shared_dsa_indices) {
    using namespace glmserve::cuda;
    Scratch& s = g->sc;
    const int64_t hidden = c.hidden_size, nheads = g->local_heads;
    const int64_t qlat = c.q_lora_rank, kvlat = c.kv_lora_rank;
    const int64_t nope = c.qk_nope_head_dim, rope = c.qk_rope_head_dim;
    const int64_t qk = c.qk_head_dim, vd = c.v_head_dim, hc = c.kv_cache_head_dim();
    const int64_t E = c.n_routed_experts, topk = c.num_experts_per_tok;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const bool il = c.rope_interleave;
    const bool tp = dist.tp_size > 1;

    // attention sub-block
    rmsnorm(s.hidden_d, d.input_norm, s.normed, n, hidden, eps);
    dbg_csum("normed", s.normed, n * hidden);
    gemm_linear(s.normed, d.q_a, nullptr, s.qa, n);
    rmsnorm(s.qa, d.q_a_norm, s.qa, n, qlat, eps);
    dbg_csum("qa", s.qa, n * qlat);
    gemm_linear(s.qa, d.q_b, nullptr, s.q, n);
    dbg_csum("q(local)", s.q, n * nheads * qk);
    gemm_linear(s.normed, d.kv_a, nullptr, s.kva, n);
    slice_rows(s.kva, s.ckv, n, kvlat + rope, 0, kvlat);
    rmsnorm(s.ckv, d.kv_a_norm, s.ckv, n, kvlat, eps);
    gemm_linear(s.ckv, d.kv_b, nullptr, s.kvb, n);
    slice_rows(s.kva, s.kpe, n, kvlat + rope, kvlat, rope);
    rope_k(s.kpe, n, rope, start_pos, c.rope_theta, il);
    rope_q(s.q, n, nheads, qk, nope, rope, start_pos, c.rope_theta, il);

    float* Kbase = Kc + start_pos * nheads * hc;
    float* Vbase = Vc + start_pos * nheads * hc;
    assemble_kv(s.kvb, s.kpe, Kbase, Vbase, n, nheads, nope, rope, vd, hc);

    const int64_t ctx_after = start_pos + n;
    const bool sparse_dsa = c.use_dsa && ctx_after > c.index_topk;
    const bool learned_dsa = sparse_dsa && d.index_wq_b.out > 0;
    const bool shared_dsa = sparse_dsa && !learned_dsa && have_shared_dsa_indices;
    if (learned_dsa) {
        gemm_linear(s.qa, d.index_wq_b, nullptr, s.index_q, n);
        gemm_linear(s.normed, d.index_wk, nullptr, s.index_k, n);
        layernorm(s.index_k, d.index_k_norm, d.index_k_bias, s.index_k, n,
                  c.index_head_dim, 1e-6f);
        rope_index_q(s.index_q, n, c.index_n_heads, c.index_head_dim, rope,
                     start_pos, c.rope_theta, false);
        rope_index_q(s.index_k, n, 1, c.index_head_dim, rope,
                     start_pos, c.rope_theta, false);
        cudaMemcpy(Ic + start_pos * c.index_head_dim, s.index_k,
                   n * c.index_head_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        gemm_linear(s.normed, d.index_weights, nullptr, s.index_w, n);
        dsa_select_topk(s.index_q, Ic, s.index_w, n, start_pos,
                        c.index_n_heads, c.index_head_dim, c.index_topk,
                        1.0f / std::sqrt(static_cast<float>(c.index_head_dim)),
                        1.0f / std::sqrt(static_cast<float>(c.index_n_heads)),
                        s.dsa_indices, s.dsa_scores);
        have_shared_dsa_indices = true;
        attention_dsa_indexed_paged(s.q, Kc, Vc, g->block_table,
                                    s.dsa_indices, n, start_pos, nheads, nheads, qk,
                                    g->max_ctx, c.index_topk, scale, s.attn);
    } else if (shared_dsa) {
        attention_dsa_indexed_paged(s.q, Kc, Vc, g->block_table,
                                    s.dsa_indices, n, start_pos, nheads, nheads, qk,
                                    g->max_ctx, c.index_topk, scale, s.attn);
    } else if (sparse_dsa) {
        attention_dsa_paged(s.q, Kc, Vc, g->block_table, n, start_pos,
                            nheads, nheads, qk, g->max_ctx, c.index_topk, scale, s.attn, 0,
                            g->part_acc, g->part_m, g->part_l, kMaxSplits);
    } else if (n == 1) {
        attention_decode_paged(s.q, Kc, Vc, g->block_table, start_pos,
                               nheads, nheads, qk, g->max_ctx, scale, s.attn,
                               g->part_acc, g->part_m, g->part_l, kMaxSplits);
    } else {
        attention_dense_paged(s.q, Kc, Vc, g->block_table, n, start_pos,
                              nheads, nheads, qk, g->max_ctx, scale, s.attn);
    }
    dbg_csum("attn(local)", s.attn, n * nheads * vd);
    gemm_linear(s.attn, d.o, nullptr, s.attn_proj, n);
    if (tp) {
        GLM_CHECK(comm, "tensor-parallel GPU forward requires a communicator");
        comm->all_reduce_sum(s.attn_proj, n * hidden);
    }
    dbg_csum("attn_proj", s.attn_proj, n * hidden);
    add_inplace(s.hidden_d, s.attn_proj, n * hidden);

    // MLP sub-block
    rmsnorm(s.hidden_d, d.post_norm, s.normed, n, hidden, eps);
    if (d.is_dense) {
        const int64_t inter = d.gate.out;
        gemm_linear(s.normed, d.gate, nullptr, s.gbuf, n);
        dbg_csum("gbuf(local)", s.gbuf, n * inter);
        gemm_linear(s.normed, d.up,   nullptr, s.ubuf, n);
        dbg_csum("ubuf(local)", s.ubuf, n * inter);
        silu_mul(s.gbuf, s.ubuf, s.gu, n * inter);
        dbg_csum("gu(local)", s.gu, n * inter);
        gemm_linear(s.gu, d.down, nullptr, s.mlp, n);
        dbg_csum("mlp_pre(local)", s.mlp, n * hidden);
    } else {
        const int64_t moei = d.expert_inter;
        gemm_linear(s.normed, d.router, nullptr, s.rlogits, n);
        moe_router(s.rlogits, d.e_bias, n, E, topk, c.norm_topk_prob,
                   static_cast<float>(c.routed_scaling_factor), s.topk_ids, s.topk_w);
        cudaMemset(s.mlp, 0, n * hidden * sizeof(float));
        if (d.quant_experts_gguf) {
            moe_expert_ffn_q(d.qexp_gate_type, d.qexp_up_type, d.qexp_down_type,
                             s.normed, s.topk_ids, s.topk_w,
                             d.qexp_gate_q, d.qexp_up_q, d.qexp_down_q,
                             n, topk, hidden, moei, E,
                             d.qexp_gate_rb, d.qexp_up_rb, d.qexp_down_rb,
                             s.moe_h_act, s.mlp);
        } else if (d.quantized_experts) {
            moe_expert_ffn_w4a16(s.normed, s.topk_ids, s.topk_w,
                                 d.qexp_gate, d.sexp_gate, d.qexp_up, d.sexp_up,
                                 d.qexp_down, d.sexp_down, n, topk, hidden, moei, E,
                                 d.exp_group_size, s.mlp);
        } else {
            moe_expert_ffn(s.normed, s.topk_ids, s.topk_w, d.exp_gate, d.exp_up, d.exp_down,
                           n, topk, hidden, moei, E, s.mlp);
        }
        if (d.sh_gate.out > 0) {
            const int64_t sh_inter = d.sh_gate.out;
            gemm_linear(s.normed, d.sh_gate, nullptr, s.gbuf, n);
            gemm_linear(s.normed, d.sh_up,   nullptr, s.ubuf, n);
            silu_mul(s.gbuf, s.ubuf, s.gu, n * sh_inter);
            gemm_linear(s.gu, d.sh_down, nullptr, s.shbuf, n);
            add_inplace(s.mlp, s.shbuf, n * hidden);
        }
    }
    if (tp) comm->all_reduce_sum(s.mlp, n * hidden);
    dbg_csum("mlp", s.mlp, n * hidden);
    add_inplace(s.hidden_d, s.mlp, n * hidden);
    dbg_csum("hidden", s.hidden_d, n * hidden);
    return have_shared_dsa_indices;
}

// Run this stage's block stack on `n` tokens whose activations are already in
// g->sc.hidden_d. Appends each local layer's K/V to the persistent cache at
// absolute offset `start_pos` and attends over [0, start_pos+n). Leaves the
// final pre-norm hidden state in g->sc.hidden_d.
bool run_block_stack(GpuState* g, const GLM52Config& c, Communicator* comm,
                     const DistConfig& dist, int64_t n, int64_t start_pos,
                     bool have_shared_dsa_indices) {
    for (size_t l = 0; l < g->layers.size(); ++l) {
        have_shared_dsa_indices =
            run_layer_step(g, c, comm, dist, g->layers[l],
                           g->Kc[l], g->Vc[l], g->Ic[l],
                           n, start_pos, have_shared_dsa_indices);
    }
    return have_shared_dsa_indices;
}

// Final RMSNorm + lm_head over the LAST row, copy logits to host. Under a
// column-parallel lm_head shard each rank fills its vocab slice of a zeroed
// buffer and the slices are combined with one all-reduce (sum of disjoint
// slices == all-gather), so every rank ends with identical full logits.
std::vector<float> finish_logits(GpuState* g, const GLM52Config& c, int64_t n,
                                 Communicator* comm) {
    using namespace glmserve::cuda;
    GLM_CHECK(g->final_norm && g->logits_d && g->lm_head.out > 0,
              "finish_logits called on a non-final or incompletely uploaded GPU stage");
    rmsnorm(g->sc.hidden_d, g->final_norm, g->sc.normed, n, c.hidden_size,
            static_cast<float>(c.rms_norm_eps));
    const bool sharded = g->lm_head.out < c.vocab_size;
    if (sharded) {
        GLM_CHECK(g->lm_head_off + g->lm_head.out <= c.vocab_size,
                  "lm_head shard exceeds vocab");
        cudaMemset(g->logits_d, 0, c.vocab_size * sizeof(float));
    }
    gemm_linear(g->sc.normed + (n - 1) * c.hidden_size, g->lm_head, nullptr,
                g->logits_d + (sharded ? g->lm_head_off : 0), 1);
    if (sharded) {
        GLM_CHECK(comm, "sharded lm_head requires a communicator");
        if (std::getenv("GLMSERVE_DEBUG_CSUM")) {
            std::fprintf(stderr, "[csum] lm_head shard off=%lld out=%lld in=%lld rb=%lld\n",
                         (long long)g->lm_head_off, (long long)g->lm_head.out,
                         (long long)g->lm_head.in, (long long)g->lm_head.row_bytes);
            dbg_csum("logits_pre", g->logits_d + g->lm_head_off, g->lm_head.out);
            dbg_csum("logits_tail", g->logits_d + g->lm_head_off + g->lm_head.out - 1184, 1184);
        }
        comm->all_reduce_sum(g->logits_d, c.vocab_size);
        dbg_csum("logits_post", g->logits_d, c.vocab_size);
    }
    std::vector<float> logits(c.vocab_size);
    cudaMemcpy(logits.data(), g->logits_d, c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return logits;
}
}  // namespace

std::vector<float> GLM52Model::forward_gpu_prefill(const std::vector<int>& tokens) {
    GLM_CHECK(gpu_state_ != nullptr, "forward_gpu_prefill: call upload_to_gpu() first");
    using namespace glmserve::cuda;
    auto* g = static_cast<GpuState*>(gpu_state_);
    const int64_t n = static_cast<int64_t>(tokens.size());
    GLM_CHECK(n > 0, "forward_gpu_prefill: empty prompt");
    GLM_CHECK(n <= g->max_ctx, "forward_gpu_prefill: prompt %lld > GPU KV ctx %lld",
              (long long)n, (long long)g->max_ctx);

    g->sc.ensure(n, cfg_, g->local_heads, g->max_dense_inter, g->max_moe_inter);
    if (dist_.is_first_stage()) {
        GLM_CHECK(g->embed || g->embed_q || g->embed_host,
                  "forward_gpu_prefill: first GPU pipeline stage is missing embeddings");
        if (g->embed_host) {
            // Host-resident quant embed table (TP): dequantize the rows on the
            // CPU and upload the fp32 hidden block once.
            std::vector<float> h(static_cast<size_t>(n) * cfg_.hidden_size);
            for (int64_t t = 0; t < n; ++t) {
                std::vector<float> row = gguf_dequantize_row(
                    embed_lin_.qtype, embed_lin_.qdata,
                    static_cast<uint64_t>(cfg_.hidden_size),
                    static_cast<uint64_t>(tokens[static_cast<size_t>(t)]));
                std::copy(row.begin(), row.end(), h.begin() + t * cfg_.hidden_size);
            }
            cudaMemcpy(g->sc.hidden_d, h.data(), h.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
        } else {
        std::vector<int> toks(tokens.begin(), tokens.end());
        cudaMemcpy(g->sc.d_tokens, toks.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        if (g->embed_q)
            embed_gather_q(g->embed_type, g->embed_q, g->sc.d_tokens, g->sc.hidden_d,
                           n, cfg_.hidden_size, g->embed_row_bytes);
        else
            embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, n, cfg_.hidden_size);
        }
    } else {
        GLM_CHECK(comm_, "forward_gpu_prefill: non-first pipeline stage requires a communicator");
        comm_->pipeline_recv_prev(g->sc.hidden_d, n * cfg_.hidden_size);
    }

    dbg_csum("embed", g->sc.hidden_d, n * cfg_.hidden_size);
    const int64_t ctx_after = n;
    const bool sparse_dsa = cfg_.use_dsa && ctx_after > cfg_.index_topk;
    const bool recv_shared_dsa = sparse_dsa && !dist_.is_first_stage() &&
                                 is_shared_dsa_layer(cfg_, layer_begin_);
    if (recv_shared_dsa) {
        GLM_CHECK(comm_, "forward_gpu_prefill: DSA mask receive requires a communicator");
        comm_->pipeline_recv_prev_int(g->sc.dsa_indices, n * cfg_.index_topk);
    }

    bool have_shared_dsa = run_block_stack(g, cfg_, comm_, dist_, n, /*start_pos=*/0,
                                           recv_shared_dsa);
    g->cur_len = n;
    if (!dist_.is_last_stage()) {
        GLM_CHECK(comm_, "forward_gpu_prefill: non-last pipeline stage requires a communicator");
        comm_->pipeline_send_next(g->sc.hidden_d, n * cfg_.hidden_size);
        int64_t next_layer = layer_begin_ + num_layers();
        if (sparse_dsa && have_shared_dsa && is_shared_dsa_layer(cfg_, next_layer))
            comm_->pipeline_send_next_int(g->sc.dsa_indices, n * cfg_.index_topk);
        cuda_check("forward_gpu_prefill");
        return {};
    }
    std::vector<float> logits = finish_logits(g, cfg_, n, comm_);
    cuda_check("forward_gpu_prefill");
    return logits;
}

std::vector<float> GLM52Model::forward_gpu_decode(int token, int64_t pos) {
    GLM_CHECK(gpu_state_ != nullptr, "forward_gpu_decode: call upload_to_gpu() first");
    using namespace glmserve::cuda;
    auto* g = static_cast<GpuState*>(gpu_state_);
    GLM_CHECK(pos < g->max_ctx, "forward_gpu_decode: position %lld >= GPU KV ctx %lld "
              "(raise --ctx)", (long long)pos, (long long)g->max_ctx);

    g->sc.ensure(1, cfg_, g->local_heads, g->max_dense_inter, g->max_moe_inter);
    if (dist_.is_first_stage()) {
        GLM_CHECK(g->embed || g->embed_q || g->embed_host,
                  "forward_gpu_decode: first GPU pipeline stage is missing embeddings");
        if (g->embed_host) {
            std::vector<float> row = gguf_dequantize_row(
                embed_lin_.qtype, embed_lin_.qdata,
                static_cast<uint64_t>(cfg_.hidden_size), static_cast<uint64_t>(token));
            cudaMemcpy(g->sc.hidden_d, row.data(), row.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
        } else {
        cudaMemcpy(g->sc.d_tokens, &token, sizeof(int), cudaMemcpyHostToDevice);
        if (g->embed_q)
            embed_gather_q(g->embed_type, g->embed_q, g->sc.d_tokens, g->sc.hidden_d,
                           1, cfg_.hidden_size, g->embed_row_bytes);
        else
            embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, 1, cfg_.hidden_size);
        }
    } else {
        GLM_CHECK(comm_, "forward_gpu_decode: non-first pipeline stage requires a communicator");
        comm_->pipeline_recv_prev(g->sc.hidden_d, cfg_.hidden_size);
    }

    const int64_t ctx_after = pos + 1;
    const bool sparse_dsa = cfg_.use_dsa && ctx_after > cfg_.index_topk;
    const bool recv_shared_dsa = sparse_dsa && !dist_.is_first_stage() &&
                                 is_shared_dsa_layer(cfg_, layer_begin_);
    if (recv_shared_dsa) {
        GLM_CHECK(comm_, "forward_gpu_decode: DSA mask receive requires a communicator");
        comm_->pipeline_recv_prev_int(g->sc.dsa_indices, cfg_.index_topk);
    }

    bool have_shared_dsa = run_block_stack(g, cfg_, comm_, dist_, /*n=*/1, /*start_pos=*/pos,
                                           recv_shared_dsa);
    g->cur_len = pos + 1;
    if (!dist_.is_last_stage()) {
        GLM_CHECK(comm_, "forward_gpu_decode: non-last pipeline stage requires a communicator");
        comm_->pipeline_send_next(g->sc.hidden_d, cfg_.hidden_size);
        int64_t next_layer = layer_begin_ + num_layers();
        if (sparse_dsa && have_shared_dsa && is_shared_dsa_layer(cfg_, next_layer))
            comm_->pipeline_send_next_int(g->sc.dsa_indices, cfg_.index_topk);
        cuda_check("forward_gpu_decode");
        return {};
    }
    std::vector<float> logits = finish_logits(g, cfg_, 1, comm_);
    cuda_check("forward_gpu_decode");
    return logits;
}

std::vector<float> GLM52Model::mtp_draft_logits_gpu(const std::vector<int>& context_tokens,
                                                    const std::vector<int>& draft_tokens) {
    GLM_CHECK(gpu_state_ != nullptr, "mtp_draft_logits_gpu: call upload_to_gpu() first");
    GLM_CHECK(!context_tokens.empty(), "mtp requires nonempty context tokens");
    GLM_CHECK(!draft_tokens.empty(), "mtp requires nonempty draft tokens");
    GLM_CHECK(dist_.world_size == 1 && dist_.tp_size == 1 && dist_.pp_size == 1,
              "mtp_draft_logits_gpu currently supports the single-rank path");
    using namespace glmserve::cuda;
    auto* g = static_cast<GpuState*>(gpu_state_);
    GLM_CHECK(g->mtp.ready, "MTP weights are not resident on the GPU");

    const int64_t H = cfg_.hidden_size;
    const int64_t n = static_cast<int64_t>(context_tokens.size());
    const int64_t steps = static_cast<int64_t>(draft_tokens.size());
    GLM_CHECK(n <= g->max_ctx && steps <= g->max_ctx,
              "mtp_draft_logits_gpu: context/draft exceeds GPU KV ctx %lld",
              (long long)g->max_ctx);
    const float eps = static_cast<float>(cfg_.rms_norm_eps);
    GpuMTP& gm = g->mtp;

    // Trunk pass over the context (mirrors forward_gpu_prefill; clobbers the
    // persistent KV cache the same way). prev <- last pre-norm hidden row.
    g->sc.ensure(n, cfg_, g->local_heads, g->max_dense_inter, g->max_moe_inter);
    std::vector<int> toks(context_tokens.begin(), context_tokens.end());
    cudaMemcpy(g->sc.d_tokens, toks.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    if (g->embed_q)
        embed_gather_q(g->embed_type, g->embed_q, g->sc.d_tokens, g->sc.hidden_d,
                       n, H, g->embed_row_bytes);
    else
        embed_gather(g->embed, g->sc.d_tokens, g->sc.hidden_d, n, H);
    run_block_stack(g, cfg_, comm_, dist_, n, /*start_pos=*/0, false);
    g->cur_len = n;
    cudaMemcpy(gm.prev, g->sc.hidden_d + (n - 1) * H, H * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Draft steps: h_s = MTPBlock(eh_proj(concat(enorm(embed(x_s)), hnorm(h_{s-1})))),
    // logits_s = lm_head(shared_head_norm(h_s)). The MTP layer keeps its own
    // single-layer KV cache over the draft positions [0, steps).
    std::vector<float> logits(static_cast<size_t>(steps) * cfg_.vocab_size);
    for (int64_t s = 0; s < steps; ++s) {
        const int tok = draft_tokens[static_cast<size_t>(s)];
        cudaMemcpy(g->sc.d_tokens, &tok, sizeof(int), cudaMemcpyHostToDevice);
        if (g->embed_q)
            embed_gather_q(g->embed_type, g->embed_q, g->sc.d_tokens, gm.emb,
                           1, H, g->embed_row_bytes);
        else
            embed_gather(g->embed, g->sc.d_tokens, gm.emb, 1, H);
        rmsnorm(gm.emb,  gm.enorm, gm.fused,     1, H, eps);
        rmsnorm(gm.prev, gm.hnorm, gm.fused + H, 1, H, eps);
        gemm_linear(gm.fused, gm.eh_proj, nullptr, g->sc.hidden_d, 1);

        run_layer_step(g, cfg_, comm_, dist_, gm.layer, gm.Kc, gm.Vc, gm.Ic,
                       /*n=*/1, /*start_pos=*/s, false);
        cudaMemcpy(gm.prev, g->sc.hidden_d, H * sizeof(float), cudaMemcpyDeviceToDevice);

        rmsnorm(g->sc.hidden_d, gm.shared_head_norm, g->sc.normed, 1, H, eps);
        gemm_linear(g->sc.normed, g->lm_head, nullptr, g->logits_d, 1);
        cudaMemcpy(logits.data() + s * cfg_.vocab_size, g->logits_d,
                   cfg_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    cuda_check("mtp_draft_logits_gpu");
    return logits;
}

GLM52Model::~GLM52Model() {
    if (!gpu_state_) return;
    auto* g = static_cast<GpuState*>(gpu_state_);
    cudaFree(g->embed); cudaFree(g->embed_q); cudaFree(g->final_norm); free_lin(g->lm_head);
    for (auto& d : g->layers) {
        for (float* p : {d.input_norm, d.post_norm, d.q_a_norm, d.kv_a_norm,
                         d.index_k_norm, d.index_k_bias,
                         d.exp_gate, d.exp_up, d.exp_down,
                         d.sexp_gate, d.sexp_up, d.sexp_down, d.e_bias})
            if (p) cudaFree(p);
        for (uint8_t* p : {d.qexp_gate, d.qexp_up, d.qexp_down,
                           d.qexp_gate_q, d.qexp_up_q, d.qexp_down_q})
            if (p) cudaFree(p);
        for (DLinear* lin : {&d.q_a, &d.q_b, &d.kv_a, &d.kv_b, &d.o,
                             &d.index_wq_b, &d.index_wk, &d.index_weights,
                             &d.gate, &d.up, &d.down, &d.router,
                             &d.sh_gate, &d.sh_up, &d.sh_down})
            free_lin(*lin);
    }
    if (g->mtp.ready) {
        GpuMTP& gm = g->mtp;
        DLayer& d = gm.layer;
        for (float* p : {d.input_norm, d.post_norm, d.q_a_norm, d.kv_a_norm,
                         d.index_k_norm, d.index_k_bias,
                         d.exp_gate, d.exp_up, d.exp_down,
                         d.sexp_gate, d.sexp_up, d.sexp_down, d.e_bias})
            if (p) cudaFree(p);
        for (uint8_t* p : {d.qexp_gate, d.qexp_up, d.qexp_down,
                           d.qexp_gate_q, d.qexp_up_q, d.qexp_down_q})
            if (p) cudaFree(p);
        for (DLinear* lin : {&d.q_a, &d.q_b, &d.kv_a, &d.kv_b, &d.o,
                             &d.index_wq_b, &d.index_wk, &d.index_weights,
                             &d.gate, &d.up, &d.down, &d.router,
                             &d.sh_gate, &d.sh_up, &d.sh_down})
            free_lin(*lin);
        free_lin(gm.eh_proj);
        for (float* p : {gm.enorm, gm.hnorm, gm.shared_head_norm,
                         gm.Kc, gm.Vc, gm.Ic, gm.emb, gm.prev, gm.fused})
            if (p) cudaFree(p);
    }
    for (float* p : g->Kc) if (p) cudaFree(p);
    for (float* p : g->Vc) if (p) cudaFree(p);
    for (float* p : g->Ic) if (p) cudaFree(p);
    if (g->block_table) cudaFree(g->block_table);
    if (g->logits_d) cudaFree(g->logits_d);
    if (g->part_acc) cudaFree(g->part_acc);
    if (g->part_m) cudaFree(g->part_m);
    if (g->part_l) cudaFree(g->part_l);
    g->sc.free_all();
    delete g;
    gpu_state_ = nullptr;
}

}  // namespace glmserve
#endif  // GLMSERVE_CUDA
