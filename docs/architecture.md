# glmserve architecture

A GLM-5.2-specialized serving engine. Generic abstractions are deliberately
absent: the model, quantization formats, attention layout, and cluster topology
are all assumed to be GLM-5.2 on 2×8 RTX 6000 Ada.

## Data flow

```
HTTP (src/http_server.cpp)  ──►  Engine (src/server.cpp)
                                   ├─ Tokenizer (src/tokenizer.cpp): chat template -> ids
                                   ├─ Scheduler (src/scheduler.cpp): admission, cancel
                                   ├─ GLM52Model (src/model_glm52.cpp): forward
                                   │     └─ KVCache (src/kv_cache.cpp): paged K/V
                                   └─ Sampler (src/sampler.cpp): greedy/temp/top-k/top-p
```

## The GLM-5.2 block (`src/model_glm52.cpp`)

Each of the 78 layers, applied to a `[n_tokens, hidden=6144]` activation buffer:

1. `input_layernorm` (RMSNorm)
2. **Attention**: Q/K/V projections → optional per-head qk-RMSNorm → partial RoPE
   (first `rotary_dim = head_dim·partial_rotary_factor` dims) → causal attention
   (GQA-aware) reading/writing the paged KV cache → output projection. Residual add.
3. `post_attention_layernorm` (RMSNorm)
4. **MLP**:
   - layers `< first_k_dense_replace` (3): dense `down(silu(gate(x))·up(x))`
   - layers `≥ 3`: **MoE** — `sigmoid(router·x)` scores, add aux-loss-free
     `e_score_correction_bias` for *selection only*, top-8, normalize selected
     gates, scale by `routed_scaling_factor=2.5`, sum experts + always-on shared
     expert. Residual add.
5. Final `model.norm` (RMSNorm) → `lm_head` → logits.

### Numerical conventions (committed to, mirrored in numpy)

- Linear: `y = x @ W^T (+b)`, weights stored row-major `[out, in]` (HF).
- RMSNorm: `x · rsqrt(mean(x²) + eps) · w`, sum-of-squares accumulated in double.
- RoPE: NeoX rotate-half on the first `rotary_dim` dims; `theta = rope_theta`.
- MoE gate weight = the *original* sigmoid score (not bias-corrected); bias only
  reorders the top-k selection (DeepSeek-V3 / GLM aux-loss-free routing).

`tools/make_tiny_checkpoint.py` reimplements exactly this in numpy;
`tests/test_logits_match.py` asserts the C++ forward matches to ~1e-6.

## Two backends

- **CPU reference** (always built): float32, thread-parallel matmul. The source
  of truth for correctness and the path that runs without a GPU.
- **CUDA** (`GLMSERVE_CUDA`, `cuda/*.cu`): launch-ready kernels for sm_89 —
  rmsnorm, per-head rmsnorm, partial RoPE, cuBLAS GEMM, W4A16 + FP8 dequant GEMM,
  flash-style dense + split-K decode + DSA-windowed paged attention, MoE
  router/dispatch/expert, argmax + softmax. The kernel API is `cuda/kernels.cuh`.
  These are wired into a device-resident forward (`src/model_gpu.cpp`):
  `upload_to_gpu()` makes the weights + a persistent per-layer KV cache resident,
  `forward_gpu_prefill()` runs a one-shot prompt prefill, and
  `forward_gpu_decode()` is the **incremental** single-token step — it appends
  one position's K/V to the cache and attends over the whole sequence (O(ctx) per
  token, *not* a re-prefill). Both share one block-stack runner and a persistent
  scratch arena, so decode does no per-token `cudaMalloc`.

### Decode attention (flash-decoding / split-K)

At decode `n_query==1`, the dense kernel would launch only `n_heads` blocks and
walk every key serially — starving the GPU and making latency grow linearly with
context. `attention_decode_paged` instead splits the key range into `S` chunks
processed by `n_heads·S` blocks (pass 1: partial online softmax per chunk; pass
2: merge the `S` partials per head). Decode latency then stays ~flat with
context. On an RTX 6000 Ada (tiny-checkpoint benchmark, `glmserve bench`) decode
holds ~1.9–2.4K tok/s from ctx 192 → 8K (vs. 1740 → 231 before the split), and
prefill runs 85K–330K tok/s. `glmserve gencheck` asserts the incremental path is
token-identical to the re-prefill reference (max logit diff ~1e-7).

### DSA status

The GPU forward now routes contexts larger than `index_topk` through
`attention_dsa_paged`. That kernel is a correctness-defined V1 sparse baseline:
it attends to the most recent `index_topk` keys and degrades to exact dense
attention when `ctx <= index_topk`.

The model loader now recognizes optional GLM lightning-indexer weights
(`self_attn.indexer.{wq_b,wk,weights_proj,k_norm.*}`) and stores them in each
layer when present. Those weights are not yet used to select sparse keys in the
CPU or GPU forward path; replacing the recent-window baseline with learned
indexer top-k remains the next DSA milestone.

## Memory: the paged KV cache (`src/kv_cache.cpp`)

Fixed-size blocks (`block_size` tokens); each sequence owns a block table
(logical→physical). Per (layer, physical block): `K,V = [block_size, kv_heads,
head_dim]`. This lets long, variable-length prompts coexist without
fragmentation. Reference stores float32 host buffers; the GPU path mirrors the
same block-table logic with device buffers and an optional FP8 KV dtype.

## Distributed (`src/nccl_comm.cpp`)

Target: TP=8 within a node, PP=2 across nodes (node 0 = layers 0–38, node 1 =
39–77). `partition_layers()` computes each stage's range.

The CUDA build now has a real NCCL communicator:

- launcher/env rank discovery (`GLMSERVE_*`, `RANK/WORLD_SIZE`, Slurm)
- filesystem or direct-hex NCCL unique-id rendezvous
- world communicator initialization
- TP subgroup creation via `ncclCommSplit`
- TP all-reduce and world barrier
- PP send/recv entry points for hidden-state handoff (host buffers are staged
  through a persistent device bounce buffer, so the CPU reference forward can
  drive the handoff directly; device-pointer callers stay zero-copy)

**Both TP and PP are wired into the CPU-reference forward.** `GLM52Model::set_distributed`
attaches the communicator; `load()` then shards weights for this rank.

*Pipeline parallel:* `load()` keeps only this stage's contiguous layer range
(`partition_layers()`) — embedding only on the first stage, `model.norm`+`lm_head`
only on the last — and `forward()` embeds-or-`pipeline_recv_prev()` → runs the
owned layers → `pipeline_send_next()` to the next stage (a non-final stage
produces no logits; the last stage runs final norm + lm_head). The per-stage KV
cache is sized to the local layer count, so layers and KV share one local index.

*Tensor parallel:* `load()` slices each block (Megatron layout) — attention is
split across heads (`q_b`/`kv_b` column-parallel, `o_proj` row-parallel), MLP/MoE
FFNs across the intermediate dim (`gate`/`up` column-parallel, `down`
row-parallel); `q_a`/`kv_a`, the router, norms, embeddings and `lm_head` stay
replicated. `forward()` attends only this rank's head slice, and `run_layer()`
`all_reduce_sum`s the row-parallel `o_proj` and MLP/MoE outputs to rebuild the
replicated residual stream. The KV cache holds only the rank's local heads
(`local_kv_heads()`). PP send/recv and TP all-reduce both auto-stage **host**
buffers (the CPU forward's activations) through a device bounce buffer.

`scripts/nccl_tp_smoke.sbatch` validates the 2-rank TP all-reduce + row-parallel
linear primitive; `scripts/nccl_pp_smoke.sbatch` the PP device-buffer send/recv.
The full-forward gates run the model itself over the tiny checkpoint on 2× RTX
6000 Ada and compare to a single-process forward:
- `tests/test_pp_forward.cpp` (`pp_forward_smoke.sbatch`): TP=1/PP=2, logits
  **bit-identical** (`max_diff=0`, job 125285).
- `tests/test_tp_forward.cpp` (`tp_forward_smoke.sbatch`): TP=2/PP=1, logits
  match to **1.5e-7** (job 125290; all-reduce reorders the float sums, so this
  is ~rounding rather than bit-exact).

What remains is carrying the same TP/PP sharding onto the device-resident GPU
forward path (`model_gpu.cpp`), and the multi-process serving driver that
launches the ranks and routes tokens/logits through the HTTP front-end.

## Why specialized beats generic here

Fixed model ⇒ no plugin/registry indirection. Fixed topology ⇒ layer partition
and expert placement are compile-time. Low concurrency (coding agents, batch
1–4) ⇒ scheduler stays simple and latency-first. Known quant format ⇒ the GEMM
dequant is inlined, not dispatched.
