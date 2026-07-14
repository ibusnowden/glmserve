# glmserve Architecture Overview

## System Design

glmserve is a C++/CUDA inference engine specialized for GLM-5.2 (753B parameter MoE model).
It supports both CPU reference execution and GPU acceleration with tensor/pipeline parallelism.

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP Server (OpenAI API)               │
│  POST /v1/chat/completions, /v1/completions, /v1/models     │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                        Engine                               │
│  - Model (GLM52Model): weights + forward                    │
│  - Tokenizer: BPE + fallback                                │
│  - KVCache: paged key-value cache                           │
│  - Sampler: greedy/top-k/top-p/penalties                    │
│  - Scheduler: request management (V0: serialized)           │
│  - Communicator: NCCL for TP/PP                            │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
┌───────────────▼───────────────┐   ┌───────────▼───────────────┐
│    CPU Reference Path         │   │     CUDA Kernel Path      │
│  - Full float32 implementation│   │  - Device-resident weights│
│  - Correctness validation     │   │  - sm_89 kernels          │
│  - No GPU required            │   │  - cuBLASLt GEMM          │
└───────────────────────────────┘   └───────────────────────────┘
```

## GLM-5.2 Model Architecture

### Transformer Block Structure

Each of the 78 layers follows this pattern:

```
Input → RMSNorm → MLA Attention → Residual → RMSNorm → {Dense MLP | MoE} → Residual
```

#### MLA Attention (DeepSeek-style Latent Projections)

```
Q: hidden → q_a (q_lora_rank) → RMSNorm → q_b → [H, qk_head_dim]
KV: hidden → [c_kv (kv_lora_rank) | k_pe (qk_rope_head_dim)]
     c_kv → RMSNorm → kv_b → [H, qk_nope_head_dim + v_head_dim]
```

- Q latent rank: 2048
- KV latent rank: 512
- Query/key head dim: 256 (192 nope + 64 rope)
- Value head dim: 256

#### DSA Sparse Attention

For contexts > 2048 tokens, a "lightning indexer" selects top-2048 keys per query:
- 64 index heads, 128-dim index vectors
- Learned top-k selection with ReLU-weighted head sum
- Fallback: recent window of last 2048 keys

#### MoE (Mixture of Experts)

- 256 routed experts + 1 shared expert
- 8 experts per token (top-k by sigmoid score)
- Sigmoid scoring with optional score-correction bias
- Routed scaling factor: 2.5
- First 3 layers: dense MLP (intermediate_size=12288)
- Remaining 75 layers: MoE (moe_intermediate_size=2048 per expert)

## Distributed Execution

### Tensor Parallelism (TP)

Megatron-style column/row parallelism:
- **Column-parallel**: q_b_proj, kv_b_proj, gate_proj, up_proj
  - Each rank owns a contiguous slice of output rows
- **Row-parallel**: o_proj, down_proj
  - Each rank owns a contiguous slice of input columns
  - Requires all-reduce to sum partial outputs
- Heads are contiguous: rank r owns heads [r*(H/TP), (r+1)*(H/TP))

### Pipeline Parallelism (PP)

- Layers partitioned into contiguous stages
- First stage: embedding lookup
- Last stage: final norm + lm_head → logits
- Hidden states pipelined between stages via NCCL

### Typical Deployment

- TP=8, PP=2 over 16 RTX 6000 Ada GPUs
- Each rank: 1/8 weight shard, replicated embed/lm_head
- Rank 0: HTTP server + sampling
- All ranks: lockstep forward execution

## Memory Layout

### KV Cache (Paged)

```
K: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
V: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
I: [num_layers, num_blocks, block_size, index_head_dim]  (DSA indexer keys)
```

- Block size: typically 16 tokens
- Single-block paged: block_table maps logical→physical, value always 0
- Latent KV mode (default): fp16 latent cache [max_ctx, kvlat+rope] per layer
  - 14x smaller than per-head K/V
  - Absorbed MQA during decode

### Weight Residency

#### CPU Path
- Safetensors: dequantized to float32 in memory
- GGUF: mmap'd quantized payloads, dequantized on access

#### GPU Path
- All weights uploaded to device VRAM
- Persistent KV cache: O(max_ctx) per token (not O(ctx²))
- Persistent scratch arena: no per-token cudaMalloc
- GGUF quant: dequantized on-the-fly in kernels

## CUDA Kernel Organization

### GEMM Kernels (cuda/qgemm.cu)

**MMVQ-style**: One warp per output row
- Row walked in 256-element groups
- Each lane decodes 8-element fragment directly into registers
- Warp shuffle-reduce for partial sums
- Prefill: token tiling (T=8) for weight reuse

**Int8 MMQ path**: For compute-bound kernels
- Activations quantized to int8 per 32-element block
- Weight fragments as integers + affine scale
- 8-element dot = 2 dp4a instructions
- ~3x faster than fp32 for MoE experts

### Attention Kernels

**Dense attention** (cuda/attention_dense.cu):
- Paged KV cache access
- Causal masking

**DSA sparse attention** (cuda/attention_dsa.cu):
- Two-stage: scoring kernel + radix select
- GEMM scoring path with cuBLAS (CUBLAS_PEDANTIC_MATH for correctness)
- Radix select: exact top-k with index tiebreak for CPU parity

**Absorbed MLA** (cuda/mla_absorb.cu):
- Decode: attend over latent cache directly
- Pass 2 merges + expands through W_UV in one fused kernel
- Flash-decoding: split-K for parallelism at n_query==1

### MoE Kernels (cuda/moe_expert.cu)

**Token-major path**:
- One block per (token, slot)
- Threads split moe_inter dimension
- Uses atomicAdd for output accumulation

**Expert-major path** (prefill with dispatch):
- Tokens grouped by expert for weight reuse
- 8-token weight-fragment reuse
- ~8x fewer quant decodes at prefill

## Performance Optimizations

### 1. Absorbed MLA + Latent KV
- fp16 latent cache: [max_ctx, kvlat+rope] per layer
- Decode attention: no per-head K/V materialization
- 14x memory savings, ~2x faster decode

### 2. Int8 MMQ
- Quantize activations to int8, keep weight fragments as integers
- dp4a instructions instead of fp32 FMAs
- ~60 GB/s effective → ~800 GB/s peak utilization

### 3. Bulk Expert Upload
- Pack all experts contiguously: [E, moe_inter, hidden]
- Parallel CPU repack + single bulk H2D transfer
- 768 transfers → 3 transfers per MoE layer

### 4. Upload Source Management
- Prefetch thread: touch-ahead of mmap'd payloads
- Eviction with lag: drop source pages after upload
- Bounded resident window: prevents kernel eviction

### 5. Stage Profiling
- cudaEvent marks at each forward stage
- Per-stage GPU timeline measurement
- Identify bottlenecks (e.g., DSA scoring at long context)

## Numerical Correctness

### FP32 vs TF32
- cuBLAS default: TF32 (10-bit mantissa) for fp32 products
- DSA scoring: TF32 rotates scores enough to flip top-k selections
- Solution: CUBLAS_PEDANTIC_MATH for DSA GEMM kernels
- Only summation order differs (ULP-level), absorbed by radix tiebreak

### MoE Accumulation
- CPU: deterministic order (top-k by score, then expert index)
- GPU: atomicAdd accumulation (non-deterministic order)
- Speculative decode: requires exact parity
- Solution: fixed-slot-order reduce for GPU MoE (moe_dpart buffer)

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| GLMSERVE_CUDA | Enable CUDA build | - |
| GLMSERVE_PROF | Enable stage profiling | off |
| GLMSERVE_LATENT_KV | Use latent KV cache | on |
| GLMSERVE_EVICT_LAG | Source eviction lag (layers) | -1 (off) |
| GLMSERVE_EVICT_ALL | Evict all source pages on start | off |
| GLMSERVE_UPLOAD_PREFETCH | Prefetch layers ahead | 4 |
| GLMSERVE_PINNED_UPLOAD | Use pinned host memory | off |
| GLMSERVE_DSA_GEMM | Use GEMM for DSA scoring | on |
| GLMSERVE_WIDE_SUFFIX | Full-width suffix chunks | on |
| GLMSERVE_PREFIX_REUSE | Multi-turn prefix reuse | on |
| GLMSERVE_HOST_EMBED | Keep embed table host-resident (TP) | off |
| GLMSERVE_MMQ | Enable int8 MMQ activations | on |
| GLMSERVE_AR_ROWWISE | Row-wise all-reduce for <64 rows | auto |

### Runtime Flags

| Flag | Purpose |
|------|---------|
| --gpu | Run forward on CUDA path |
| --ctx N | Max context length |
| --block-size N | KV block size |
| --max-layers N | Truncate layer stack (testing) |
| --max-seqs N | Max concurrent sequences (V0: 1) |

## Testing Strategy

### CPU Reference
- Full forward pass in float32
- Matches numpy reference to 6e-7
- Used as correctness oracle

### GPU Validation
- `check_decode`: incremental vs re-prefill parity
- `check_chunk_parity`: speculative vs serial decode
- `check_mtp_speculative`: MTP draft vs target
- `check_turn_reuse`: multi-turn prefix reuse

### Distributed Validation
- TP=2/PP=1: matches single-process to 1.5e-7
- TP=1/PP=2: bit-identical to single-process
- All-reduce order matches CPU reference

## Limitations

1. **V0 Server**: Serialized generation, no true batching
2. **MTP + PP**: MTP requires PP=1 (embed + lm_head co-resident)
3. **DSA GEMM**: Requires CUBLAS_PEDANTIC_MATH (slower)
4. **Quantized Experts**: Only GGUF or int4, not both
5. **Block Table**: Single-entry (no true paging fragmentation handling)

## File Organization

```
include/    - Headers: config, model, kv_cache, sampler, server, etc.
src/        - C++ control plane + CPU reference forward
  model_glm52.cpp  - CPU forward, weight loading, MoE, attention
  model_gpu.cpp    - GPU forward, weight upload, MTP (2252 lines)
cuda/       - CUDA kernels
  qgemm.cu         - GGUF quant GEMM (1663 lines, largest)
  attention_dsa.cu - DSA sparse attention
  mla_absorb.cu    - Absorbed MLA for decode
  moe_expert.cu    - MoE expert FFN kernels
tests/      - Unit tests + correctness gates
bench/      - Micro-benchmarks (prefill, decode, MoE, KV cache)
tools/      - HF→GGUF conversion, checkpoint generation
scripts/    - Build, profiling, distributed launch
```
