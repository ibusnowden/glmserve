# GLM-5.2 Inference Conversation

**Topic:** Serving GLM-5.2 on 2 nodes of 8× RTX GPUs, connecting it to a Pi Code-style coding harness, and designing a custom C++/CUDA inference engine.

**Generated:** 2026-06-19  
**Primary hardware assumption:** 2 nodes × 8 NVIDIA RTX 6000 Ada 48GB GPUs = 16 GPUs / 768GB aggregate VRAM.

---

## 1. Questions From the Conversation

### Q1

> Can 2 nodes of 8 RTX serve the new GLM-5.2 open weight with vLLM and a harness like Pi Code?

### Q2

> Can we write a C/CUDA inference engine specifically for GLM-5.2 like vLLM?

---

## 2. Executive Summary

Yes, a 2-node RTX cluster can be useful for GLM-5.2, but the answer depends heavily on precision and GPU memory.

For **2 nodes × 8 RTX 6000 Ada 48GB**, full **BF16** GLM-5.2 is not realistic because the model is listed as **753B parameters** on Hugging Face. BF16 weights alone are roughly:

```text
753B parameters × 2 bytes ≈ 1.506 TB
```

Your assumed cluster has:

```text
16 GPUs × 48GB = 768GB aggregate VRAM
```

That is not enough for BF16 weights, before even considering KV cache, CUDA graphs, activations, MoE buffers, temporary workspaces, or vLLM runtime overhead.

Full **FP8** is also very tight on 16×48GB. FP8 weights are roughly:

```text
753B parameters × 1 byte ≈ 753GB
```

That nearly fills the entire 768GB pool, leaving almost no room for KV cache or runtime overhead. In practice, this is not a safe production configuration.

The realistic path on 16×48GB RTX 6000 Ada is:

```text
GLM-5.2 quantized to ~4-bit / NVFP4 / W4A16
FP16/BF16 activations
FP8 KV cache where supported
64K–256K context first
low concurrency
OpenAI-compatible endpoint
Pi Code / OpenCode / Cline harness on top
```

The custom C++/CUDA engine is possible and could be a serious systems project, but the right target is not “clone all of vLLM.” The right target is:

> A GLM-5.2-specialized inference engine optimized for 2-node RTX serving and coding-agent workloads.

---

## 3. Important GLM-5.2 Facts

GLM-5.2 is not a small model just because it is MoE. MoE reduces **active compute**, not total model storage. You still need to store all routed experts.

From the public model/config information:

| Property | Value |
|---|---:|
| Model size listed on HF | 753B params |
| Context window | 1,048,576 tokens |
| Architecture | `GlmMoeDsaForCausalLM` |
| Model type | `glm_moe_dsa` |
| Layers | 78 |
| Hidden size | 6144 |
| Routed experts | 256 |
| Experts per token | 8 |
| Shared experts | 1 |
| First dense MLP layers | 3 |
| Attention heads | 64 |
| KV heads | 64 |
| MTP speculative path | 5 draft tokens in vLLM recipe |
| License | MIT |

The key architectural parts for inference are:

```text
RMSNorm
GLM DSA / sparse attention
IndexShare-style index reuse
MoE routing
Top-8 expert dispatch
Shared expert path
MTP speculative decoding
Long-context KV cache
```

---

## 4. Can 2×8 RTX Serve GLM-5.2 With vLLM?

### Assumed cluster

```text
2 nodes
8 RTX 6000 Ada GPUs per node
48GB VRAM per GPU
16 GPUs total
768GB aggregate VRAM
```

### Feasibility matrix

| Target | Feasible? | Why |
|---|---:|---|
| GLM-5.2 BF16 | No | ~1.5TB just for weights |
| GLM-5.2 full FP8 | Very unlikely | ~753GB weights vs 768GB total VRAM, almost no room for KV cache/runtime |
| GLM-5.2 FP8 with tiny context | Maybe as an experiment | Fragile, likely OOM-prone |
| GLM-5.2 4-bit / NVFP4 | Likely yes | Much better memory fit |
| GLM-5.2 + Pi Code harness | Yes, once model server is up | Pi can call an OpenAI-compatible API |
| Full 1M context on 16×48GB | No / not realistic | KV cache dominates memory |
| 64K–256K coding-agent context | More realistic | Tune max model length and concurrency |

### Why FP8 is still too tight

Aggregate VRAM is not the same as usable memory.

Even if raw FP8 weights are near 753GB, the serving runtime also needs memory for:

```text
KV cache
CUDA graphs
temporary GEMM workspaces
MoE dispatch buffers
expert routing metadata
attention metadata
NCCL buffers
fragmentation/headroom
HTTP/server overhead
scheduler state
```

A safe serving deployment usually needs meaningful headroom. A configuration where weights alone consume almost the whole VRAM budget is not production-safe.

### If the GPUs are RTX PRO 6000 Blackwell 96GB

If “RTX 6000” means **RTX PRO 6000 Blackwell 96GB**, then:

```text
16 GPUs × 96GB = 1536GB aggregate VRAM
```

That is a completely different setup. FP8 becomes much more realistic, although full 1M context and high concurrency are still hard.

---

## 5. Recommended vLLM Topology

For 2 nodes × 8 GPUs, the standard vLLM layout is:

```text
tensor_parallel_size = 8
pipeline_parallel_size = 2
distributed_executor_backend = ray
```

That means:

```text
TP=8 inside each node
PP=2 across nodes
```

This is preferable to global TP=16 across nodes because cross-node tensor parallelism can make every layer depend on network all-reduce. Pipeline parallelism sends hidden states across the node boundary and can reduce cross-node synchronization pressure.

### Starting vLLM command for a memory-rich FP8 cluster

This is the kind of command to use on a cluster with enough memory, such as 96GB+ GPUs or H200/H20/B200-class systems:

```bash
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.88 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --served-model-name glm-5.2-fp8
```

For 16×48GB, this command is likely to OOM unless using a smaller/quantized checkpoint. For RTX 6000 Ada, the practical variant should target a 4-bit or NVFP4 checkpoint if vLLM supports the exact format.

### Practical RTX 6000 Ada target

```bash
vllm serve <glm-5.2-4bit-or-nvfp4-checkpoint> \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --served-model-name glm-5.2-local
```

Then scale upward:

```text
64K context → 128K → 256K
max_num_seqs 1 → 2 → 4
enable prefix caching
try FP8 KV cache
profile memory and tokens/sec
```

---

## 6. Pi Code / Harness Integration

The harness part is straightforward once the model server exposes an OpenAI-compatible endpoint.

Architecture:

```text
Pi Code / Cline / OpenCode / custom coding harness
        ↓
OpenAI-compatible API
        ↓
LiteLLM or direct vLLM endpoint
        ↓
vLLM GLM-5.2 server
        ↓
2-node GPU cluster
```

Example OpenAI-compatible client call:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="glm-5.2-local",
    messages=[
        {
            "role": "user",
            "content": "Inspect this repository and propose a minimal patch."
        }
    ],
    temperature=0.2,
    max_tokens=4096,
)

print(response.choices[0].message.content)
```

Example Pi-style model config concept:

```json
{
  "models": {
    "glm-5.2-local": {
      "provider": "openai-compatible",
      "baseURL": "http://localhost:8000/v1",
      "apiKey": "EMPTY",
      "model": "glm-5.2-local"
    }
  }
}
```

The exact Pi configuration path may differ by version, but the main concept is the same: point the harness at the local OpenAI-compatible endpoint.

---

## 7. Can We Write a C++/CUDA Inference Engine Specifically for GLM-5.2?

Yes.

But the correct goal is not:

> Build all of vLLM from scratch.

The correct goal is:

> Build a GLM-5.2-specialized serving engine with fewer abstractions and hard-coded assumptions for your cluster and coding-agent workload.

A good project name could be:

```text
glmserve

```

### Why a specialized engine can make sense

vLLM is general. A custom engine can assume:

```text
Only GLM-5.2
Only 2 nodes × 8 GPUs
Only low-concurrency coding-agent workloads
Known quantization format
Known max context window
Known expert placement
Known attention layout
Known tokenizer/chat template
```

That allows you to remove generic abstractions and tune aggressively.

---

## 8. Custom Engine Architecture

High-level serving stack:

```text
client / Pi Code / OpenAI SDK
        ↓
HTTP server: /v1/chat/completions
        ↓
scheduler
        ↓
tokenizer + chat template
        ↓
prefill engine
        ↓
decode engine
        ↓
CUDA kernels
        ↓
16-GPU distributed GLM-5.2 runtime
```

Core implementation stack:

```text
C++17 or C++20 control plane
CUDA kernels
cuBLASLt / CUTLASS for GEMM
NCCL for multi-GPU communication
safetensors loader
paged KV cache
MoE router
HTTP server
SSE streaming
OpenAI-compatible API
```

---

## 9. Core Engine Components

### 9.1 Weight loader

The loader needs to read HF safetensors shards and place weights across GPUs.

Required support:

```text
safetensors parsing
tensor name mapping
weight sharding metadata
FP8 tensors
4-bit/NVFP4 tensors
per-block scales
BF16 fallback tensors
GPU placement map
CPU staging buffers
```

For 16×48GB RTX, the serious version should prioritize:

```text
W4A16 / NVFP4 / INT4-style weights
FP16/BF16 activations
FP8 KV cache where possible
```

### 9.2 Tokenizer and chat template

Options:

```text
Use Hugging Face tokenizer through a small Python sidecar
Use tokenizers C++ bindings
Export tokenizer to a lightweight C++ format
Use a pre-tokenized benchmark path for early correctness
```

For V0, it is acceptable to use a Python tokenizer bridge. For production, move tokenizer and chat-template handling into the C++ server.

### 9.3 GLM-5.2 block implementation

Each block needs:

```text
input RMSNorm
Q/K/V projection path
GLM DSA/sparse attention
RoPE
output projection
post-attention RMSNorm
dense MLP or sparse MoE MLP
residual connection
```

The first 3 MLP layers are dense. The remaining layers are sparse/MoE.

### 9.4 Attention path

This is one of the hardest parts.

Implementation order:

```text
V0: dense attention correctness path
V1: sparse indexer path
V2: fused sparse attention kernel
V3: paged sparse attention with prefix cache
V4: IndexShare reuse across layers
```

Do not begin with the perfect GLM DSA kernel. First make logits match.

### 9.5 MoE engine

GLM-5.2 uses:

```text
256 routed experts
8 selected experts per token
1 shared expert
sigmoid scoring
top-k routing
routed scaling factor 2.5
```

The MoE path needs:

```text
gate projection
top-k select
token → expert assignment
expert dispatch
expert GEMM
shared expert GEMM
combine outputs
residual
```

Naive MoE will be slow. The optimized path needs fused dispatch, grouped GEMM, and careful expert placement.

### 9.6 Paged KV cache

A minimal C++ structure:

```cpp
struct KVBlock {
    void* k;
    void* v;
    int block_id;
    int refcount;
};

struct Sequence {
    int64_t request_id;
    std::vector<int> kv_blocks;
    int prompt_len;
    int generated_len;
};
```

Paged KV cache is necessary for long-context serving. Without it, memory fragmentation and prompt length variation become painful.

### 9.7 Continuous batching

Build gradually:

```text
V0: batch=1
V1: static batch
V2: continuous decode batching
V3: prefill/decode split
V4: prefix cache
V5: chunked prefill
```

For coding-agent workloads, batch size 1–4 may be enough if latency and reliability are good.

### 9.8 Sampling

Start with:

```text
greedy decoding
temperature
top-p
top-k
repetition penalty
stop sequences
EOS handling
```

Then add:

```text
structured output
tool-call parsing
JSON-constrained decoding
MTP speculative decoding
```

### 9.9 MTP speculative decoding

GLM-5.2’s vLLM recipe uses 5 speculative tokens.

Implementation order:

```text
normal greedy decode
normal sampled decode
MTP head forward
draft 5 tokens
verify tokens
accept/reject
fallback on low acceptance
```

Do this after base generation is correct.

---

## 10. Distributed Strategy for 2 Nodes

Recommended first distributed layout:

```text
Node 0: layers 0–38
Node 1: layers 39–77
Tensor parallel = 8 within each node
Pipeline parallel = 2 across nodes
```

This avoids doing cross-node TP all-reduce at every layer.

### Why not global TP=16 first?

Global TP=16 means many operations require cross-node communication. On RTX workstation/server clusters, inter-node bandwidth is usually far weaker than intra-node GPU communication. Pipeline parallelism across nodes is simpler and may be more stable for low-concurrency coding-agent serving.

### Expert placement

Initial plan:

```text
Keep each layer’s experts local to the node holding that layer.
Shard experts across the 8 GPUs inside the node.
Avoid cross-node expert dispatch.
Use routing only within the local layer group.
```

Later, test expert parallelism across nodes only if networking is strong enough.

---

## 11. Minimal Repository Layout

```text
glmserve/
  CMakeLists.txt

  include/
    config.hpp
    tensor.hpp
    safetensors.hpp
    scheduler.hpp
    kv_cache.hpp
    model.hpp
    sampler.hpp
    nccl_comm.hpp
    http_server.hpp

  src/
    main.cpp
    server.cpp
    tokenizer.cpp
    safetensors.cpp
    scheduler.cpp
    kv_cache.cpp
    model_glm52.cpp
    sampler.cpp
    nccl_comm.cpp
    http_server.cpp

  cuda/
    rmsnorm.cu
    rope.cu
    fp8_gemm.cu
    int4_gemm.cu
    attention_dense.cu
    attention_dsa.cu
    moe_router.cu
    moe_dispatch.cu
    moe_expert.cu
    sampling.cu

  tools/
    convert_hf_to_glmserve.py
    inspect_safetensors.py
    compare_transformers.py
    export_tokenizer.py

  tests/
    test_rmsnorm.cpp
    test_rope.cpp
    test_moe_router.cpp
    test_attention.cpp
    test_logits_match.py

  bench/
    bench_prefill.cpp
    bench_decode.cpp
    bench_moe.cpp
    bench_kv_cache.cpp

  docs/
    architecture.md
    glm52_tensor_map.md
    serving.md
    pi_integration.md
```

---

## 12. Build Roadmap

### Phase 0 — Reference and inspection

Goal:

```text
Understand exact GLM-5.2 tensor names and shapes.
Create a tensor map.
Load config.json.
List safetensors metadata.
```

Deliverables:

```text
tools/inspect_safetensors.py
docs/glm52_tensor_map.md
minimal config parser
```

### Phase 1 — CPU/GPU correctness path

Goal:

```text
Run a small number of layers and compare outputs against Transformers/vLLM.
```

Implement:

```text
RMSNorm
RoPE
linear layers
dense attention fallback
dense MLP
MoE router
logits
greedy sampling
```

Success criterion:

```text
same greedy tokens for short fixed prompts
logit error within expected precision tolerance
```

### Phase 2 — Single GPU toy path

Goal:

```text
Run a tiny or partial GLM-5.2 layer stack on one GPU.
```

This is for debugging kernels, not full serving.

### Phase 3 — Single-node 8-GPU engine

Goal:

```text
Run quantized GLM-5.2 on one node with TP=8.
```

Target:

```text
batch=1
32K–64K context
OpenAI-compatible endpoint
greedy + sampling
basic KV cache
```

### Phase 4 — Paged KV and serving

Goal:

```text
Make it useful for coding-agent traffic.
```

Implement:

```text
paged KV cache
SSE streaming
request scheduler
max-token accounting
timeout/cancel handling
prefix cache
```

### Phase 5 — 2-node distributed

Goal:

```text
Run the model across 2 nodes.
```

Implement:

```text
NCCL setup
pipeline send/recv
layer partitioning
per-node KV cache
TP=8 inside node
PP=2 across nodes
```

### Phase 6 — Performance kernels

Optimize:

```text
RMSNorm fusion
RoPE in-place
FP8/INT4 GEMM
MoE dispatch
grouped expert GEMM
paged attention
sparse DSA attention
sampling kernel
```

### Phase 7 — GLM-specific features

Add:

```text
IndexShare reuse
MTP speculative decoding
tool-call parser
reasoning mode control
JSON-constrained decoding
long-context optimizations
```

---

## 13. Suggested Milestones

### M0: Tensor map and loader

```text
Can read config.json.
Can list all safetensors.
Can map tensor names to engine modules.
Can load selected tensors into CPU memory.
```

### M1: One-layer correctness

```text
Run one GLM-5.2 block.
Compare hidden states with reference.
```

### M2: Full forward correctness on small prompt

```text
Run all layers in a slow path.
Generate 10 greedy tokens.
Match reference tokens.
```

### M3: CUDA decode path

```text
Generate tokens on GPU.
Support KV cache.
Batch size 1.
```

### M4: Single-node serving

```text
Serve /v1/chat/completions.
Stream tokens.
Use 8 GPUs.
Run coding harness against it.
```

### M5: Two-node serving

```text
Ray/vLLM-style deployment no longer needed for this engine.
Use custom NCCL/distributed runtime.
TP=8, PP=2.
```

### M6: Competitive specialization

```text
MoE optimized.
Paged sparse attention.
Prefix cache.
MTP.
Coding-agent tuned.
```

---

## 14. What Not To Build First

Avoid these in V0:

```text
full 1M context
high concurrency
perfect MTP
full vLLM-compatible plugin system
multi-tenant scheduling
tool-use parser
JSON constrained decoding
disaggregated prefill/decode
global TP=16 across nodes
```

These features are valuable, but they should come after correctness and basic serving.

---

## 15. Main Technical Risks

### 15.1 Memory

The model is enormous. Even FP8 is tight on 16×48GB. A practical RTX engine needs quantization.

### 15.2 GLM DSA attention

Plain FlashAttention is not enough. GLM-5.2 uses DSA/sparse attention and IndexShare-style reuse.

### 15.3 MoE dispatch

MoE can become bottlenecked by token dispatch, memory movement, and expert imbalance.

### 15.4 Inter-node communication

Two-node RTX clusters usually do not have the same communication profile as HGX H100/H200/B200 machines.

### 15.5 Correctness debugging

A custom engine requires careful reference comparisons at every stage.

---

## 16. Recommended Development Strategy

Use existing frameworks as references, not enemies.

Reference stack:

```text
Transformers for correctness
vLLM for serving behavior
SGLang for alternative serving behavior
KTransformers/llama.cpp-style quantization ideas
CUTLASS/cuBLASLt for GEMMs
NCCL for communication
```

Best first implementation path:

```text
1. Load config and tensors.
2. Implement slow correctness path.
3. Compare against Transformers/vLLM.
4. Add CUDA kernels one by one.
5. Add paged KV.
6. Add OpenAI-compatible server.
7. Add single-node TP.
8. Add two-node PP.
9. Add GLM-specific sparse attention.
10. Add MTP.
```

---

## 17. Practical Serving Recommendation

For your 2×8 RTX 6000 Ada 48GB cluster, do not start with full BF16 or raw FP8.

Start with:

```text
GLM-5.2 4-bit / NVFP4 checkpoint
64K context
batch size 1
max_num_seqs 1
OpenAI-compatible server
Pi Code harness
```

Then gradually push:

```text
64K → 128K → 256K context
batch 1 → 2 → 4
FP8 KV cache
prefix caching
MTP
custom MoE kernels
```

The best project framing is:

> A 2-node RTX-optimized GLM-5.2 inference engine for coding agents.

That is a compelling systems project because it combines:

```text
LLM serving
CUDA kernels
distributed inference
MoE runtime
long-context memory management
agent harness integration
```

---

## 18. Condensed Answer

### Can 2×8 RTX serve GLM-5.2?

With **BF16**: no.  
With **raw FP8 on 16×48GB**: probably not safely.  
With **4-bit/NVFP4 quantization**: likely yes for lower context/concurrency.  
With **RTX PRO 6000 Blackwell 96GB**: FP8 is much more realistic.  
With **Pi Code harness**: yes, once an OpenAI-compatible endpoint exists.

### Can we write a C++/CUDA GLM-5.2 engine?

Yes. Build it as a specialized engine, not a full vLLM clone.

The first useful target:

```text
Quantized GLM-5.2
64K–256K context
low concurrency
2-node TP=8/PP=2 topology
OpenAI-compatible API
Pi Code integration
```

---

## 19. Source Links

1. [GLM-5.2 Hugging Face model card](https://huggingface.co/zai-org/GLM-5.2)
2. [GLM-5.2 config.json on Hugging Face](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
3. [GLM-5.2 vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.2)
4. [vLLM parallelism and scaling docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
5. [vLLM distributed serving docs](https://docs.vllm.ai/en/v0.8.1/serving/distributed_serving.html)
6. [NVIDIA RTX 6000 Ada product page](https://www.nvidia.com/en-us/products/workstations/rtx-6000/)
7. [Pi Coding Agent website](https://pi.dev/)
8. [Pi Agent Harness GitHub repository](https://github.com/earendil-works/pi)

---

## 20. Appendix: Minimal OpenAI-Compatible Endpoint Shape

The local server should implement at least:

```text
POST /v1/chat/completions
GET  /v1/models
```

Minimal request body:

```json
{
  "model": "glm-5.2-local",
  "messages": [
    {"role": "user", "content": "Fix the failing test in this repository."}
  ],
  "temperature": 0.2,
  "max_tokens": 4096,
  "stream": true
}
```

Minimal streaming response should use SSE-compatible chunks:

```text
data: {"choices":[{"delta":{"content":"..."}}]}

data: [DONE]
```

This makes the engine usable from Pi Code, OpenAI SDK-compatible tools, Cline, OpenCode, LiteLLM, and other coding-agent harnesses.

