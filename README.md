# glmserve

A C++/CUDA inference engine **specialized for GLM-5.2** — built to the design in
[`glm52_inference_conversation.md`](glm52_inference_conversation.md). Not a vLLM
clone: it hard-codes GLM-5.2's architecture (MoE + DSA sparse attention + MTP)
and targets a 2-node × 8 RTX 6000 Ada cluster with a coding-agent (Pi Code /
Cline / OpenCode) workload over an OpenAI-compatible API.

```
client / Pi Code / OpenAI SDK
        │  POST /v1/chat/completions (SSE)
        ▼
HTTP server ──► scheduler ──► tokenizer + chat template
                                   │
                              prefill / decode
                                   │
                    GLM-5.2 forward (RMSNorm · qk-norm · partial RoPE ·
                    dense/DSA attention · dense MLP · sigmoid top-k MoE +
                    shared expert) over a paged KV cache
                                   │
                    CPU reference path  ──or──  CUDA kernels (sm_89)
                                   │
                         16-GPU TP=8 / PP=2 runtime
```

## Status (what runs today)

This repository is a **working V0** plus the GPU acceleration scaffold:

| Capability | State |
|---|---|
| config.json parse, safetensors mmap loader (single + sharded) | ✅ done |
| Full GLM-5.2 forward — **CPU float32 reference** | ✅ correct (matches numpy ref to 6e-7) |
| Paged KV cache, sampler (greedy/temp/top-k/top-p/penalties), scheduler | ✅ done |
| OpenAI-compatible server: `/v1/chat/completions` (+SSE), `/v1/completions`, `/v1/models`, `/health` | ✅ V0; generation is serialized and parser hardening is pending |
| Tokenizer: byte-level fallback + best-effort `tokenizer.json` BPE + python bridge | ✅ V0 |
| CUDA kernels (rmsnorm, rope, gemm, w4a16/fp8 gemm, dense+DSA-window attention, MoE router/dispatch/expert, sampling) | ✅ compile & link sm_89; RTX kernel tests pass |
| GPU forward wiring (device residency, incremental decode, DSA-window sparse path) | ✅ works on RTX tiny-checkpoint gate |
| GLM DSA indexer weights | ✅ loader/checker recognize tensors; ✅ CPU/CUDA learned top-k active on full-indexer layers; ✅ shared IndexShare mask reuse |
| GLM MTP weights | ✅ checker recognizes `model.layers.78.*`; ✅ CPU MTP draft-logits + greedy accept/reject gates; ✅ opt-in CPU greedy generation path; 🚧 GPU/distributed MTP not wired |
| W4A16 real-weight path | ✅ converter + CPU dequant + GPU resident qweight tiny gate; ✅ shard fetch/probe helper; ✅ 16× RTX TP=8/PP=2 distributed load gate script; 🚧 needs full shard set for real-weight run |
| NCCL distributed runtime | ✅ world + TP subgroup init/all-reduce, row-parallel linear reconstruction, PP send/recv (host buffers auto-staged) — all smoke tested on 2× RTX |
| **PP layer-stage execution** (per-stage layer sharding + hidden-state handoff in the forward) | ✅ TP=1/PP=2 forward over the tiny checkpoint is **bit-identical** (`max_diff=0`) to single-process (2× RTX, job 125285) |
| **TP weight-sharding in the CPU-reference forward** (head-sharded MLA, gate/up column- + down row-parallel MLP/MoE, all-reduce) | ✅ TP=2/PP=1 forward matches single-process to **1.5e-7** (2× RTX, job 125290) |
| GPU-path TP/PP, multi-process serving driver, MTP | 🚧 roadmap |

The CPU path lets the entire stack be exercised end-to-end **without a GPU and
without the 753B weights** (a tiny same-architecture checkpoint stands in). The
CUDA kernels are real, run on RTX 6000 Ada, and are wired into both the
device-resident forward and — on the CPU reference path — the TP/PP distributed
forward.

### Tiny checkpoint vs. real weights — where the line is

The 753B GLM-5.2 weights are **not** required for the current work and aren't on
disk. A tiny same-architecture checkpoint (`tools/make_tiny_checkpoint.py`) plus a
single-process / numpy forward as the correctness oracle covers everything that is
about *logic*:

- **Proven on the tiny checkpoint** (weight *values* don't matter for these):
  forward correctness (CPU & GPU, ~1e-6 vs numpy), incremental decode, and the
  distributed forward — PP layer-stage execution is bit-identical to
  single-process (`max_diff=0`) and TP weight-sharding matches to ~1e-7. Sharding
  correctness is independent of the weights, so the tiny model is the *right* tool:
  the tolerance-based comparison runs on 2 GPUs in under a minute. Device-resident
  GPU MoE uses atomic accumulation for top-k expert outputs, so GPU MoE checks are
  numerical/tolerance gates, not bit-exact reductions against the CPU order.
- **Needs the real weights** (only when moving from "is it correct" to "does it
  serve GLM-5.2 well"): actual generation quality, honest throughput/latency for
  the real 78-layer 753B MoE (the tiny model's numbers are engine efficiency, not
  model behavior), exercising the W4A16/int4 quant path on real tensors, and
  confirming the 16×48 GB memory fit.

## Build

```bash
source scripts/env.sh        # CUDA 12.8 toolkit (.cudaenv) + sm_89 default
make                         # CPU reference engine (no GPU needed)
make GPU=1                   # full CUDA build (links cuBLAS + NCCL)
make tests   && make run-tests
make bench
```

`make` works on a login node with no GPU. `make GPU=1` needs the toolkit on
`PATH` (provided by `scripts/env.sh`).

## Quickstart (no real weights required)

```bash
# 1) build a tiny GLM-5.2-architecture checkpoint + numpy reference
python tools/make_tiny_checkpoint.py --out /tmp/glm52_tiny

# 2) correctness: engine logits == numpy reference (max-abs diff ~1e-6)
python tests/test_logits_match.py --bin build/glmserve

# 3) one-shot generation
./build/glmserve generate --model /tmp/glm52_tiny --prompt "hello" --max-tokens 16

# 4) serve the OpenAI-compatible API
./build/glmserve serve --model /tmp/glm52_tiny --port 8000 --ctx 4096
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-local","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

## Serving real GLM-5.2

Point `--model` at an HF GLM-5.2 checkpoint directory (config.json + safetensors).
The loader reads BF16/FP16/FP8 natively; for 16×48GB use a W4A16/int4 repack
(`tools/convert_hf_to_glmserve.py`). Validate tensor coverage first:

```bash
python tools/inspect_safetensors.py /path/to/GLM-5.2
python tools/convert_hf_to_glmserve.py --model /path/to/GLM-5.2 --check
```

Both tools now check that every shard named by `model.safetensors.index.json`
exists on disk; an index-only cache is reported as incomplete and exits nonzero.
If only the index/sidecar files are present, use
`tools/fetch_safetensors_shards.py --model /path/to/GLM-5.2 --repo zai-org/GLM-5.2`
to fetch the missing safetensors shards before quantizing.

See [`docs/serving.md`](docs/serving.md) for the 2-node TP=8/PP=2 topology and
[`docs/pi_integration.md`](docs/pi_integration.md) to wire it to Pi Code.

### Real 3-bit llama.cpp GGUF reference

The sibling `/project/inniang/inference` stack currently serves the real
GLM-5.2 Unsloth split GGUF checkpoint:

```text
/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL/
```

Validate that weight set from this repo with:

```bash
make
bash scripts/check_glm52_gguf.sh
```

This is the required target for a native 3-bit `glmserve` loader. Today it is a
GGUF load gate: the `glmserve` binary parses the split GGUF shards, validates the
GLM-5.2 tensor inventory, builds the GGUF-to-glmserve module layout, mmaps the
payload files, touches every quantized tensor range, loads stable quant tensor
views into `GLM52Model` via `glmserve load-gguf`, and dequantizes a real block
from each observed GGML type including `IQ3_XXS`. Generation still
executes safetensors or glmserve W4A16 repacks while llama.cpp executes the split
GGUF tensors.

## Layout

```
include/  config, tensor, safetensors, kv_cache, model, sampler, scheduler,
          http_server, nccl_comm, json (dependency-free), common
src/      the C++ control plane + the CPU reference forward
cuda/     the sm_89 kernels (gated by GLMSERVE_CUDA)
tools/    inspect / convert / compare-vs-transformers / export-tokenizer /
          make_tiny_checkpoint
tests/    kernel unit tests (CUDA) + test_logits_match.py (e2e gate)
bench/    prefill / decode / moe / kv-cache micro-benchmarks
docs/     architecture · glm52_tensor_map · serving · pi_integration
```

## Roadmap (from the design doc §12)

- ✅ **Phase 0–1** reference + correctness path (logits match)
- ✅ **serving** OpenAI API, paged KV, sampler, scheduler
- ✅ **kernels** all CUDA kernels compile for sm_89
- ✅ **Phase 3a** single-GPU device-resident forward + incremental decode validated on RTX 6000 Ada
- 🚧 **Phase 3b** single-node TP=8 GPU forward (weight sharding, streams, collectives)
- ✅ **Phase 5a** NCCL communicator bring-up: env/Slurm ranks, TP subgroups, 2-rank RTX all-reduce + TP-linear + PP send/recv smoke
- ✅ **Phase 5b-PP** PP layer-stage execution in the forward: per-stage layer sharding + NCCL hidden-state handoff; TP=1/PP=2 forward bit-matches single-process on 2× RTX
- ✅ **Phase 5b-TP** TP weight-sharding in the forward: head-sharded MLA + column/row-parallel MLP/MoE with in-forward all-reduce; TP=2/PP=1 forward matches single-process to 1.5e-7 on 2× RTX
- 🚧 **Phase 5c** the CPU-reference TP/PP path on the device-resident GPU forward + a multi-process serving driver (launch ranks, route tokens/logits through the HTTP front-end, remove the generation-wide V0 server lock)
- 🚧 **Phase 6–7** fused MoE grouped-GEMM, DSA performance optimization, MTP speculative decoding, prefix cache
