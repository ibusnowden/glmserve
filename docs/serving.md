# Serving GLM-5.2 with glmserve

## OpenAI-compatible API

| route | purpose |
|---|---|
| `POST /v1/chat/completions` | chat (stream + non-stream); applies the GLM chat template |
| `POST /v1/completions` | raw prompt completion |
| `GET /v1/models` | lists the served model id |
| `GET /health` | liveness |

Request fields honored: `messages`/`prompt`, `temperature` (0 ⇒ greedy),
`top_p`, `top_k`, `max_tokens`/`max_completion_tokens`, `stop` (string or list),
`repetition_penalty`, `frequency_penalty`, `presence_penalty`, `seed`,
`ignore_eos`, `stream`. `mtp_draft_k` enables the CPU greedy MTP path when
`temperature=0`, no penalties are active, and MTP weights are present; GPU and
distributed MTP are still pending. Streaming uses SSE `chat.completion.chunk`
frames ending with `data: [DONE]`; the final chunk carries `usage`. Streaming is
UTF-8-safe (multi-byte characters are never split mid-sequence) and holds back a
tail so a `stop` string is never emitted.

```bash
./build/glmserve serve --model /path/to/GLM-5.2 \
  --port 8000 --host 0.0.0.0 --ctx 65536 --max-seqs 1 --name glm-5.2-local
```

`--ctx` sizes the per-sequence context (and thus the KV cache). `--max-seqs`
is the concurrency cap (V0 = 1 for coding agents). `--max-layers N` truncates
the stack for bring-up/debugging.

### V0 serving limits

The HTTP API is usable for trusted local agent traffic, but it is not hardened
as a public endpoint yet. Generation holds a single engine mutex for the full
request, so requests run serially even if `--max-seqs` is raised. The JSON layer
is dependency-free and intentionally small; fuzz or replace it before exposing
the server to arbitrary internet clients.

## Memory reality on 2×8 RTX 6000 Ada (16×48GB = 768GB)

GLM-5.2 is **753B params** (MoE shrinks *active* compute, not storage).

| precision | weights | verdict |
|---|---|---|
| BF16 | ~1.5 TB | does not fit |
| FP8 | ~753 GB | fills VRAM, no room for KV/runtime → unsafe |
| **W4A16 / NVFP4** | ~376 GB | **the realistic path**; leaves room for KV + 64–256K ctx |

So on RTX 6000 Ada: quantize to 4-bit, FP16/BF16 activations, FP8 KV cache where
possible, start at 64K context / `max-seqs 1`, then scale 64K→128K→256K and
concurrency 1→2→4. (On RTX PRO 6000 Blackwell 96GB = 1536GB aggregate, FP8 is
viable.)

## 2-node topology (Phase 5)

```
TP = 8   within each node (intra-node NVLink/PCIe all-reduce)
PP = 2   across nodes      (node 0: layers 0–38, node 1: 39–77)
```

Pipeline-parallel across the node boundary avoids per-layer cross-node
all-reduce (inter-node bandwidth ≪ intra-node). `partition_layers()` in
`src/nccl_comm.cpp` computes the per-stage layer range; experts stay local to
the node that owns their layer (no cross-node expert dispatch initially).

Current distributed status:

- NCCL world + TP subgroup initialization is implemented.
- `all_reduce_sum()` is active for TP groups and has a 2-rank RTX smoke test in
  `scripts/nccl_tp_smoke.sbatch`.
- `tests/test_tp_linear.cpp` verifies a row-parallel linear projection can be
  reconstructed with TP all-reduce on the same 2-rank RTX path.
- `tests/test_nccl_pipeline.cpp` verifies `pipeline_send_next()` /
  `pipeline_recv_prev()` with a device buffer under `TP=1, PP=2`.
- **PP layer-stage execution is wired into the forward**: each stage loads only
  its layer range and the forward hands the hidden state to the next stage over
  the PP link. `tests/test_pp_forward.cpp` (`scripts/pp_forward_smoke.sbatch`)
  runs the tiny checkpoint split TP=1/PP=2 and confirms the last stage's logits
  are bit-identical (`max_diff=0`) to a single-process full forward (job 125285).
- **TP weight-sharding is wired into the CPU-reference forward**: head-sharded MLA attention
  (`q_b`/`kv_b` column-parallel, `o_proj` row-parallel) + column/row-parallel
  MLP/MoE FFNs, with an `all_reduce` rebuilding the residual stream each
  sub-layer. `tests/test_tp_forward.cpp` (`scripts/tp_forward_smoke.sbatch`) runs
  TP=2/PP=1 and matches the single-process forward to 1.5e-7 (job 125290);
  this is a tolerance gate because all-reduce changes floating-point order.
- Still pending: carrying the same TP/PP sharding onto the device-resident GPU
  forward path, and the multi-process serving driver (launch ranks, route
  tokens/logits through the HTTP front-end).

## Quantizing a checkpoint

```bash
python tools/inspect_safetensors.py /path/to/GLM-5.2          # see names/dtypes
python tools/convert_hf_to_glmserve.py --model /path/to/GLM-5.2 --check
python tools/convert_hf_to_glmserve.py --model /path/to/GLM-5.2 \
       --quantize-int4 /path/to/GLM-5.2-w4a16 --group 128
```

`--check` validates both the tensor names in the index and the presence of every
referenced safetensors shard. A directory with only config/tokenizer/index files
is not loadable and exits nonzero. Optional DSA indexer and MTP tensor coverage
is reported separately from core loadability.

If the directory has `model.safetensors.index.json` but not the referenced
shards, fetch them explicitly:

```bash
GLMSERVE_REAL_MODEL=/path/to/GLM-5.2 bash scripts/fetch_glm52_shards.sh --dry-run
GLMSERVE_REAL_MODEL=/path/to/GLM-5.2 bash scripts/fetch_glm52_shards.sh --probe 1 --max-files 1
HF_TOKEN=... GLMSERVE_REAL_MODEL=/path/to/GLM-5.2 \
  bash scripts/fetch_glm52_shards.sh --require-free-gib 1800 --verify-present --jobs 4
```

`fetch_glm52_shards.sh` is intended for a login or data-transfer node with
internet access. It supports shard ranges (`--first-shard`, `--last-shard`) so a
large fetch can be resumed or split, `--jobs` enables parallel downloads, and
`--probe` performs HEAD checks without downloading shard data. `--verify-present`
HEAD-checks existing selected shards and repairs size mismatches before the W4
conversion runs.

If the fetch fails with `Disk quota exceeded` while `df` still reports free
space, stop increasing `--jobs`/`--max-files`: the limiting factor is a project
or user quota on writes. Keep the completed shards, remove any `.part` files,
and resume only after moving to a shared path with enough quota or increasing
the quota. The 16-rank W4 gate needs the full source plus W4 output visible from
both nodes.

To plan a PP-stage-specific checkpoint instead of one monolithic W4 directory:

```bash
python tools/plan_stage_shards.py --model /path/to/GLM-5.2 --pp-size 2
python tools/plan_stage_shards.py --model /path/to/GLM-5.2 --pp-size 2 --stage 0 --show-files
python tools/plan_stage_shards.py --model /path/to/GLM-5.2 --pp-size 2 --stage 1 --show-files
```

The distributed load smoke uses prefix-filtered safetensors loading, so each PP
stage maps only shards containing its embeddings/final head/MTP block and owned
layer range. `scripts/w4_real_16x.sbatch` accepts
`GLMSERVE_W4_MODEL_STAGE0`/`GLMSERVE_W4_MODEL_STAGE1` for stage-specific W4
directories; otherwise all ranks use `GLMSERVE_W4_MODEL`.

For `PP>1`, `test_dist_load` validates distributed rank setup and selective
checkpoint loading by default (`GLMSERVE_DIST_LOAD_UPLOAD=0`). GPU upload still
requires the device-resident PP handoff path; set `GLMSERVE_DIST_LOAD_UPLOAD=1`
only for TP-only topologies or while developing that PP upload path.

Repeatable W4 gates:

```bash
sbatch scripts/w4_gpu_check.sbatch

GLMSERVE_REAL_MODEL=/path/to/full/GLM-5.2 \
GLMSERVE_W4_MODEL=/path/to/GLM-5.2-w4a16 \
sbatch scripts/w4_real_16x.sbatch

GLMSERVE_FETCH_REPO=zai-org/GLM-5.2 \
GLMSERVE_REAL_MODEL=/path/to/GLM-5.2 \
GLMSERVE_W4_MODEL=/path/to/GLM-5.2-w4a16 \
sbatch scripts/w4_real_16x.sbatch

GLMSERVE_W4_MODEL_STAGE0=/path/to/GLM-5.2-w4a16-stage0 \
GLMSERVE_W4_MODEL_STAGE1=/path/to/GLM-5.2-w4a16-stage1 \
sbatch scripts/w4_real_16x.sbatch
```

`w4_gpu_check` proves the converter, CPU W4 dequant path, GPU resident qweight
path, and incremental decode on a quantized tiny checkpoint. `w4_real_16x`
checks the full checkpoint, builds/validates the W4 repack if needed, then runs a
TP=8/PP=2 distributed load/upload smoke across 16 RTX ranks. It is a fit/load
gate; combined TP/PP device serving is still a separate runtime milestone. The
W4 repack writes each shard through a `.part` file and atomically replaces the
final shard on success; rerunning the converter skips already-valid output
shards and repairs corrupt/incomplete ones.

## Tokenizer

V0 ships a byte-level fallback (always correct round-trips) and a best-effort
`tokenizer.json` BPE loader. For exact GLM-5.2 tokenization before the C++ BPE is
fully validated, bridge through HF:

```bash
python tools/export_tokenizer.py --model /path/to/GLM-5.2 --text "fix the test"
# -> prints token ids; feed to:  glmserve dump --model ... --tokens "..."
```

## Verifying correctness against a reference

```bash
python tests/test_logits_match.py                       # tiny-model self-check (~1e-6)
python tools/compare_transformers.py --model /path/to/GLM-5.2 --tokens "1 2 3 4"
```
