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
`ignore_eos`, `stream`. Streaming uses SSE `chat.completion.chunk` frames ending
with `data: [DONE]`; the final chunk carries `usage`. Streaming is UTF-8-safe
(multi-byte characters are never split mid-sequence) and holds back a tail so a
`stop` string is never emitted.

```bash
./build/glmserve serve --model /path/to/GLM-5.2 \
  --port 8000 --host 0.0.0.0 --ctx 65536 --max-seqs 1 --name glm-5.2-local
```

`--ctx` sizes the per-sequence context (and thus the KV cache). `--max-seqs`
is the concurrency cap (V0 = 1 for coding agents). `--max-layers N` truncates
the stack for bring-up/debugging.

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
- **TP weight-sharding is wired into the forward**: head-sharded MLA attention
  (`q_b`/`kv_b` column-parallel, `o_proj` row-parallel) + column/row-parallel
  MLP/MoE FFNs, with an `all_reduce` rebuilding the residual stream each
  sub-layer. `tests/test_tp_forward.cpp` (`scripts/tp_forward_smoke.sbatch`) runs
  TP=2/PP=1 and matches the single-process forward to 1.5e-7 (job 125290).
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
is not loadable and exits nonzero.

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
