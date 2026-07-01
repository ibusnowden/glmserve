# GLM-5.2 tensor map

How HF safetensors tensor names map to glmserve modules
(`src/model_glm52.cpp::load`). The loader is tolerant: it tries the canonical
name, then known aliases, and treats qk-norm / shared-expert / bias tensors as
optional. Run `python tools/inspect_safetensors.py <model_dir>` to dump the real
names from a checkpoint and `tools/convert_hf_to_glmserve.py --check` to verify
coverage.

## Config (`config.json` → `include/config.hpp`)

| config.json key | field | GLM-5.2 |
|---|---|---|
| `num_hidden_layers` | layers | 78 |
| `hidden_size` | hidden | 6144 |
| `num_attention_heads` / `num_key_value_heads` | heads | 64 / 64 |
| `head_dim` | config head dim | 192 |
| `q_lora_rank` / `kv_lora_rank` | MLA latent ranks | 2048 / 512 |
| `qk_nope_head_dim` / `qk_rope_head_dim` | QK split | 192 / 64 |
| `qk_head_dim` / `v_head_dim` | attention dims | 256 / 256 |
| `intermediate_size` | dense MLP width | 12288 |
| `moe_intermediate_size` | per-expert width | 2048 |
| `n_routed_experts` / `num_experts_per_tok` | experts / top-k | 256 / 8 |
| `n_shared_experts` | shared experts | 1 |
| `first_k_dense_replace` | dense leading layers | 3 |
| `routed_scaling_factor` | MoE scale | 2.5 |
| `scoring_func` | router | sigmoid |
| `rope_theta`, `rms_norm_eps` | — | 8000000, 1e-5 |
| `index_topk` (DSA) | keys/query | 2048 |
| `num_nextn_predict_layers` | MTP heads | 1 |

## Global tensors

| name (+ aliases) | module |
|---|---|
| `model.embed_tokens.weight` | token embedding `[vocab, hidden]` |
| `model.norm.weight` | final RMSNorm |
| `lm_head.weight` (absent ⇒ tied to embeddings) | output projection |

## Per-layer `model.layers.{i}.`

| name | module | notes |
|---|---|---|
| `input_layernorm.weight` | pre-attn RMSNorm | |
| `post_attention_layernorm.weight` | pre-MLP RMSNorm | |
| `self_attn.q_a_proj.weight` | Q latent down projection | `[q_lora_rank, hidden]` |
| `self_attn.q_a_layernorm.weight` | Q latent RMSNorm | `[q_lora_rank]` |
| `self_attn.q_b_proj.weight` | Q head projection | `[heads * qk_head_dim, q_lora_rank]` |
| `self_attn.kv_a_proj_with_mqa.weight` | KV latent + shared RoPE projection | `[kv_lora_rank + qk_rope_head_dim, hidden]` |
| `self_attn.kv_a_layernorm.weight` | KV latent RMSNorm | `[kv_lora_rank]` |
| `self_attn.kv_b_proj.weight` | per-head K-nope/V projection | `[heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` |
| `self_attn.o_proj.weight` | output projection | |
| `self_attn.indexer.wq_b.weight` | DSA lightning indexer query projection | optional |
| `self_attn.indexer.wk.weight` | DSA lightning indexer key projection | optional |
| `self_attn.indexer.weights_proj.weight` | DSA index score projection | optional |
| `self_attn.indexer.k_norm.{weight,bias}` | DSA indexer key norm | optional |

The DSA indexer tensors are loaded when present. The CPU reference path and CUDA
path use them for learned top-k sparse selection on full-indexer layers once
`ctx > index_topk`; shared IndexShare layers reuse the most recent full-indexer
mask against their own K/V cache.

### Dense MLP (layers `i < first_k_dense_replace`)

`mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`.

### MoE MLP (layers `i ≥ first_k_dense_replace`)

| name (+ aliases) | module |
|---|---|
| `mlp.gate.weight` (or `mlp.router.weight`) | router `[n_experts, hidden]` |
| `mlp.gate.e_score_correction_bias` | aux-loss-free routing bias `[n_experts]` (optional) |
| `mlp.experts.{e}.{gate,up,down}_proj.weight` | routed expert `e` |
| `mlp.shared_experts.{gate,up,down}_proj.weight` (or `shared_expert`) | shared expert (optional) |

### MTP (optional, `num_nextn_predict_layers`)

GLM-5.2's multi-token-prediction head ships as extra layer(s) after the base
stack. For the public GLM-5.2 config (`num_hidden_layers=78`,
`num_nextn_predict_layers=1`) the extra predictor is stored as
`model.layers.78.*`.

| name | purpose |
|---|---|
| `model.layers.78.eh_proj.weight` | MTP embedding/hidden projection |
| `model.layers.78.enorm.weight` | MTP embedding norm |
| `model.layers.78.hnorm.weight` | MTP hidden norm |
| `model.layers.78.shared_head.norm.weight` | shared prediction-head norm |
| `model.layers.78.input_layernorm.weight` / `post_attention_layernorm.weight` | MTP block norms |
| `model.layers.78.self_attn.*` | MTP MLA attention + DSA indexer |
| `model.layers.78.mlp.*` | MTP MoE block |

`tools/convert_hf_to_glmserve.py --check` validates that these optional tensors
are indexed. The CPU runtime now has a draft-logits verification path
(`glmserve mtp`) that loads the MTP block, runs `eh_proj` + the extra MLA/MoE
block, and emits draft-token logits through `shared_head.norm` + `lm_head`.
`glmserve mtpcheck` adds a greedy speculative accept/reject correctness gate on
the tiny checkpoint. The normal generation loop can opt into CPU greedy MTP with
`mtp_draft_k` / `--mtp-draft-k`; GPU/distributed MTP remains a serving milestone.

## Quantized weights

The loader dequantizes scale-free dtypes (BF16/FP16/FP8) directly. For integer
quantization it pairs a weight tensor with companion scales:

- **INT8**: `X.weight` (`I8`) + `X.weight_scale` → `dequant_int8_to_f32`.
- **INT4 / W4A16**: `X.qweight` (`U8`, two signed nibbles/byte, zero-point 8) +
  `X.scales` (per-group) → `dequant_int4_to_f32` (CPU) / `gemm_w4a16` (GPU).

Exact group layout varies by checkpoint; adjust `load_linear` in
`src/model_glm52.cpp` and the repack in `convert_hf_to_glmserve.py` to match.
