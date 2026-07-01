#!/usr/bin/env python3
"""Generate a tiny GLM-5.2-architecture checkpoint + numpy reference forward.

Mirrors the REAL GLM-5.2 architecture (verified from zai-org/GLM-5.2 config.json):
DeepSeek-style MLA (q_a/q_b + kv_a/kv_b latent projections, decoupled interleaved
RoPE), sigmoid top-k MoE with aux-loss-free bias + shared expert, dense leading
layers. Same numerical conventions as src/model_glm52.cpp, at miniature dims, so
`glmserve dump` logits can be compared to this numpy reference (the real 753B
weights aren't on disk). tests/test_logits_match.py is the gate.

Usage:  python tools/make_tiny_checkpoint.py --out /tmp/glm52_tiny
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file

# tiny MLA config (field names match include/config.hpp / HF config.json)
TINY = dict(
    vocab_size=320,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=12,                  # == qk_nope_head_dim (HF convention)
    # MLA
    q_lora_rank=32,
    kv_lora_rank=24,
    qk_nope_head_dim=12,
    qk_rope_head_dim=4,
    qk_head_dim=16,               # nope + rope
    v_head_dim=16,
    attention_bias=False,
    rope_interleave=True,
    # MLP / MoE
    intermediate_size=128,
    moe_intermediate_size=48,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    first_k_dense_replace=2,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=2.5,
    norm_topk_prob=True,
    scoring_func="sigmoid",
    # norm / rope
    rms_norm_eps=1e-5,
    rope_parameters=dict(rope_theta=10000.0, rope_type="default"),
    max_position_embeddings=4096,
    # DSA indexer: intentionally tiny top-k so the default 8-token test prompt
    # exercises learned sparse selection instead of degenerating to dense.
    use_dsa=True,
    index_n_heads=2,
    index_head_dim=8,
    index_topk=3,
    indexer_rope_interleave=False,
    indexer_types=["full", "shared", "shared", "full"],
    num_nextn_predict_layers=1,
    model_type="glm_moe_dsa",
    architectures=["GlmMoeDsaForCausalLM"],
    dtype="float32",
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    tie_word_embeddings=False,
)


def rn(rng, *shape, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def rmsnorm(x, w, eps):
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return ((x * (1.0 / np.sqrt(var + eps))).astype(np.float32)) * w


def layernorm(x, w, b, eps=1e-6):
    mean = np.mean(x.astype(np.float64), axis=-1, keepdims=True)
    var = np.mean((x.astype(np.float64) - mean) ** 2, axis=-1, keepdims=True)
    return (((x - mean) * (1.0 / np.sqrt(var + eps))).astype(np.float32)) * w + b


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def linear(x, W, b=None):
    y = x @ W.T
    return (y + b if b is not None else y).astype(np.float32)


def rope_inplace(v, pos, theta, interleave):
    # v: 1-D, even length. Returns rotated copy.
    v = v.copy()
    dim = v.shape[0]
    half = dim // 2
    i = np.arange(half)
    freq = 1.0 / (theta ** ((2.0 * i) / dim))
    ang = pos * freq
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    if interleave:
        a, b = 2 * i, 2 * i + 1
    else:
        a, b = i, i + half
    x0, x1 = v[a].copy(), v[b].copy()
    v[a] = x0 * c - x1 * s
    v[b] = x1 * c + x0 * s
    return v


def dsa_layer_type(cfg, layer):
    types = cfg.get("indexer_types") or []
    return types[layer] if layer < len(types) else "full"


def mla_attention(cfg, W, p, h, layer_idx, shared_dsa_indices=None):
    n, _ = h.shape
    H = cfg["num_attention_heads"]
    nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
    qk, vd = cfg["qk_head_dim"], cfg["v_head_dim"]
    kvlat = cfg["kv_lora_rank"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_parameters"]["rope_theta"]
    il = cfg["rope_interleave"]
    scale = 1.0 / np.sqrt(qk)
    index_topk = cfg.get("index_topk", 2048)

    qa = rmsnorm(linear(h, W[p + "self_attn.q_a_proj.weight"]),
                 W[p + "self_attn.q_a_layernorm.weight"], eps)
    q = linear(qa, W[p + "self_attn.q_b_proj.weight"]).reshape(n, H, qk)

    kva = linear(h, W[p + "self_attn.kv_a_proj_with_mqa.weight"])
    c_kv = rmsnorm(kva[:, :kvlat], W[p + "self_attn.kv_a_layernorm.weight"], eps)
    k_pe = kva[:, kvlat:kvlat + rope]
    kvb = linear(c_kv, W[p + "self_attn.kv_b_proj.weight"]).reshape(n, H, nope + vd)

    # build per-head K (qk) / V (vd) with rope
    K = np.zeros((n, H, qk), np.float32)
    V = np.zeros((n, H, vd), np.float32)
    Q = q.copy()
    for t in range(n):
        kpe_t = rope_inplace(k_pe[t], t, theta, il)
        for hh in range(H):
            Q[t, hh, nope:qk] = rope_inplace(q[t, hh, nope:qk], t, theta, il)
            K[t, hh, :nope] = kvb[t, hh, :nope]
            K[t, hh, nope:qk] = kpe_t
            V[t, hh] = kvb[t, hh, nope:nope + vd]

    selected = None
    layer_type = dsa_layer_type(cfg, layer_idx)
    if cfg.get("use_dsa", True) and n > index_topk and layer_type == "full":
        iH, idim = cfg["index_n_heads"], cfg["index_head_dim"]
        iq = linear(qa, W[p + "self_attn.indexer.wq_b.weight"]).reshape(n, iH, idim)
        ik = layernorm(linear(h, W[p + "self_attn.indexer.wk.weight"]),
                       W[p + "self_attn.indexer.k_norm.weight"],
                       W[p + "self_attn.indexer.k_norm.bias"])
        iw = linear(h, W[p + "self_attn.indexer.weights_proj.weight"]) / np.sqrt(iH)
        for t in range(n):
            for ih in range(iH):
                iq[t, ih, :rope] = rope_inplace(iq[t, ih, :rope], t, theta, False)
            ik[t, :rope] = rope_inplace(ik[t, :rope], t, theta, False)

        selected = []
        iscale = 1.0 / np.sqrt(idim)
        for t in range(n):
            sc = np.zeros(t + 1, np.float32)
            for j in range(t + 1):
                per_head = np.maximum(0.0, (iq[t] @ ik[j]) * iscale)
                sc[j] = np.sum(iw[t] * per_head)
            k = min(index_topk, t + 1)
            selected.append(np.sort(np.argsort(-sc, kind="stable")[:k]))
    elif cfg.get("use_dsa", True) and n > index_topk and shared_dsa_indices is not None:
        selected = shared_dsa_indices

    out = np.zeros((n, H * vd), np.float32)
    for t in range(n):
        keys = selected[t] if selected is not None else range(t + 1)
        for hh in range(H):
            sc = np.array([np.dot(Q[t, hh], K[j, hh]) * scale for j in keys], np.float32)
            sc -= sc.max()
            e = np.exp(sc); e /= e.sum()
            acc = np.zeros(vd, np.float32)
            for jj, j in enumerate(keys):
                acc += e[jj] * V[j, hh]
            out[t, hh * vd:(hh + 1) * vd] = acc
    return linear(out, W[p + "self_attn.o_proj.weight"]), selected


def moe(cfg, W, p, x):
    n, Hd = x.shape
    topk = cfg["num_experts_per_tok"]
    scale = cfg["routed_scaling_factor"]
    logits = linear(x, W[p + "mlp.gate.weight"])
    bias = W.get(p + "mlp.gate.e_score_correction_bias")
    out = np.zeros((n, Hd), np.float32)
    for t in range(n):
        score = sigmoid(logits[t])
        choose = score + (bias if bias is not None else 0.0)
        idx = np.argsort(-choose, kind="stable")[:topk]
        w = score[idx].astype(np.float32)
        if cfg["norm_topk_prob"] and w.sum() > 0:
            w = w / w.sum()
        w = w * scale
        xt = x[t]
        acc = np.zeros(Hd, np.float32)
        for kk, e in enumerate(idx):
            ep = p + f"mlp.experts.{e}."
            g = silu(linear(xt[None], W[ep + "gate_proj.weight"]))[0]
            u = linear(xt[None], W[ep + "up_proj.weight"])[0]
            acc += w[kk] * linear((g * u)[None], W[ep + "down_proj.weight"])[0]
        sp = p + "mlp.shared_experts."
        g = silu(linear(xt[None], W[sp + "gate_proj.weight"]))[0]
        u = linear(xt[None], W[sp + "up_proj.weight"])[0]
        acc += linear((g * u)[None], W[sp + "down_proj.weight"])[0]
        out[t] = acc
    return out


def transformer_hidden(cfg, W, tokens):
    eps = cfg["rms_norm_eps"]
    hidden = np.stack([W["model.embed_tokens.weight"][t] for t in tokens]).astype(np.float32)
    shared_dsa_indices = None
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        normed = rmsnorm(hidden, W[p + "input_layernorm.weight"], eps)
        attn, selected = mla_attention(cfg, W, p, normed, L, shared_dsa_indices)
        if selected is not None and dsa_layer_type(cfg, L) == "full":
            shared_dsa_indices = selected
        hidden = hidden + attn
        normed = rmsnorm(hidden, W[p + "post_attention_layernorm.weight"], eps)
        if L < cfg["first_k_dense_replace"]:
            g = silu(linear(normed, W[p + "mlp.gate_proj.weight"]))
            u = linear(normed, W[p + "mlp.up_proj.weight"])
            hidden = hidden + linear(g * u, W[p + "mlp.down_proj.weight"])
        else:
            hidden = hidden + moe(cfg, W, p, normed)
    return hidden


def reference_forward(cfg, W, tokens):
    eps = cfg["rms_norm_eps"]
    hidden = transformer_hidden(cfg, W, tokens)
    hidden = rmsnorm(hidden, W["model.norm.weight"], eps)
    return linear(hidden, W["lm_head.weight"])


def mtp_reference_logits(cfg, W, tokens, draft_tokens):
    eps = cfg["rms_norm_eps"]
    p = f"model.layers.{cfg['num_hidden_layers']}."
    prev = transformer_hidden(cfg, W, tokens)[-1].astype(np.float32)
    mtp_inputs = []
    logits = []
    for tok in draft_tokens:
        e = W["model.embed_tokens.weight"][tok]
        en = rmsnorm(e[None], W[p + "enorm.weight"], eps)[0]
        hn = rmsnorm(prev[None], W[p + "hnorm.weight"], eps)[0]
        inp = linear(np.concatenate([en, hn])[None], W[p + "eh_proj.weight"])[0]
        mtp_inputs.append(inp)
        h = np.stack(mtp_inputs).astype(np.float32)
        normed = rmsnorm(h, W[p + "input_layernorm.weight"], eps)
        attn, _ = mla_attention(cfg, W, p, normed, cfg["num_hidden_layers"], None)
        h = h + attn
        normed = rmsnorm(h, W[p + "post_attention_layernorm.weight"], eps)
        h = h + moe(cfg, W, p, normed)
        prev = h[-1].astype(np.float32)
        head = rmsnorm(prev[None], W[p + "shared_head.norm.weight"], eps)
        logits.append(linear(head, W["lm_head.weight"])[0])
    return np.stack(logits).astype(np.float32)


def build_weights(cfg, seed=1234):
    rng = np.random.default_rng(seed)
    H = cfg["hidden_size"]
    nH = cfg["num_attention_heads"]
    qk, vd = cfg["qk_head_dim"], cfg["v_head_dim"]
    nope = cfg["qk_nope_head_dim"]
    qlat, kvlat, rope = cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
    inter, moei, V = cfg["intermediate_size"], cfg["moe_intermediate_size"], cfg["vocab_size"]
    iH, idim = cfg["index_n_heads"], cfg["index_head_dim"]
    ones = lambda d: (np.ones(d, np.float32) + rn(rng, d, scale=0.01))
    W = {
        "model.embed_tokens.weight": rn(rng, V, H),
        "model.norm.weight": ones(H),
        "lm_head.weight": rn(rng, V, H),
    }
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        W[p + "input_layernorm.weight"] = ones(H)
        W[p + "post_attention_layernorm.weight"] = ones(H)
        # MLA
        W[p + "self_attn.q_a_proj.weight"] = rn(rng, qlat, H)
        W[p + "self_attn.q_a_layernorm.weight"] = ones(qlat)
        W[p + "self_attn.q_b_proj.weight"] = rn(rng, nH * qk, qlat)
        W[p + "self_attn.kv_a_proj_with_mqa.weight"] = rn(rng, kvlat + rope, H)
        W[p + "self_attn.kv_a_layernorm.weight"] = ones(kvlat)
        W[p + "self_attn.kv_b_proj.weight"] = rn(rng, nH * (nope + vd), kvlat)
        W[p + "self_attn.o_proj.weight"] = rn(rng, H, nH * vd)
        if dsa_layer_type(cfg, L) == "full":
            W[p + "self_attn.indexer.wq_b.weight"] = rn(rng, iH * idim, qlat)
            W[p + "self_attn.indexer.wk.weight"] = rn(rng, idim, H)
            W[p + "self_attn.indexer.weights_proj.weight"] = rn(rng, iH, H)
            W[p + "self_attn.indexer.k_norm.weight"] = ones(idim)
            W[p + "self_attn.indexer.k_norm.bias"] = rn(rng, idim, scale=0.01)
        if L < cfg["first_k_dense_replace"]:
            W[p + "mlp.gate_proj.weight"] = rn(rng, inter, H)
            W[p + "mlp.up_proj.weight"] = rn(rng, inter, H)
            W[p + "mlp.down_proj.weight"] = rn(rng, H, inter)
        else:
            W[p + "mlp.gate.weight"] = rn(rng, cfg["n_routed_experts"], H)
            W[p + "mlp.gate.e_score_correction_bias"] = rn(rng, cfg["n_routed_experts"], scale=0.05)
            for e in range(cfg["n_routed_experts"]):
                ep = p + f"mlp.experts.{e}."
                W[ep + "gate_proj.weight"] = rn(rng, moei, H)
                W[ep + "up_proj.weight"] = rn(rng, moei, H)
                W[ep + "down_proj.weight"] = rn(rng, H, moei)
            sp = p + "mlp.shared_experts."
            W[sp + "gate_proj.weight"] = rn(rng, moei, H)
            W[sp + "up_proj.weight"] = rn(rng, moei, H)
            W[sp + "down_proj.weight"] = rn(rng, H, moei)
    if cfg.get("num_nextn_predict_layers", 0) > 0:
        L = cfg["num_hidden_layers"]
        p = f"model.layers.{L}."
        W[p + "eh_proj.weight"] = rn(rng, H, 2 * H)
        W[p + "enorm.weight"] = ones(H)
        W[p + "hnorm.weight"] = ones(H)
        W[p + "shared_head.norm.weight"] = ones(H)
        W[p + "input_layernorm.weight"] = ones(H)
        W[p + "post_attention_layernorm.weight"] = ones(H)
        W[p + "self_attn.q_a_proj.weight"] = rn(rng, qlat, H)
        W[p + "self_attn.q_a_layernorm.weight"] = ones(qlat)
        W[p + "self_attn.q_b_proj.weight"] = rn(rng, nH * qk, qlat)
        W[p + "self_attn.kv_a_proj_with_mqa.weight"] = rn(rng, kvlat + rope, H)
        W[p + "self_attn.kv_a_layernorm.weight"] = ones(kvlat)
        W[p + "self_attn.kv_b_proj.weight"] = rn(rng, nH * (nope + vd), kvlat)
        W[p + "self_attn.o_proj.weight"] = rn(rng, H, nH * vd)
        W[p + "self_attn.indexer.wq_b.weight"] = rn(rng, iH * idim, qlat)
        W[p + "self_attn.indexer.wk.weight"] = rn(rng, idim, H)
        W[p + "self_attn.indexer.weights_proj.weight"] = rn(rng, iH, H)
        W[p + "self_attn.indexer.k_norm.weight"] = ones(idim)
        W[p + "self_attn.indexer.k_norm.bias"] = rn(rng, idim, scale=0.01)
        W[p + "mlp.gate.weight"] = rn(rng, cfg["n_routed_experts"], H)
        W[p + "mlp.gate.e_score_correction_bias"] = rn(rng, cfg["n_routed_experts"], scale=0.05)
        for e in range(cfg["n_routed_experts"]):
            ep = p + f"mlp.experts.{e}."
            W[ep + "gate_proj.weight"] = rn(rng, moei, H)
            W[ep + "up_proj.weight"] = rn(rng, moei, H)
            W[ep + "down_proj.weight"] = rn(rng, H, moei)
        sp = p + "mlp.shared_experts."
        W[sp + "gate_proj.weight"] = rn(rng, moei, H)
        W[sp + "up_proj.weight"] = rn(rng, moei, H)
        W[sp + "down_proj.weight"] = rn(rng, H, moei)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default="3 1 4 1 5 9 2 6")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = dict(TINY)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    W = build_weights(cfg, args.seed)
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()},
              os.path.join(args.out, "model.safetensors"))

    prompt = [int(x) for x in args.prompt.split()]
    last = reference_forward(cfg, W, prompt)[-1]
    json.dump(dict(prompt=prompt, vocab=cfg["vocab_size"], argmax=int(np.argmax(last)),
                   logits=[float(x) for x in last]),
              open(os.path.join(args.out, "reference_logits.json"), "w"))

    draft = [2, 7, 1]
    mtp = mtp_reference_logits(cfg, W, prompt, draft)
    json.dump(dict(context=prompt, draft=draft, vocab=cfg["vocab_size"],
                   argmax=[int(np.argmax(row)) for row in mtp],
                   logits=[float(x) for x in mtp.reshape(-1)]),
              open(os.path.join(args.out, "reference_mtp_logits.json"), "w"))

    print(f"[make_tiny] MLA checkpoint -> {args.out}: {len(W)} tensors, "
          f"layers={cfg['num_hidden_layers']}, prompt={prompt}, argmax={int(np.argmax(last))}")


if __name__ == "__main__":
    main()
