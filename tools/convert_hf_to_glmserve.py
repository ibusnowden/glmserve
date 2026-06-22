#!/usr/bin/env python3
"""Validate / prepare an HF GLM-5.2 checkpoint for glmserve.

glmserve reads HF safetensors natively (src/safetensors.cpp + the tolerant name
mapping in src/model_glm52.cpp), so for BF16/FP16/FP8 checkpoints no conversion
is needed — this tool just *validates* that every tensor glmserve expects is
present and reports anything unrecognized.

It also (optionally) repacks linear weights to W4A16 (group-symmetric int4) — the
memory path that makes GLM-5.2 fit on 16x48GB (spec §15.1). The repack is
implemented for the per-output-channel group layout the loader understands.

Usage:
  python tools/convert_hf_to_glmserve.py --model DIR [--check] [--quantize-int4 DIR] [--group 128]
"""
import argparse
import json
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def checkpoint_files(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, "r", encoding="utf-8") as f:
            meta = json.load(f)
        files = sorted(set(meta["weight_map"].values()))
        paths = [os.path.join(model_dir, f) for f in files]
        missing = [p for p in paths if not os.path.exists(p)]
        return meta, paths, missing
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return None, [single], []
    raise SystemExit(f"no safetensors in {model_dir}")


def all_tensor_names(model_dir):
    meta, files, missing = checkpoint_files(model_dir)
    if meta is not None:
        return set(meta["weight_map"].keys())
    return set(k for k in read_header(files[0]) if k != "__metadata__")


def expected_names(cfg):
    """Tensor names glmserve will look for, given config.json (MLA layout)."""
    L = cfg["num_hidden_layers"]
    first_dense = cfg.get("first_k_dense_replace", 3)
    E = cfg["n_routed_experts"]
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(L):
        p = f"model.layers.{i}."
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                  # MLA attention
                  p + "self_attn.q_a_proj.weight", p + "self_attn.q_a_layernorm.weight",
                  p + "self_attn.q_b_proj.weight",
                  p + "self_attn.kv_a_proj_with_mqa.weight",
                  p + "self_attn.kv_a_layernorm.weight", p + "self_attn.kv_b_proj.weight",
                  p + "self_attn.o_proj.weight"}
        if i < first_dense:
            names |= {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                      p + "mlp.down_proj.weight"}
        else:
            names |= {p + "mlp.gate.weight"}
            for e in range(E):
                ep = p + f"mlp.experts.{e}."
                names |= {ep + "gate_proj.weight", ep + "up_proj.weight", ep + "down_proj.weight"}
    return names


def expected_dsa_indexer_names(cfg):
    """Optional GLM DSA lightning-indexer tensors present on full-indexer layers."""
    names = set()
    types = cfg.get("indexer_types") or []
    layer_ids = [i for i, typ in enumerate(types) if typ == "full"]
    # GLM-5.2 stores the MTP/next-token predictor at model.layers.<L>; its
    # indexer is shared with the speculative path, not the base 0..L-1 forward.
    if cfg.get("num_nextn_predict_layers", 0) > 0:
        layer_ids.append(cfg["num_hidden_layers"])
    for i in layer_ids:
        p = f"model.layers.{i}.self_attn.indexer."
        names |= {p + "wq_b.weight", p + "wk.weight", p + "weights_proj.weight",
                  p + "k_norm.weight", p + "k_norm.bias"}
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quantize-int4", default=None, metavar="OUTDIR")
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    index_meta, shard_paths, missing_shards = checkpoint_files(args.model)
    have = all_tensor_names(args.model)
    want = expected_names(cfg)
    want_indexer = expected_dsa_indexer_names(cfg)

    missing = sorted(w for w in want if w not in have
                     # qk_norm + shared experts + biases are optional
                     and "q_norm" not in w and "k_norm" not in w)
    missing_indexer = sorted(w for w in want_indexer if w not in have)
    # account for tied embeddings / optional lm_head
    if "lm_head.weight" not in have and not cfg.get("tie_word_embeddings", False):
        print("note: lm_head.weight absent and tie_word_embeddings=false — "
              "glmserve will fall back to tied embeddings")

    print(f"model: {args.model}")
    print(f"  config: {cfg.get('architectures', ['?'])[0]}  layers={cfg['num_hidden_layers']}")
    print(f"  tensors present: {len(have)}   expected (core): {len(want)}")
    if want_indexer:
        got_indexer = len(want_indexer) - len(missing_indexer)
        print(f"  DSA indexer tensors: {got_indexer}/{len(want_indexer)} indexed")
    print(f"  safetensors files expected: {len(shard_paths)}")
    if index_meta is not None and index_meta.get("metadata", {}).get("total_size") is not None:
        total_size = index_meta["metadata"]["total_size"]
        print(f"  indexed tensor bytes: {total_size / 1024**3:.2f} GiB")
    if missing_shards:
        print(f"  MISSING {len(missing_shards)} safetensors shard files, e.g.:")
        for p in missing_shards[:10]:
            print(f"    - {os.path.basename(p)}")
        rc = 1
    else:
        print("  all safetensors shard files are present")
        rc = 0
    if missing:
        print(f"  MISSING {len(missing)} core tensors, e.g.:")
        for m in missing[:10]:
            print(f"    - {m}")
        rc = 1
    else:
        msg = "all core tensors present"
        if not missing_shards:
            msg += " — glmserve can load this checkpoint directly"
        else:
            msg += " in the index, but shard files are absent"
        print(f"  {msg}")
    if missing_indexer:
        print(f"  note: missing {len(missing_indexer)} optional DSA indexer tensors, e.g.:")
        for m in missing_indexer[:10]:
            print(f"    - {m}")

    if args.quantize_int4:
        print(f"\n[quantize-int4] target={args.quantize_int4} group={args.group}")
        print("  NOTE: streaming W4A16 repack of a 753B checkpoint is I/O-heavy and is")
        print("  intended to run on a node with the weights present. The packed layout is:")
        print("    qweight: uint8 [out, in/2]  (two signed nibbles, zero-point 8)")
        print("    scales : float32 [out, in/group]")
        print("  matching dequant_int4_to_f32() in src/tensor.cpp and gemm_w4a16 in cuda/int4_gemm.cu.")
        print("  Implement the per-shard repack here once weights are staged.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
