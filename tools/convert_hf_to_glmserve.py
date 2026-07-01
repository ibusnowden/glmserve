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
  python tools/convert_hf_to_glmserve.py --model DIR [--check] [--quantize-int4 DIR] [--group 128] [--force]
"""
import argparse
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np


DTYPE_SIZE = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def read_header_and_data_offset(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8")), 8 + n


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


def has_tensor(have, name):
    if name in have:
        return True
    if name.endswith(".weight"):
        return name[:-len(".weight")] + ".qweight" in have
    return False


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


def expected_mtp_names(cfg):
    """Optional GLM MTP/next-token-prediction layer tensors."""
    names = set()
    n_mtp = int(cfg.get("num_nextn_predict_layers", 0) or 0)
    if n_mtp <= 0:
        return names
    L = cfg["num_hidden_layers"]
    E = cfg["n_routed_experts"]
    for i in range(L, L + n_mtp):
        p = f"model.layers.{i}."
        names |= {p + "eh_proj.weight", p + "enorm.weight", p + "hnorm.weight",
                  p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                  p + "shared_head.norm.weight",
                  p + "self_attn.q_a_proj.weight", p + "self_attn.q_a_layernorm.weight",
                  p + "self_attn.q_b_proj.weight",
                  p + "self_attn.kv_a_proj_with_mqa.weight",
                  p + "self_attn.kv_a_layernorm.weight", p + "self_attn.kv_b_proj.weight",
                  p + "self_attn.o_proj.weight",
                  p + "self_attn.indexer.wq_b.weight",
                  p + "self_attn.indexer.wk.weight",
                  p + "self_attn.indexer.weights_proj.weight",
                  p + "self_attn.indexer.k_norm.weight",
                  p + "self_attn.indexer.k_norm.bias",
                  p + "mlp.gate.weight",
                  p + "mlp.gate.e_score_correction_bias",
                  p + "mlp.shared_experts.gate_proj.weight",
                  p + "mlp.shared_experts.up_proj.weight",
                  p + "mlp.shared_experts.down_proj.weight"}
        for e in range(E):
            ep = p + f"mlp.experts.{e}."
            names |= {ep + "gate_proj.weight", ep + "up_proj.weight", ep + "down_proj.weight"}
    return names


def is_quantizable_linear(name, meta):
    if not name.endswith(".weight"):
        return False
    if name in {"model.embed_tokens.weight", "embed_tokens.weight"}:
        return False
    shape = meta.get("shape") or []
    if len(shape) != 2:
        return False
    return meta.get("dtype") in {"F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


def dtype_nbytes(dtype):
    if dtype not in DTYPE_SIZE:
        raise SystemExit(f"unsupported dtype in safetensors: {dtype}")
    return DTYPE_SIZE[dtype]


def tensor_nbytes(meta):
    n = 1
    for d in meta["shape"]:
        n *= int(d)
    return n * dtype_nbytes(meta["dtype"])


def decode_block_to_f32(raw, dtype, shape):
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=False)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(shape)
    if dtype == "F8_E4M3":
        return decode_f8(np.frombuffer(raw, dtype=np.uint8), exp_bits=4, mant_bits=3, bias=7).reshape(shape)
    if dtype == "F8_E5M2":
        return decode_f8(np.frombuffer(raw, dtype=np.uint8), exp_bits=5, mant_bits=2, bias=15).reshape(shape)
    raise SystemExit(f"cannot quantize dtype {dtype}")


def decode_f8(u, exp_bits, mant_bits, bias):
    u = u.astype(np.uint16)
    sign = np.where((u & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp_mask = (1 << exp_bits) - 1
    mant_mask = (1 << mant_bits) - 1
    exp = ((u >> mant_bits) & exp_mask).astype(np.int32)
    mant = (u & mant_mask).astype(np.float32)
    out = np.zeros(u.shape, dtype=np.float32)
    normal = (exp != 0) & (exp != exp_mask)
    sub = exp == 0
    out[normal] = sign[normal] * np.ldexp(1.0 + mant[normal] / (1 << mant_bits),
                                          exp[normal] - bias)
    out[sub] = sign[sub] * np.ldexp(mant[sub] / (1 << mant_bits), 1 - bias)
    inf_nan = exp == exp_mask
    out[inf_nan] = sign[inf_nan] * np.inf
    return out


def quantize_rows_to_w4(fin, fout, abs_offset, dtype, shape, group, tmpdir):
    out, in_dim = map(int, shape)
    groups = (in_dim + group - 1) // group
    packed_in = (in_dim + 1) // 2
    row_bytes = in_dim * dtype_nbytes(dtype)
    block_rows = max(1, min(256, 64 * 1024 * 1024 // max(1, row_bytes)))

    scale_tmp = tempfile.TemporaryFile(dir=tmpdir)
    for r0 in range(0, out, block_rows):
        rows = min(block_rows, out - r0)
        fin.seek(abs_offset + r0 * row_bytes)
        raw = fin.read(rows * row_bytes)
        if len(raw) != rows * row_bytes:
            raise SystemExit("short read while quantizing safetensors shard")
        x = decode_block_to_f32(raw, dtype, (rows, in_dim))
        qvals = np.zeros((rows, in_dim), dtype=np.uint8)
        scales = np.empty((rows, groups), dtype="<f4")
        for g in range(groups):
            c0 = g * group
            c1 = min(in_dim, c0 + group)
            chunk = x[:, c0:c1]
            s = np.max(np.abs(chunk), axis=1) / 7.0
            s = np.where(s > 0.0, s, 1.0).astype(np.float32)
            scales[:, g] = s
            q = np.rint(chunk / s[:, None])
            q = np.clip(q, -8, 7).astype(np.int16) + 8
            qvals[:, c0:c1] = q.astype(np.uint8)
        packed = np.zeros((rows, packed_in), dtype=np.uint8)
        packed[:, :qvals[:, 0::2].shape[1]] = qvals[:, 0::2]
        hi = qvals[:, 1::2]
        packed[:, :hi.shape[1]] |= (hi << 4)
        fout.write(packed.tobytes(order="C"))
        scale_tmp.write(scales.tobytes(order="C"))
    scale_tmp.seek(0)
    shutil.copyfileobj(scale_tmp, fout, length=16 * 1024 * 1024)
    scale_tmp.close()


def copy_range(fin, fout, abs_begin, nbytes):
    fin.seek(abs_begin)
    remaining = nbytes
    while remaining:
        chunk = fin.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise SystemExit("short read while copying safetensors shard")
        fout.write(chunk)
        remaining -= len(chunk)


def write_safetensors_header(fout, entries):
    header = {}
    offset = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {
            "dtype": dtype,
            "shape": [int(x) for x in shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    fout.write(struct.pack("<Q", len(raw)))
    fout.write(raw)


def expected_header(entries):
    header = {}
    offset = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {
            "dtype": dtype,
            "shape": [int(x) for x in shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    return header, offset


def output_shard_matches(path, entries):
    if not os.path.exists(path):
        return False
    try:
        header, data_offset = read_header_and_data_offset(path)
    except Exception:
        return False
    want, total = expected_header(entries)
    if header != want:
        return False
    return os.path.getsize(path) == data_offset + total


def copy_sidecar_files(model_dir, out_dir):
    skip = {os.path.abspath(out_dir)}
    for name in os.listdir(model_dir):
        src = os.path.join(model_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.abspath(src) in skip:
            continue
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def quantized_names(name):
    base = name[:-len(".weight")]
    return base + ".qweight", base + ".scales"


def quantize_checkpoint(model_dir, out_dir, group, force=False):
    index_meta, shard_paths, missing = checkpoint_files(model_dir)
    if missing:
        raise SystemExit("cannot quantize: safetensors shard files are missing")
    if os.path.abspath(model_dir) == os.path.abspath(out_dir):
        raise SystemExit("cannot quantize in place: choose a different --quantize-int4 OUTDIR")
    os.makedirs(out_dir, exist_ok=True)
    copy_sidecar_files(model_dir, out_dir)

    weight_map = {}
    total_size = 0
    skipped_alignment = 0
    for src_path in shard_paths:
        header, data_offset = read_header_and_data_offset(src_path)
        tensor_names = [n for n in header.keys() if n != "__metadata__"]
        out_file = os.path.basename(src_path)
        out_path = os.path.join(out_dir, out_file)

        generated_names = set()
        for name in tensor_names:
            meta = header[name]
            if is_quantizable_linear(name, meta):
                in_dim = int(meta["shape"][1])
                if in_dim % 2 == 0 and in_dim % group == 0:
                    generated_names.update(quantized_names(name))

        entries = []
        plans = []
        for name in tensor_names:
            meta = header[name]
            if name in generated_names:
                continue
            if is_quantizable_linear(name, meta):
                out, in_dim = map(int, meta["shape"])
                if in_dim % 2 != 0 or in_dim % group != 0:
                    skipped_alignment += 1
                    nbytes = tensor_nbytes(meta)
                    entries.append((name, meta["dtype"], meta["shape"], nbytes))
                    plans.append(("copy", name, meta))
                    weight_map[name] = out_file
                    total_size += nbytes
                    continue
                packed_in = (in_dim + 1) // 2
                groups = (in_dim + group - 1) // group
                qname, sname = quantized_names(name)
                qbytes = out * packed_in
                sbytes = out * groups * 4
                entries.append((qname, "U8", [out, packed_in], qbytes))
                entries.append((sname, "F32", [out, groups], sbytes))
                plans.append(("quant", name, meta))
                weight_map[qname] = out_file
                weight_map[sname] = out_file
                total_size += qbytes + sbytes
            else:
                nbytes = tensor_nbytes(meta)
                entries.append((name, meta["dtype"], meta["shape"], nbytes))
                plans.append(("copy", name, meta))
                weight_map[name] = out_file
                total_size += nbytes

        part_path = out_path + ".part"
        if not force and output_shard_matches(out_path, entries):
            if os.path.exists(part_path):
                os.remove(part_path)
            print(f"  skipped {out_file}: existing output shard is valid")
            continue
        if os.path.exists(part_path):
            os.remove(part_path)

        with open(src_path, "rb") as fin, open(part_path, "wb") as fout:
            write_safetensors_header(fout, entries)
            with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
                for kind, name, meta in plans:
                    begin, end = meta["data_offsets"]
                    abs_begin = data_offset + int(begin)
                    if kind == "copy":
                        copy_range(fin, fout, abs_begin, int(end) - int(begin))
                    else:
                        quantize_rows_to_w4(fin, fout, abs_begin, meta["dtype"],
                                            meta["shape"], group, tmpdir)
        os.replace(part_path, out_path)
        print(f"  wrote {out_file}: {len(entries)} tensors")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"  wrote model.safetensors.index.json ({len(weight_map)} tensors, {total_size / 1024**3:.2f} GiB)")
    if skipped_alignment:
        print(f"  kept {skipped_alignment} linear weights unquantized because their input dim is not even and group-aligned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quantize-int4", default=None, metavar="OUTDIR")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--force", action="store_true",
                    help="rebuild W4 output shards even when existing shards validate")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    index_meta, shard_paths, missing_shards = checkpoint_files(args.model)
    have = all_tensor_names(args.model)
    want = expected_names(cfg)
    want_indexer = expected_dsa_indexer_names(cfg)
    want_mtp = expected_mtp_names(cfg)

    missing = sorted(w for w in want if not has_tensor(have, w)
                     # qk_norm + shared experts + biases are optional
                     and "q_norm" not in w and "k_norm" not in w)
    missing_indexer = sorted(w for w in want_indexer if not has_tensor(have, w))
    missing_mtp = sorted(w for w in want_mtp if not has_tensor(have, w))
    # account for tied embeddings / optional lm_head
    if not has_tensor(have, "lm_head.weight") and not cfg.get("tie_word_embeddings", False):
        print("note: lm_head.weight absent and tie_word_embeddings=false — "
              "glmserve will fall back to tied embeddings")

    print(f"model: {args.model}")
    print(f"  config: {cfg.get('architectures', ['?'])[0]}  layers={cfg['num_hidden_layers']}")
    print(f"  tensors present: {len(have)}   expected (core): {len(want)}")
    if want_indexer:
        got_indexer = len(want_indexer) - len(missing_indexer)
        print(f"  DSA indexer tensors: {got_indexer}/{len(want_indexer)} indexed")
    if want_mtp:
        got_mtp = len(want_mtp) - len(missing_mtp)
        print(f"  MTP tensors: {got_mtp}/{len(want_mtp)} indexed")
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
    if missing_mtp:
        print(f"  note: missing {len(missing_mtp)} optional MTP tensors, e.g.:")
        for m in missing_mtp[:10]:
            print(f"    - {m}")

    if args.quantize_int4:
        print(f"\n[quantize-int4] target={args.quantize_int4} group={args.group}")
        if args.group <= 0:
            raise SystemExit("--group must be positive")
        print("  streaming W4A16 repack; this is I/O-heavy on the full checkpoint")
        print("  packed layout:")
        print("    qweight: uint8 [out, in/2]  (two signed nibbles, zero-point 8)")
        print("    scales : float32 [out, in/group]")
        print("  matching dequant_int4_to_f32() in src/tensor.cpp and gemm_w4a16 in cuda/int4_gemm.cu.")
        quantize_checkpoint(args.model, args.quantize_int4, args.group, force=args.force)

    return rc


if __name__ == "__main__":
    sys.exit(main())
