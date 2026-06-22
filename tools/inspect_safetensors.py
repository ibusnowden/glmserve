#!/usr/bin/env python3
"""Inspect a safetensors checkpoint without loading tensor data.

Reads the JSON header(s) (8-byte length prefix + header) of each shard and
prints tensor names, dtypes, shapes, and aggregate size. Handles single-file
and sharded (model.safetensors.index.json) layouts. This is the Phase-0 / M0
tool: understand exact GLM-5.2 tensor names and shapes before writing loaders.

Usage:
  python tools/inspect_safetensors.py <model_dir_or_file> [--grep PATTERN] [--full]
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


def dtype_bytes(dt):
    return {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
            "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}.get(dt, 0)


def collect(path):
    if os.path.isdir(path):
        idx = os.path.join(path, "model.safetensors.index.json")
        single = os.path.join(path, "model.safetensors")
        if os.path.exists(idx):
            with open(idx, "r", encoding="utf-8") as f:
                meta = json.load(f)
            wm = meta["weight_map"]
            files = sorted(set(wm.values()))
            resolved = [os.path.join(path, f) for f in files]
            missing = [f for f in resolved if not os.path.exists(f)]
            if missing:
                total_size = meta.get("metadata", {}).get("total_size")
                print(f"checkpoint index: {idx}", file=sys.stderr)
                print(f"shards expected: {len(files)}", file=sys.stderr)
                if total_size is not None:
                    print(f"indexed tensor bytes: {total_size / 1024**3:.2f} GiB", file=sys.stderr)
                print(f"missing shard files: {len(missing)}", file=sys.stderr)
                for fp in missing[:10]:
                    print(f"  - {os.path.basename(fp)}", file=sys.stderr)
                if len(missing) > 10:
                    print(f"  ... {len(missing) - 10} more", file=sys.stderr)
                raise SystemExit("safetensors shards are missing; stage them before inspecting/loading")
            return resolved
        if os.path.exists(single):
            return [single]
        # any .safetensors
        st = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")]
        if st:
            return sorted(st)
        raise SystemExit(f"no safetensors found in {path}")
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", default=None)
    ap.add_argument("--full", action="store_true", help="list every tensor")
    args = ap.parse_args()

    files = collect(args.path)
    total_bytes = 0
    total_params = 0
    rows = []
    dtype_hist = {}
    for fp in files:
        hdr = read_header(fp)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dt = meta["dtype"]
            shape = meta["shape"]
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * dtype_bytes(dt)
            total_bytes += nbytes
            total_params += numel
            dtype_hist[dt] = dtype_hist.get(dt, 0) + numel
            if args.grep is None or args.grep in name:
                rows.append((name, dt, shape, numel))

    print(f"files: {len(files)}")
    print(f"tensors: {len(rows)} shown / {total_params/1e9:.2f}B params total, "
          f"{total_bytes/1024**3:.2f} GiB on disk")
    print("dtype histogram (params):")
    for dt, c in sorted(dtype_hist.items(), key=lambda x: -x[1]):
        print(f"  {dt:8s} {c/1e9:8.3f}B")

    rows.sort()
    limit = len(rows) if args.full else min(40, len(rows))
    print(f"\n{'name':<56}{'dtype':<9}{'shape'}")
    for name, dt, shape, _ in rows[:limit]:
        print(f"{name:<56}{dt:<9}{shape}")
    if limit < len(rows):
        print(f"... ({len(rows)-limit} more; use --full)")


if __name__ == "__main__":
    main()
