#!/usr/bin/env python3
"""Report safetensors shard requirements for PP stages.

This reads config.json + model.safetensors.index.json only. It does not require
the large shard files to be present, so it is safe to run while staging a real
checkpoint under quota pressure.
"""
import argparse
import json
import os


def partition_layers(total, stage, pp):
    base = total // pp
    rem = total % pp
    begin = stage * base + min(stage, rem)
    end = begin + base + (1 if stage < rem else 0)
    return begin, end


def stage_prefixes(cfg, stage, pp, max_layers=-1):
    total = min(max_layers, cfg["num_hidden_layers"]) if max_layers > 0 else cfg["num_hidden_layers"]
    begin, end = partition_layers(total, stage, pp)
    prefixes = []
    first = stage == 0
    last = stage == pp - 1
    if first or last:
        prefixes += ["model.embed_tokens.", "embed_tokens."]
    if last:
        prefixes += ["model.norm.", "norm.", "lm_head."]
    for i in range(begin, end):
        prefixes.append(f"model.layers.{i}.")
    if last:
        for m in range(int(cfg.get("num_nextn_predict_layers", 0) or 0)):
            prefixes.append(f"model.layers.{cfg['num_hidden_layers'] + m}.")
    return prefixes, (begin, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pp-size", type=int, default=2)
    ap.add_argument("--stage", type=int, default=-1, help="stage to report; default all")
    ap.add_argument("--max-layers", type=int, default=-1)
    ap.add_argument("--show-files", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json"), encoding="utf-8"))
    idx_path = os.path.join(args.model, "model.safetensors.index.json")
    idx = json.load(open(idx_path, encoding="utf-8"))
    weight_map = idx["weight_map"]
    all_files = sorted(set(weight_map.values()))

    stages = range(args.pp_size) if args.stage < 0 else [args.stage]
    for stage in stages:
        if stage < 0 or stage >= args.pp_size:
            raise SystemExit(f"stage {stage} out of range for pp={args.pp_size}")
        prefixes, (begin, end) = stage_prefixes(cfg, stage, args.pp_size, args.max_layers)
        files = sorted(set(
            fname for name, fname in weight_map.items()
            if any(name.startswith(p) for p in prefixes)
        ))
        present = [f for f in files if os.path.exists(os.path.join(args.model, f))]
        missing = [f for f in files if f not in present]
        print(f"stage={stage}/{args.pp_size} layers=[{begin},{end}) "
              f"shards={len(files)}/{len(all_files)} present={len(present)} missing={len(missing)}")
        if args.show_files:
            for f in files:
                mark = "present" if f in present else "missing"
                print(f"  {mark:7s} {f}")


if __name__ == "__main__":
    main()
