#!/usr/bin/env python3
"""W4A16 checkpoint gate.

This test exercises the real W4 converter and loader contract:
  1. create a tiny GLM-style checkpoint,
  2. repack linear weights to packed int4 + scales,
  3. load the W4 checkpoint on CPU, where weights are dequantized to f32,
  4. optionally load the same W4 checkpoint on GPU with quantized-only resident
     weights and compare logits against the CPU W4 path.

The original fp checkpoint is used only to report quantization drift; W4 changes
the model, so CPU-W4 vs GPU-W4 is the correctness oracle.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print("FAIL:", " ".join(cmd))
        if p.stdout:
            print(p.stdout)
        if p.stderr:
            print(p.stderr)
        raise SystemExit(1)
    return p


def dump_logits(bin_path, model, prompt, out_path, gpu=False):
    env = dict(os.environ)
    env["GLMSERVE_LOG"] = "info" if gpu else "error"
    env["GLMSERVE_QUANT_ONLY"] = "1" if gpu else "0"
    cmd = [bin_path, "dump", "--model", model, "--tokens", prompt, "--out", out_path]
    if gpu:
        cmd.append("--gpu")
    p = run(cmd, env=env)
    if gpu and "forward path: CUDA" not in p.stderr:
        print("FAIL: --gpu requested but CUDA path did not activate")
        print(p.stderr)
        raise SystemExit(1)
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def max_abs(a, b):
    x = np.array(a["logits"], dtype=np.float64)
    y = np.array(b["logits"], dtype=np.float64)
    if x.shape != y.shape:
        print(f"FAIL: shape mismatch {x.shape} != {y.shape}")
        raise SystemExit(1)
    return float(np.max(np.abs(x - y)))


def count_qweights(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    return sum(1 for name in weight_map if name.endswith(".qweight"))


def first_shard(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    return sorted(set(weight_map.values()))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "glmserve"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--prompt", default="3 1 4 1 5 9 2 6")
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2,
                    help="CPU-W4 vs GPU-W4 max absolute logits tolerance")
    ap.add_argument("--gencheck", type=int, default=0,
                    help="optional GPU incremental-decode check steps")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"FAIL: engine binary not found at {args.bin}")
        return 1

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        w4 = os.path.join(d, "w4")
        run([args.python, os.path.join(ROOT, "tools", "make_tiny_checkpoint.py"),
             "--out", src, "--prompt", args.prompt])
        run([args.python, os.path.join(ROOT, "tools", "convert_hf_to_glmserve.py"),
             "--model", src, "--check", "--quantize-int4", w4, "--group", str(args.group)])
        run([args.python, os.path.join(ROOT, "tools", "convert_hf_to_glmserve.py"),
             "--model", w4, "--check"])

        index_path = os.path.join(w4, "model.safetensors.index.json")
        qweights = count_qweights(index_path)
        if qweights <= 0:
            print("FAIL: quantized checkpoint contains no .qweight tensors")
            return 1
        shard = os.path.join(w4, first_shard(index_path))
        stale_part = shard + ".part"
        with open(stale_part, "wb") as f:
            f.write(b"stale")
        rerun = run([args.python, os.path.join(ROOT, "tools", "convert_hf_to_glmserve.py"),
                    "--model", src, "--check", "--quantize-int4", w4, "--group", str(args.group)])
        if "skipped" not in rerun.stdout or os.path.exists(stale_part):
            print("FAIL: resumable W4 conversion did not skip a valid shard and remove stale .part")
            print(rerun.stdout)
            return 1
        with open(shard, "r+b") as f:
            f.truncate(16)
        repair = run([args.python, os.path.join(ROOT, "tools", "convert_hf_to_glmserve.py"),
                      "--model", src, "--check", "--quantize-int4", w4, "--group", str(args.group)])
        if "wrote" not in repair.stdout:
            print("FAIL: resumable W4 conversion did not rebuild a corrupt shard")
            print(repair.stdout)
            return 1
        run([args.python, os.path.join(ROOT, "tools", "convert_hf_to_glmserve.py"),
             "--model", w4, "--check"])

        ref = json.load(open(os.path.join(src, "reference_logits.json"), encoding="utf-8"))
        cpu_w4 = dump_logits(args.bin, w4, args.prompt, os.path.join(d, "cpu_w4.json"),
                             gpu=False)
        drift = max_abs(ref, cpu_w4)
        print(f"qweight_tensors={qweights}  cpu_w4_argmax={cpu_w4['argmax']}  "
              f"fp_to_w4_drift={drift:.3e}")

        if args.gpu:
            gpu_w4 = dump_logits(args.bin, w4, args.prompt, os.path.join(d, "gpu_w4.json"),
                                 gpu=True)
            diff = max_abs(cpu_w4, gpu_w4)
            argmax_ok = cpu_w4["argmax"] == gpu_w4["argmax"]
            print(f"gpu_w4_argmax={gpu_w4['argmax']}  cpu_gpu_w4_max_abs={diff:.3e}  "
                  f"tol={args.tol:.1e}  argmax_match={argmax_ok}")
            if not argmax_ok or diff > args.tol:
                print("test_w4_quantized: FAIL")
                return 1
            if args.gencheck > 0:
                env = dict(os.environ, GLMSERVE_LOG="error", GLMSERVE_QUANT_ONLY="1")
                p = run([args.bin, "gencheck", "--model", w4, "--tokens", args.prompt,
                         "--gen", str(args.gencheck), "--gpu"], env=env)
                print(p.stdout.strip())

        print("test_w4_quantized: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
