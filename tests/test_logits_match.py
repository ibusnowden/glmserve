#!/usr/bin/env python3
"""End-to-end correctness gate: glmserve forward == numpy reference.

Builds a tiny GLM-5.2-architecture checkpoint (tools/make_tiny_checkpoint.py),
runs `glmserve dump` to get the engine's last-token logits, and compares them to
the numpy reference forward. This is the real Phase-1 / M1 / M2 check — same
architecture, same math conventions, exercised through the full C++ stack
(safetensors load -> embed -> 78-style block forward -> MoE -> logits).

Usage:
  python tests/test_logits_match.py [--bin build/glmserve] [--tol 1e-3]
Exit code 0 on pass.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "glmserve"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--prompt", default="3 1 4 1 5 9 2 6")
    ap.add_argument("--gpu", action="store_true",
                    help="run the engine forward on the CUDA path (needs a GPU=1 build + a GPU)")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"FAIL: engine binary not found at {args.bin} (run `make` first)")
        return 1

    with tempfile.TemporaryDirectory() as d:
        # 1) build tiny checkpoint + reference
        gen = subprocess.run(
            [args.python, os.path.join(ROOT, "tools", "make_tiny_checkpoint.py"),
             "--out", d, "--prompt", args.prompt],
            capture_output=True, text=True)
        if gen.returncode != 0:
            print("FAIL: make_tiny_checkpoint failed\n", gen.stderr)
            return 1

        ref = json.load(open(os.path.join(d, "reference_logits.json")))

        # 2) run the engine's dump (optionally on the CUDA forward path).
        # For --gpu, raise the log level to "info" so we can confirm the engine
        # actually activated the CUDA path instead of silently falling back to CPU.
        out_path = os.path.join(d, "glmserve_logits.json")
        env = dict(os.environ, GLMSERVE_LOG="info" if args.gpu else "error")
        cmd = [args.bin, "dump", "--model", d, "--tokens", args.prompt, "--out", out_path]
        if args.gpu:
            cmd.append("--gpu")
        run = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if run.returncode != 0:
            print("FAIL: glmserve dump failed\n", run.stderr)
            return 1
        if args.gpu and "forward path: CUDA" not in run.stderr:
            print("FAIL: --gpu requested but engine did not activate the CUDA path\n", run.stderr)
            return 1
        got = json.load(open(out_path))

        # 3) compare
        r = np.array(ref["logits"], dtype=np.float64)
        g = np.array(got["logits"], dtype=np.float64)
        if r.shape != g.shape:
            print(f"FAIL: shape mismatch ref={r.shape} got={g.shape}")
            return 1
        max_abs = float(np.max(np.abs(r - g)))
        argmax_ok = ref["argmax"] == got["argmax"]
        print(f"vocab={ref['vocab']}  ref_argmax={ref['argmax']}  got_argmax={got['argmax']}")
        print(f"max_abs_diff={max_abs:.3e}  tol={args.tol:.1e}  argmax_match={argmax_ok}")

        if not argmax_ok or max_abs > args.tol:
            print("test_logits_match: FAIL")
            return 1
        print("test_logits_match: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
