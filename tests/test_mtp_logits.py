#!/usr/bin/env python3
"""MTP CPU correctness gate: glmserve mtp == tiny numpy reference."""
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
    ap.add_argument("--gpu", action="store_true",
                    help="validate the GPU MTP path (mtp --gpu) against the numpy reference")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"FAIL: engine binary not found at {args.bin}")
        return 1

    with tempfile.TemporaryDirectory() as d:
        gen = subprocess.run(
            [args.python, os.path.join(ROOT, "tools", "make_tiny_checkpoint.py"), "--out", d],
            capture_output=True, text=True)
        if gen.returncode != 0:
            print("FAIL: make_tiny_checkpoint failed")
            print(gen.stderr)
            return 1

        ref = json.load(open(os.path.join(d, "reference_mtp_logits.json"), encoding="utf-8"))
        out_path = os.path.join(d, "mtp_logits.json")
        cmd = [
            args.bin, "mtp", "--model", d,
            "--tokens", " ".join(map(str, ref["context"])),
            "--draft", " ".join(map(str, ref["draft"])),
            "--out", out_path,
        ]
        if args.gpu:
            cmd.append("--gpu")
        run = subprocess.run(cmd, capture_output=True, text=True,
                             env=dict(os.environ, GLMSERVE_LOG="error"))
        if run.returncode != 0:
            print("FAIL: glmserve mtp failed")
            print(run.stderr)
            return 1
        got = json.load(open(out_path, encoding="utf-8"))

        r = np.array(ref["logits"], dtype=np.float64)
        g = np.array(got["logits"], dtype=np.float64)
        if r.shape != g.shape:
            print(f"FAIL: shape mismatch ref={r.shape} got={g.shape}")
            return 1
        max_abs = float(np.max(np.abs(r - g)))
        argmax_ok = ref["argmax"] == got["argmax"]
        print(f"draft={len(ref['draft'])} vocab={ref['vocab']} ref_argmax={ref['argmax']} got_argmax={got['argmax']}")
        print(f"max_abs_diff={max_abs:.3e} tol={args.tol:.1e} argmax_match={argmax_ok}")
        if not argmax_ok or max_abs > args.tol:
            print("test_mtp_logits: FAIL")
            return 1
        print("test_mtp_logits: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
