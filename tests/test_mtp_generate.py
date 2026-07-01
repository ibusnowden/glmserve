#!/usr/bin/env python3
"""MTP-assisted greedy generation must match ordinary greedy generation."""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run_tokgen(bin_path, model_dir, mtp_draft_k=0):
    cmd = [
        bin_path, "tokgen", "--model", model_dir,
        "--tokens", "3 1 4 1 5 9 2 6",
        "--max-tokens", "12",
        "--ignore-eos",
    ]
    if mtp_draft_k:
        cmd += ["--mtp-draft-k", str(mtp_draft_k)]
    run = subprocess.run(cmd, capture_output=True, text=True,
                         env=dict(os.environ, GLMSERVE_LOG="error"))
    if run.returncode != 0:
        raise RuntimeError(run.stderr or run.stdout)
    summary = re.search(
        r"tokgen: prompt=(\d+) generated=(\d+) finish=(\w+) mtp=(\w+) groups=(\d+) accepted=(\d+) rejected=(\d+)",
        run.stdout)
    tokens = re.search(r"tokens:\s*([0-9 ]*)", run.stdout)
    if not summary or not tokens:
        raise RuntimeError(f"cannot parse tokgen output:\n{run.stdout}")
    return {
        "prompt": int(summary.group(1)),
        "generated": int(summary.group(2)),
        "finish": summary.group(3),
        "mtp": summary.group(4),
        "groups": int(summary.group(5)),
        "accepted": int(summary.group(6)),
        "rejected": int(summary.group(7)),
        "tokens": [int(x) for x in tokens.group(1).split()],
        "stdout": run.stdout,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "glmserve"))
    ap.add_argument("--python", default=sys.executable)
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

        base = run_tokgen(args.bin, d)
        mtp = run_tokgen(args.bin, d, mtp_draft_k=5)
        print(f"base_tokens={base['tokens']}")
        print(f"mtp_tokens ={mtp['tokens']}")
        print(f"mtp_stats groups={mtp['groups']} accepted={mtp['accepted']} rejected={mtp['rejected']}")
        if base["tokens"] != mtp["tokens"]:
            print("FAIL: MTP-assisted generation changed greedy token stream")
            return 1
        if mtp["mtp"] != "on" or mtp["groups"] <= 0 or mtp["accepted"] < mtp["groups"]:
            print("FAIL: MTP path did not run as expected")
            return 1
        print("test_mtp_generate: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
