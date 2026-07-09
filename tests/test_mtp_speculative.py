#!/usr/bin/env python3
"""MTP speculative checker gate on a tiny checkpoint."""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "glmserve"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--draft-k", type=int, default=5)
    ap.add_argument("--gpu", action="store_true",
                    help="run the GPU speculative loop (gates spec == plain greedy)")
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

        cmd = [
            args.bin, "mtpcheck", "--model", d,
            "--tokens", "3 1 4 1 5 9 2 6",
            "--gen", str(args.gen),
            "--draft-k", str(args.draft_k),
        ]
        if args.gpu:
            cmd.append("--gpu")
        run = subprocess.run(cmd, capture_output=True, text=True,
                             env=dict(os.environ, GLMSERVE_LOG="error"))
        if run.returncode != 0:
            print("FAIL: glmserve mtpcheck failed")
            print(run.stdout)
            print(run.stderr)
            return 1
        print(run.stdout.strip())

        m = re.search(
            r"mtpcheck: backend=\S+ steps=(\d+) draft_k=(\d+) groups=(\d+) accepted=(\d+) rejected=(\d+)",
            run.stdout)
        if not m:
            print("FAIL: missing mtpcheck summary")
            return 1
        steps, draft_k, groups, accepted, rejected = map(int, m.groups())
        if steps != args.gen or draft_k != args.draft_k:
            print(f"FAIL: summary mismatch steps={steps} draft_k={draft_k}")
            return 1
        # The first proposal in each speculative group is seeded from target
        # logits, so every group should have at least one accepted token.
        if accepted < groups or accepted + rejected < steps:
            print(f"FAIL: invalid accept/reject counts groups={groups} accepted={accepted} rejected={rejected}")
            return 1
        if "RESULT: PASS" not in run.stdout:
            print("FAIL: missing PASS marker")
            return 1
        print("test_mtp_speculative: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
