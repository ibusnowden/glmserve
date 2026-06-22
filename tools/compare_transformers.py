#!/usr/bin/env python3
"""Compare glmserve logits against HuggingFace Transformers (Phase-1 §16 check).

When real GLM-5.2 weights are staged and `transformers` can load the architecture,
this runs the same token prompt through both HF and `glmserve dump` and reports
the max-abs logit difference and whether the greedy argmax matches. If weights or
transformers are unavailable it explains what's missing and exits cleanly.

Usage:
  python tools/compare_transformers.py --model DIR --bin build/glmserve \
      --tokens "1 2 3 4" [--tol 1e-1]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default="build/glmserve")
    ap.add_argument("--tokens", default="1 2 3 4 5 6 7 8")
    ap.add_argument("--tol", type=float, default=1e-1)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as e:  # noqa
        print(f"transformers/torch unavailable ({e}); cannot build a reference.")
        return 0

    ids = [int(x) for x in args.tokens.split()]

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True)
    except Exception as e:  # noqa
        print("Could not load GLM-5.2 via transformers (weights not staged or arch")
        print(f"not registered): {e}")
        print("This harness becomes the reference once weights + a compatible")
        print("transformers/vLLM build are present on the node.")
        return 0

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([ids]))
        ref = out.logits[0, -1].float().cpu().numpy()

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "glmserve_logits.json")
        env = dict(os.environ, GLMSERVE_LOG="error")
        subprocess.run([args.bin, "dump", "--model", args.model,
                        "--tokens", args.tokens, "--out", out_path],
                       check=True, env=env)
        got = json.load(open(out_path))

    import numpy as np
    g = np.array(got["logits"])
    md = float(np.max(np.abs(ref - g)))
    print(f"HF argmax={int(np.argmax(ref))}  glmserve argmax={got['argmax']}")
    print(f"max_abs_diff={md:.3e}  tol={args.tol:.1e}")
    ok = (int(np.argmax(ref)) == got["argmax"]) and md <= args.tol
    print("compare_transformers:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
