#!/usr/bin/env bash
# Quick single-GPU gate for adaptive GPU speculative decode fallback.
# Forces a one-group fallback on a tiny checkpoint and checks the bench log
# reports it; also runs the GPU MTP parity checker when a CUDA device is present.
#SBATCH --job-name=glmserve-adapt
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/inniang/glmserve/scripts/adaptive_smoke.%j.out

set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

TINY="$(mktemp -d)"
trap 'rm -rf "$TINY"' EXIT
python3 tools/make_tiny_checkpoint.py --out "$TINY" --prompt "3 1 4 1 5 9 2 6"

echo "=== mtpcheck GPU parity ==="
GLMSERVE_REQUIRE_CUDA=1 GLMSERVE_LOG=error \
  build/glmserve mtpcheck --model "$TINY" --tokens "3 1 4 1 5 9 2 6" \
  --gen 12 --draft-k 4 --gpu

echo "=== adaptive fallback bench ==="
GLMSERVE_REQUIRE_CUDA=1 \
GLMSERVE_SPEC_PROBE_GROUPS=1 \
GLMSERVE_SPEC_MIN_ACCEPT=99 \
GLMSERVE_LOG=error \
  build/glmserve bench --model "$TINY" --gpu --ctx 128 \
  --prompt-len 8 --gen-len 16 --mtp-draft-k 4 | tee "$TINY/bench.out"

grep -q "adaptive-fallback" "$TINY/bench.out"
echo "adaptive fallback smoke: PASS"
