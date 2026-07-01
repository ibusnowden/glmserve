#!/usr/bin/env bash
# Validate the incremental GPU decode path and measure prefill/decode throughput.
# Invoke via:
#   srun -p bigTiger --gres=gpu:rtx_6000:1 -t 20:00 bash scripts/gpu_bench.sh
#
# Runs three things on real hardware against a node-local tiny checkpoint:
#   1. prefill logits gate (full device forward == numpy MLA reference)
#   2. gencheck: incremental forward_gpu_decode == re-prefill reference
#   3. bench: prefill + decode tok/s at a few sizes (CPU baseline for contrast)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

echo "=== GPU ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found; run on a GPU node"; exit 1; }
nvidia-smi -L >/dev/null || { echo "no NVIDIA GPU visible; run on a GPU node"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export GLMSERVE_REQUIRE_CUDA=1

# Always (re)build a CUDA binary. CPU and GPU builds share build/ object paths,
# so clean first to avoid linking CPU-compiled objects into the GPU binary.
JOBS="${GLMSERVE_BUILD_JOBS:-4}"
make clean >/dev/null 2>&1
make GPU=1 -j"$JOBS" || { echo "BUILD FAILED"; exit 1; }

TINY=$(mktemp -d)
trap 'rm -rf "$TINY"' EXIT
python3 tools/make_tiny_checkpoint.py --out "$TINY" --prompt "3 1 4 1 5 9 2 6" >/dev/null
echo "tiny checkpoint -> $TINY"

echo
echo "=== 1. prefill logits gate (device forward vs numpy MLA reference) ==="
GLMSERVE_LOG=error python3 tests/test_logits_match.py --bin build/glmserve --gpu \
  --tol "${GLMSERVE_GPU_LOGITS_TOL:-5e-2}"

echo
echo "=== 2. incremental decode correctness (forward_gpu_decode vs re-prefill) ==="
GLMSERVE_LOG=error build/glmserve gencheck --model "$TINY" --tokens "3 1 4 1 5 9 2 6" --gen 48 --gpu

echo
echo "=== 3. throughput (CUDA) ==="
for cfg in "128 64" "512 128" "1024 256" "2048 256"; do
  set -- $cfg
  GLMSERVE_LOG=error build/glmserve bench --model "$TINY" --prompt-len "$1" --gen-len "$2" --gpu
  echo
done

echo "=== CPU baseline (same engine, reference path) ==="
GLMSERVE_LOG=error build/glmserve bench --model "$TINY" --prompt-len 512 --gen-len 128

echo "=== gpu_bench done ==="
