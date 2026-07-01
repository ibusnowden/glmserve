#!/usr/bin/env bash
# Build CUDA and run the repeatable W4A16 tiny-checkpoint gate on one RTX GPU.
# Invoke via:
#   srun -p bigTiger --gres=gpu:rtx_6000:1 -t 20:00 bash scripts/w4_gpu_check.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

echo "=== GPU ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found; run on a GPU node"; exit 1; }
nvidia-smi -L >/dev/null || { echo "no NVIDIA GPU visible; run on a GPU node"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export GLMSERVE_REQUIRE_CUDA=1

echo "=== build GPU=1 ==="
JOBS="${GLMSERVE_BUILD_JOBS:-4}"
make clean >/dev/null 2>&1
make GPU=1 -j"$JOBS"
make tests GPU=1 -j"$JOBS"

echo "=== W4A16 tiny checkpoint: converter + CPU dequant + GPU resident qweight ==="
PY="${GLMSERVE_PYTHON:-python3}"
"$PY" tests/test_w4_quantized.py --bin build/glmserve --gpu \
  --tol "${GLMSERVE_W4_GPU_TOL:-5e-2}" --gencheck "${GLMSERVE_W4_GENCHECK_STEPS:-16}"

echo "=== w4_gpu_check done ==="
