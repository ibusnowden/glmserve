#!/usr/bin/env bash
# Build glmserve with CUDA on a GPU node and run the kernel unit tests against
# the CPU reference on real hardware. Invoke via:
#   srun -p bigTiger --gres=gpu:rtx_6000:1 -t 20:00 bash scripts/gpu_check.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

echo "=== GPU ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found; run on a GPU node"; exit 1; }
nvidia-smi -L >/dev/null || { echo "no NVIDIA GPU visible; run on a GPU node"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export GLMSERVE_REQUIRE_CUDA=1

echo "=== build GPU=1 ==="
echo "CXX=$CXX"
JOBS="${GLMSERVE_BUILD_JOBS:-4}"
make clean >/dev/null 2>&1
make GPU=1 -j"$JOBS" || { echo "BUILD FAILED"; exit 1; }
make tests GPU=1 -j"$JOBS" || { echo "TEST BUILD FAILED"; exit 1; }

echo "=== run kernel tests on GPU ==="
rc=0
for t in build/gpu/tests/*; do
  [ -x "$t" ] || continue
  echo "--- $t"
  "$t" || rc=1
done

echo "=== end-to-end GPU forward vs numpy reference ==="
# Validates the full device forward (model_gpu.cpp + all kernels) against the
# numpy MLA reference through `glmserve dump --gpu`. Learned sparse DSA makes the
# end-to-end logits more sensitive to small GEMM/selector rounding than the dense
# path; the exact selector and indexed-attention kernels are gated above.
python3 tests/test_logits_match.py --bin build/glmserve --gpu --tol "${GLMSERVE_GPU_LOGITS_TOL:-5e-2}" || rc=1

echo "=== gpu_check exit $rc ==="
exit $rc
