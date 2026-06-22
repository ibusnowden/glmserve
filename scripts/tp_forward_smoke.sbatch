#!/usr/bin/env bash
#SBATCH --job-name=glmserve-tpfwd
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/project/inniang/glmserve/scripts/tp_forward_smoke.%j.out
#
# Build, generate a tiny GLM-5.2 checkpoint, then run the tensor-parallel forward
# across 2 ranks (TP=2, PP=1) and check each rank's logits match a single-process
# full forward. This validates that the NCCL TP all-reduce primitive is wired
# into real GLM layer execution (head-sharded attention + row-parallel o_proj /
# down_proj reconstructed with all_reduce).
# Submit with:
#   sbatch scripts/tp_forward_smoke.sbatch
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

JOBS="${GLMSERVE_BUILD_JOBS:-4}"
make GPU=1 -j"$JOBS"
make tests GPU=1 -j"$JOBS"

# Tiny same-architecture checkpoint (4 heads, MoE) shared by both ranks on this
# node's /tmp. TP=2 splits attention into 2 heads/rank and each FFN in half.
CKPT="/tmp/glmserve_tp_tiny_${SLURM_JOB_ID}"
PY="${GLMSERVE_PYTHON:-python3}"
echo "=== generate tiny checkpoint -> $CKPT ==="
"$PY" tools/make_tiny_checkpoint.py --out "$CKPT"
test -f "$CKPT/model.safetensors" || { echo "FAIL: tiny checkpoint not generated"; exit 1; }

export GLMSERVE_TINY_CKPT="$CKPT"
export GLMSERVE_TP_SIZE=2
export GLMSERVE_PP_SIZE=1
export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_tpfwd_${SLURM_JOB_ID}.id"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
rm -f "$GLMSERVE_NCCL_ID_FILE"

echo "=== 2-rank TP=2 forward correctness (head/FFN-sharded logits == single-process) ==="
srun --ntasks=2 bash -lc \
  'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exec build/gpu/tests/test_tp_forward'

rm -f "$GLMSERVE_NCCL_ID_FILE"
rm -rf "$CKPT"
echo "=== tp_forward_smoke done ==="
