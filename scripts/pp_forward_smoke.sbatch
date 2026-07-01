#!/usr/bin/env bash
#SBATCH --job-name=glmserve-ppfwd
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/project/inniang/glmserve/scripts/pp_forward_smoke.%j.out
#
# Build, generate a tiny GLM-5.2 checkpoint, then run the pipeline-parallel
# GPU forward across 2 ranks (TP=1, PP=2) and check the last stage's logits
# match a single-process full forward of the same checkpoint. This validates
# that the NCCL PP send/recv primitives are wired into device-resident GLM
# layer-stage execution.
# Submit with:
#   sbatch scripts/pp_forward_smoke.sbatch
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

JOBS="${GLMSERVE_BUILD_JOBS:-4}"
make GPU=1 -j"$JOBS"
make tests GPU=1 -j"$JOBS"

# Tiny same-architecture checkpoint (4 layers: 2 dense + 2 MoE) shared by both
# ranks on this node's /tmp. PP=2 splits it into [0,2) and [2,4).
CKPT="/tmp/glmserve_pp_tiny_${SLURM_JOB_ID}"
PY="${GLMSERVE_PYTHON:-python3}"
echo "=== generate tiny checkpoint -> $CKPT ==="
"$PY" tools/make_tiny_checkpoint.py --out "$CKPT"
test -f "$CKPT/model.safetensors" || { echo "FAIL: tiny checkpoint not generated"; exit 1; }

export GLMSERVE_TINY_CKPT="$CKPT"
export GLMSERVE_TP_SIZE=1
export GLMSERVE_PP_SIZE=2
export GLMSERVE_PP_GPU_FORWARD="${GLMSERVE_PP_GPU_FORWARD:-1}"
export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_ppfwd_${SLURM_JOB_ID}.id"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
rm -f "$GLMSERVE_NCCL_ID_FILE"

echo "=== 2-rank PP=2 GPU forward correctness (pp-split logits == single-process) ==="
srun --ntasks=2 bash -lc \
  'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exec build/gpu/tests/test_pp_forward'

rm -f "$GLMSERVE_NCCL_ID_FILE"
rm -rf "$CKPT"
echo "=== pp_forward_smoke done ==="
