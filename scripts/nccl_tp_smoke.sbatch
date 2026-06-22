#!/usr/bin/env bash
#SBATCH --job-name=glmserve-nccl
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/project/inniang/glmserve/scripts/nccl_tp_smoke.%j.out
#
# Build once, then launch 2-rank tensor-parallel NCCL smoke tests.
# Submit with:
#   sbatch scripts/nccl_tp_smoke.sbatch
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

JOBS="${GLMSERVE_BUILD_JOBS:-4}"
make GPU=1 -j"$JOBS"
make tests GPU=1 -j"$JOBS"

export GLMSERVE_TP_SIZE=2
export GLMSERVE_PP_SIZE=1
export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_${SLURM_JOB_ID}.id"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
rm -f "$GLMSERVE_NCCL_ID_FILE"

echo "=== 2-rank TP all-reduce smoke ==="
srun --ntasks=2 bash -lc \
  'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exec build/gpu/tests/test_nccl_comm'

echo "=== 2-rank TP row-linear smoke ==="
rm -f "$GLMSERVE_NCCL_ID_FILE"
srun --ntasks=2 bash -lc \
  'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exec build/gpu/tests/test_tp_linear'

rm -f "$GLMSERVE_NCCL_ID_FILE"
echo "=== nccl_tp_smoke done ==="
