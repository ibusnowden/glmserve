#!/usr/bin/env bash
#SBATCH --job-name=glmserve-car
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --exclude=itiger04
#SBATCH --output=/project/inniang/glmserve/scripts/custom_ar_smoke.%j.out
#
# Custom P2P all-reduce validation on the real TP=8 topology:
#   1) test_nccl_comm x8 with GLMSERVE_CUSTOM_AR=1 (values + parity reuse +
#      NCCL-fallback interleave + latency probe)
#   2) same with GLMSERVE_CUSTOM_AR=0 (NCCL latency baseline)
#   3) test_tp_forward TP=2 with GLMSERVE_CUSTOM_AR=1 (real forward gate)
# Submit with: sbatch scripts/custom_ar_smoke.sbatch
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
JOBS="${GLMSERVE_BUILD_JOBS:-8}"
make GPU=1 -j"$JOBS"
make tests GPU=1 -j"$JOBS"

export GLMSERVE_PP_SIZE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS

run8() {
    export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_car_${SLURM_JOB_ID}_${RANDOM}.id"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
    srun -l --ntasks=8 "$@"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
}

echo "=== test_nccl_comm TP=8, custom AR ON ==="
GLMSERVE_TP_SIZE=8 GLMSERVE_CUSTOM_AR=1 run8 build/gpu/tests/test_nccl_comm

echo "=== test_nccl_comm TP=8, custom AR OFF (NCCL baseline) ==="
GLMSERVE_TP_SIZE=8 GLMSERVE_CUSTOM_AR=0 run8 build/gpu/tests/test_nccl_comm

CKPT="/tmp/glmserve_car_tiny_${SLURM_JOB_ID}"
PY="${GLMSERVE_PYTHON:-python3}"
echo "=== generate tiny checkpoint -> $CKPT ==="
"$PY" tools/make_tiny_checkpoint.py --out "$CKPT"
export GLMSERVE_TINY_CKPT="$CKPT"

echo "=== test_tp_forward TP=2, custom AR ON ==="
export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_car_${SLURM_JOB_ID}_tp2.id"
rm -f "$GLMSERVE_NCCL_ID_FILE"
GLMSERVE_TP_SIZE=2 GLMSERVE_CUSTOM_AR=1 srun -l --ntasks=2 build/gpu/tests/test_tp_forward
rm -f "$GLMSERVE_NCCL_ID_FILE"
rm -rf "$CKPT"
echo "=== custom_ar_smoke done ==="
