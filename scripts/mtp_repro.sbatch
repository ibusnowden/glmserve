#!/usr/bin/env bash
#SBATCH --job-name=glmserve-mtp-repro
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/project/inniang/glmserve/scripts/mtp_repro.%j.out
#
# Isolate the GPU MTP spec-decode divergence. mtpcheck (check_mtp_speculative)
# PASSES on tiny TP=8; the real-model phase F uses tokgen --gpu (the run()
# path). Test tokgen --gpu (run() GPU path) vs mtpcheck on the tiny checkpoint.
set -uo pipefail
cd /project/inniang/glmserve
source scripts/env.sh
export GLMSERVE_REQUIRE_CUDA=1
export GLMSERVE_LOG=error
export NCCL_P2P_LEVEL=SYS

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
JOBS="${GLMSERVE_BUILD_JOBS:-8}"
make GPU=1 -j"$JOBS"

PY="${GLMSERVE_PYTHON:-python3}"
CKPT="/tmp/glmserve_mtp_tiny_${SLURM_JOB_ID}"
"$PY" tools/make_tiny_checkpoint.py --out "$CKPT" --index-topk 2048 --heads 8 >/dev/null
echo "tiny checkpoint: DSA off (index_topk=2048), 8 heads"

IDF="/tmp/glmserve_mtp_${SLURM_JOB_ID}.id"
export GLMSERVE_NCCL_ID_FILE="$IDF"

# TP=1 (single process): tokgen --gpu (run() GPU path) vs mtpcheck --gpu.
echo "=== TP=1 tokgen --gpu plain greedy (reference) ==="
timeout 120 env GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 build/glmserve tokgen --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --max-tokens 24 --temp 0 --gpu
echo "tokgen_greedy_tp1_rc=$?"

echo "=== TP=1 tokgen --gpu --mtp-draft-k 4 (run() GPU path, must match greedy) ==="
timeout 120 env GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 build/glmserve tokgen --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --max-tokens 24 --temp 0 --gpu --mtp-draft-k 4
echo "tokgen_spec_tp1_rc=$?"

echo "=== TP=1 mtpcheck --gpu (control) ==="
timeout 120 env GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 build/glmserve mtpcheck --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --gen 24 --draft-k 4 --gpu
echo "mtpcheck_tp1_rc=$?"

# TP=8: tokgen --gpu (run() GPU path) vs mtpcheck --gpu.
echo "=== TP=8 tokgen --gpu plain greedy (reference) ==="
rm -f "$IDF"; timeout 180 srun -l --ntasks=8 env GLMSERVE_TP_SIZE=8 GLMSERVE_PP_SIZE=1 build/glmserve tokgen --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --max-tokens 24 --temp 0 --gpu
echo "tokgen_greedy_tp8_rc=$?"

echo "=== TP=8 tokgen --gpu --mtp-draft-k 4 (run() GPU spec path, must match greedy) ==="
rm -f "$IDF"; timeout 180 srun -l --ntasks=8 env GLMSERVE_TP_SIZE=8 GLMSERVE_PP_SIZE=1 build/glmserve tokgen --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --max-tokens 24 --temp 0 --gpu --mtp-draft-k 4
echo "tokgen_spec_tp8_rc=$?"

echo "=== TP=8 mtpcheck --gpu --draft-k 4 (control) ==="
rm -f "$IDF"; timeout 180 srun -l --ntasks=8 env GLMSERVE_TP_SIZE=8 GLMSERVE_PP_SIZE=1 build/glmserve mtpcheck --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --gen 24 --draft-k 4 --gpu
echo "mtpcheck_tp8_rc=$?"

echo "=== mtp_repro done ==="
