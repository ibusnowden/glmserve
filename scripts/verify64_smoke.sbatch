#!/usr/bin/env bash
#SBATCH --job-name=glmserve-v64
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/project/inniang/glmserve/scripts/verify64_smoke.%j.out
#
# Validate the maximum speculative verify width before a full real-model run.
# The chunk checker compares every row's logits and argmax against 64 serial
# incremental decode steps, covering the widened absorbed-MLA parity buffers.
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

TINY="$(mktemp -d)"
BIN="/tmp/glmserve_v64_${SLURM_JOB_ID}"
trap 'rm -rf "$TINY" "$BIN"' EXIT

cp build/glmserve "$BIN"
chmod +x "$BIN"
python3 tools/make_tiny_checkpoint.py --out "$TINY" \
  --prompt "3 1 4 1 5 9 2 6"

GLMSERVE_REQUIRE_CUDA=1 GLMSERVE_LOG=error \
  "$BIN" chunkcheck --model "$TINY" --gpu --ctx 128 \
  --tokens "3 1 4 1 5 9 2 6" --k 64

echo "64-token verify parity smoke: PASS"
