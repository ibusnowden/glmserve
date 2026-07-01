#!/usr/bin/env bash
#SBATCH --job-name=glmserve-w4
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/project/inniang/glmserve/scripts/w4_gpu_check.%j.out
#
# Submit with:
#   sbatch scripts/w4_gpu_check.sbatch
set -euo pipefail
cd /project/inniang/glmserve
exec bash scripts/w4_gpu_check.sh
