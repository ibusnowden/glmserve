#!/usr/bin/env bash
#SBATCH --job-name=glmserve-gpu
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/project/inniang/glmserve/scripts/gpu_check.%j.out
#
# Build glmserve with CUDA on a GPU node and validate the full device forward
# (model_gpu.cpp + all kernels) against the numpy MLA reference. Submit with:
#   sbatch scripts/gpu_check.sbatch
# or run interactively:
#   srun -p bigTiger --gres=gpu:rtx_6000:1 -t 20:00 bash scripts/gpu_check.sh
set -e
cd /project/inniang/glmserve
exec bash scripts/gpu_check.sh
