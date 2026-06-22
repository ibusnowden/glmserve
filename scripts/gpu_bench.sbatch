#!/usr/bin/env bash
#SBATCH --job-name=glmserve-bench
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/project/inniang/glmserve/scripts/gpu_bench.%j.out
#
# Run the RTX GPU correctness/throughput benchmark. Submit with:
#   sbatch scripts/gpu_bench.sbatch
set -e
cd /project/inniang/glmserve
exec bash scripts/gpu_bench.sh
