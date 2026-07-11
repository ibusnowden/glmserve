#!/usr/bin/env bash
#SBATCH --job-name=glmserve-qmoe-v
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/inniang/glmserve/scripts/qmoe_verify_bench.%j.out

set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

# Each invocation is a fresh process because qgemm caches its environment
# policy.  Sweep around the verify sizes; cutoff=64 forces MMA for all cases.
for N in 8 16 32 64; do
    GLMSERVE_MOE_TILE=8 build/gpu/bench/bench_qmoe_verify "$N" 10
    build/gpu/bench/bench_qmoe_verify "$N" 10
    GLMSERVE_MMA_MOE_MIN_TS=64 build/gpu/bench/bench_qmoe_verify "$N" 10
done
