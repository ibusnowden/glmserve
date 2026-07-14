#!/usr/bin/env bash
#SBATCH --job-name=glmserve-dsa-test
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/project/inniang/glmserve/scripts/dsa_gemm_test.%j.out
#
# Fast single-GPU gate for the DSA selector GEMM path (cuda/attention_dsa.cu):
# runs test_attention_dsa, which now exercises the cuBLAS branch of the GEMM
# scoring path (iH=32 -> n=288 > kMlaParityMaxQ -> cublasSgemm) and checks the
# selected indices match the CPU fp32 reference. With CUBLAS_PEDANTIC_MATH the
# GEMM uses true fp32 (no TF32) and must match; a TF32 regression (default
# math mode) would flip top-k and MISMATCH.
set -e
cd /project/inniang/glmserve
source scripts/env.sh
export GLMSERVE_REQUIRE_CUDA=1
nvidia-smi -L
echo "=== rebuild tests (GPU=1) ==="
make tests GPU=1 -j4
echo "=== test_attention_dsa (scalar + gemv + cuBLAS-pedantic paths) ==="
build/gpu/tests/test_attention_dsa
echo "=== test_attention_dsa exit $? ==="
