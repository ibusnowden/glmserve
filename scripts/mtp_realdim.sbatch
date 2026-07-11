#!/usr/bin/env bash
#SBATCH --job-name=glmserve-realdim
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/project/inniang/glmserve/scripts/mtp_realdim.%j.out
#
# Real-dim synthetic checkpoint: kvlat=512, rope=64, nope=192, vd=256, H=8,
# hidden=6144, 4 layers, small vocab/MoE. Tests the absorbed-MLA n>1 path at
# real GLM-5.2 dimensions without GGUF/slow NFS. mtpcheck --gpu shows
# proposed (draft) vs target (trunk verify) vs greedy. If target != greedy,
# the bug is in the absorbed n>1 attention (dimension-specific), not the GGUF
# quant or the full 78-layer depth.
set -uo pipefail
cd /project/inniang/glmserve
source scripts/env.sh
export GLMSERVE_REQUIRE_CUDA=1
export GLMSERVE_LOG=info
export NCCL_P2P_LEVEL=SYS

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
make GPU=1 -j8 2>&1 | tail -1

CKPT=/project/inniang/glmserve/build/glm_real_dim
echo "=== TP=1 mtpcheck --gpu (real-dim, target vs greedy) ==="
timeout 120 env GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 build/glmserve mtpcheck \
    --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --gen 24 --draft-k 4 --gpu
echo "mtpcheck_rc=$?"

echo "=== TP=1 tokgen --gpu plain greedy (reference) ==="
timeout 120 env GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 build/glmserve tokgen \
    --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --max-tokens 24 --temp 0 --gpu
echo "tokgen_greedy_rc=$?"

echo "=== TP=1 tokgen --gpu --mtp-draft-k 4 (spec, must match greedy) ==="
timeout 120 env GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 build/glmserve tokgen \
    --model "$CKPT" --tokens "3 1 4 1 5 9 2 6" --max-tokens 24 --temp 0 --gpu --mtp-draft-k 4
echo "tokgen_spec_rc=$?"

echo "=== mtp_realdim done ==="
