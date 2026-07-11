#!/usr/bin/env bash
#SBATCH --job-name=glmserve-mtp-diag4
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/project/inniang/glmserve/scripts/mtp_diag4.%j.out
#
# Fast 4-layer mtpcheck --gpu: isolates whether the trunk verify (absorbed
# MLA n>1) is buggy at reduced depth. --max-layers 4 loads 4 trunk layers +
# the MTP block (layer 78). The MTP draft is garbage (mismatched trunk), but
# the `target` (trunk verify) and `greedy` (plain decode) are self-consistent
# for the 4-layer trunk. If target != greedy, the absorbed n>1 attention is
# buggy (trunk), independent of the MTP. If target == greedy, the bug needs
# the full 74 layers or is MTP-specific.
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

GGUF="${GLMSERVE_GGUF_PATH:-/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL}"
DRAFT_K="${GLMSERVE_DRAFT_K:-4}"
SEM_TOKENS="${GLMSERVE_TOKGEN_TOKENS:-151331 785 6722 315 9621 374}"

echo "=== $(date -Iseconds) glmserve MTP diag4 (4 layers) on $(hostname -s) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "GGUF=$GGUF draft_k=$DRAFT_K SEM_TOKENS=$SEM_TOKENS"

if [ ! -f "$GGUF/$(ls "$GGUF" | grep -m1 '00001-of')" ]; then
    echo "FAIL: GGUF first shard not found in $GGUF"; exit 2
fi

JOBS="${GLMSERVE_BUILD_JOBS:-8}"
echo "=== build GPU=1 ==="
make GPU=1 -j"$JOBS"

BIN="/tmp/glmserve_bindiag4_${SLURM_JOB_ID}"
cp build/glmserve "$BIN"
chmod +x "$BIN"
echo "staged binary -> $BIN"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export GLMSERVE_NO_PREFAULT="${GLMSERVE_NO_PREFAULT:-1}"
export GLMSERVE_LOG="${GLMSERVE_LOG:-info}"

run_tp8() {
    export GLMSERVE_TP_SIZE=8
    export GLMSERVE_PP_SIZE=1
    export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_diag4_${SLURM_JOB_ID}.id"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
    srun --ntasks=8 --ntasks-per-node=8 "$@"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
}

echo "=== phase M8: TP=8 4-layer mtpcheck --gpu (target vs greedy) ==="
run_tp8 "$BIN" mtpcheck --model "$GGUF" --gpu --ctx 256 --max-layers 4 \
    --tokens "$SEM_TOKENS" --gen 32 --draft-k "$DRAFT_K"
echo "mtpcheck_rc=${PIPESTATUS[0]:-$?}"

echo "=== mtp_diag4 done $(date -Iseconds) ==="
