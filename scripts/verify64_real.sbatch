#!/usr/bin/env bash
#SBATCH --job-name=glmserve-v64-real
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/project/inniang/glmserve/scripts/verify64_real.%j.out

# Full-model correctness gate for the repetitive lookup-draft benchmark.
# mtpcheck generates the same continuation once with serial greedy decode and
# once with 64-token n-gram drafts plus batched trunk verification, then fails
# unless the two token streams are identical.
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

GGUF="${GLMSERVE_GGUF_PATH:-/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL}"
PROMPT_FILE="${GLMSERVE_PROMPT_FILE:-bench/prompts/repetitive_1024.ids}"
GEN_LEN="${GLMSERVE_GEN_LEN:-128}"
DRAFT_K="${GLMSERVE_DRAFT_K:-64}"
BIN_SOURCE="${GLMSERVE_BIN_PATH:-build/glmserve}"
BIN="/tmp/glmserve_verify64_${SLURM_JOB_ID}"

cp "$BIN_SOURCE" "$BIN"
chmod +x "$BIN"
PROMPT_IDS="$(tr '\n' ' ' < "$PROMPT_FILE")"

echo "=== $(date -Iseconds) glmserve real-model verify64 on $(hostname -s) ==="
echo "GGUF=$GGUF prompt=$PROMPT_FILE gen_len=$GEN_LEN draft_k=$DRAFT_K bin=$BIN_SOURCE"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export GLMSERVE_NO_PREFAULT="${GLMSERVE_NO_PREFAULT:-1}"
export GLMSERVE_TP_SIZE=8
export GLMSERVE_PP_SIZE=1
export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_verify64_${SLURM_JOB_ID}.id"
rm -f "$GLMSERVE_NCCL_ID_FILE"

srun --ntasks=8 --ntasks-per-node=8 "$BIN" mtpcheck \
    --model "$GGUF" --gpu --ctx 1536 --tokens "$PROMPT_IDS" \
    --gen "$GEN_LEN" --draft-k "$DRAFT_K"

echo "=== verify64 real-model gate passed $(date -Iseconds) ==="
