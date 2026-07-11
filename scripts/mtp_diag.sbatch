#!/usr/bin/env bash
#SBATCH --job-name=glmserve-mtp-diag
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/project/inniang/glmserve/scripts/mtp_diag.%j.out
#
# Real-model MTP spec-decode diagnostic. Phase F (tokgen --gpu --mtp-draft-k)
# diverges from phase D (plain greedy) at token 4 (spec = greedy shifted forward
# ~19). The tiny checkpoint passes TP=8 (both run() and mtpcheck GPU paths), so
# the bug is real-model-specific (GGUF 3-bit, 78 layers, hidden 6144, vocab
# 154880, absorbed MLA n>1). mtpcheck --gpu prints proposed (draft) vs target
# (trunk verify) vs greedy, isolating draft vs verify vs position.
#
#   GLMSERVE_PHASES=M sbatch scripts/mtp_diag.sbatch   # default: just mtpcheck (1 cold load)
#   GLMSERVE_PHASES=DFM sbatch scripts/mtp_diag.sbatch  # full: greedy + spec + mtpcheck
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

GGUF="${GLMSERVE_GGUF_PATH:-/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL}"
PHASES="${GLMSERVE_PHASES:-M}"
DRAFT_K="${GLMSERVE_DRAFT_K:-4}"
SEM_TOKENS="${GLMSERVE_TOKGEN_TOKENS:-151331 785 6722 315 9621 374}"

echo "=== $(date -Iseconds) glmserve MTP diag on $(hostname -s) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "GGUF=$GGUF phases=$PHASES draft_k=$DRAFT_K SEM_TOKENS=$SEM_TOKENS"

if [ ! -f "$GGUF/$(ls "$GGUF" | grep -m1 '00001-of')" ]; then
    echo "FAIL: GGUF first shard not found in $GGUF"; exit 2
fi

JOBS="${GLMSERVE_BUILD_JOBS:-8}"
echo "=== build GPU=1 ==="
make GPU=1 -j"$JOBS"

BIN="/tmp/glmserve_bindiag_${SLURM_JOB_ID}"
cp build/glmserve "$BIN"
chmod +x "$BIN"
echo "staged binary -> $BIN"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export GLMSERVE_NO_PREFAULT="${GLMSERVE_NO_PREFAULT:-1}"
export GLMSERVE_LOG="${GLMSERVE_LOG:-info}"

# Node monitor: page cache + GPU memory every 60 s so a stalled cold load is diagnosable.
(
  while sleep 60; do
    echo "[monitor] $(date +%H:%M:%S) $(awk '/MemAvailable|^Cached/{printf "%s=%.1fG ", $1, $2/1048576}' /proc/meminfo)gpu_used_MiB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)"
  done
) &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

run_tp8() {
    export GLMSERVE_TP_SIZE=8
    export GLMSERVE_PP_SIZE=1
    export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_diag_${SLURM_JOB_ID}.id"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
    srun --ntasks=8 --ntasks-per-node=8 "$@"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
}

TOKS_D="/tmp/glmserve_diagD_${SLURM_JOB_ID}.tokens"

if [[ "$PHASES" == *D* ]]; then
    echo "=== phase D: TP=8 full-model plain greedy tokgen (reference) ==="
    run_tp8 "$BIN" tokgen --model "$GGUF" --gpu --ctx 256 \
        --tokens "$SEM_TOKENS" --max-tokens 32 --temp 0 \
        | tee "$TOKS_D.full"
    grep -m1 "tokens:" "$TOKS_D.full" > "$TOKS_D" || true
fi

if [[ "$PHASES" == *F* ]]; then
    echo "=== phase F: TP=8 full-model spec tokgen (run() path, must match D) ==="
    TOKS_F="/tmp/glmserve_diagF_${SLURM_JOB_ID}.tokens"
    run_tp8 "$BIN" tokgen --model "$GGUF" --gpu --ctx 256 \
        --tokens "$SEM_TOKENS" --max-tokens 32 --temp 0 \
        --mtp-draft-k "$DRAFT_K" \
        | tee "$TOKS_F.full"
    grep -m1 "tokens:" "$TOKS_F.full" > "$TOKS_F" || true
    if [ -s "$TOKS_D" ] && [ -s "$TOKS_F" ]; then
        if diff -q "$TOKS_D" "$TOKS_F" >/dev/null; then
            echo "phase D==F spec token gate: PASS"
        else
            echo "phase D==F spec token gate: FAIL"
            echo "D: $(cat "$TOKS_D")"; echo "F: $(cat "$TOKS_F")"
        fi
    fi
fi

if [[ "$PHASES" == *M* ]]; then
    echo "=== phase M: TP=8 full-model mtpcheck --gpu (proposed/target/output/greedy) ==="
    run_tp8 "$BIN" mtpcheck --model "$GGUF" --gpu --ctx 256 \
        --tokens "$SEM_TOKENS" --gen 32 --draft-k "$DRAFT_K"
    echo "mtpcheck_rc=${PIPESTATUS[0]:-$?}"
fi

echo "=== mtp_diag done $(date -Iseconds) ==="
