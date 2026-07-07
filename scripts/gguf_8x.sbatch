#!/usr/bin/env bash
#SBATCH --job-name=glmserve-gguf-8x
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/project/inniang/glmserve/scripts/gguf_8x.%j.out
#
# Serve/bench the real GLM-5.2 3-bit GGUF (UD-Q3_K_XL) on one 8x RTX 6000 Ada
# node with glmserve's native GGUF quant path, tensor-parallel across all 8 GPUs
# (TP=8, PP=1). Each rank prefaults its stripe of the payload ranges, mmaps the
# GGUF, uploads its 1/8 weight shard (strided cudaMemcpy2D, no host copies), and
# runs the dequant-on-the-fly forward (qgemm/moe_expert_ffn_q) in lockstep;
# rank 0 prints throughput.
#
# Phases: A) single-GPU 4-layer tokgen (reference tokens)
#         B) TP=8 4-layer tokgen (must produce the same tokens as A)
#         C) TP=8 full 78-layer bench (the llama.cpp comparison number)
#
# Usage:
#   sbatch scripts/gguf_8x.sbatch                        # A+B+C
#   GLMSERVE_PHASES=C sbatch scripts/gguf_8x.sbatch      # bench only
#   GLMSERVE_PROMPT_LEN=2048 GLMSERVE_GEN_LEN=256 sbatch scripts/gguf_8x.sbatch
#
# Env overrides:
#   GLMSERVE_GGUF_PATH   (default: the inference UD-Q3_K_XL split)
#   GLMSERVE_MAX_LAYERS  (default: -1 = all 78 layers, phase C)
#   GLMSERVE_CTX         (default: 1536)
#   GLMSERVE_PROMPT_LEN  (default: 1024)
#   GLMSERVE_GEN_LEN     (default: 128)
#   GLMSERVE_PHASES      (default: ABC; any subset, e.g. C)
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

GGUF="${GLMSERVE_GGUF_PATH:-/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL}"
MAX_LAYERS="${GLMSERVE_MAX_LAYERS:--1}"
CTX="${GLMSERVE_CTX:-1536}"
PROMPT_LEN="${GLMSERVE_PROMPT_LEN:-1024}"
GEN_LEN="${GLMSERVE_GEN_LEN:-128}"
PHASES="${GLMSERVE_PHASES:-ABC}"
TOKGEN_TOKENS="${GLMSERVE_TOKGEN_TOKENS:-151331 151331 198 26287}"

echo "=== $(date -Iseconds) glmserve GGUF 8x on $(hostname -s) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "GGUF=$GGUF phases=$PHASES max_layers=$MAX_LAYERS ctx=$CTX prompt_len=$PROMPT_LEN gen_len=$GEN_LEN"

if [ ! -f "$GGUF/$(ls "$GGUF" | grep -m1 '00001-of')" ]; then
    echo "FAIL: GGUF first shard not found in $GGUF"; exit 2
fi

JOBS="${GLMSERVE_BUILD_JOBS:-8}"
echo "=== build GPU=1 ==="
make GPU=1 -j"$JOBS"

# Node monitor: page cache + GPU memory every 60 s, so a stalled phase is
# diagnosable from the log (cache eviction -> crawling reads vs frozen = hang).
(
  while sleep 60; do
    echo "[monitor] $(date +%H:%M:%S) $(awk '/MemAvailable|^Cached/{printf "%s=%.1fG ", $1, $2/1048576}' /proc/meminfo)gpu_used_MiB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)"
    { ps -C glmserve -o pid=,state=,%cpu=,rss=,min_flt=,maj_flt= 2>/dev/null || true; } | \
      awk '{printf "[monitor]   pid=%s st=%s cpu=%s rss=%.1fG minflt=%s majflt=%s\n", $1, $2, $3, $4/1048576, $5, $6}'
  done
) &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# P2P across the root complex: these GPUs sit on different PCI root ports, so
# NCCL's default NCCL_P2P_LEVEL rejects P2P and silently falls back to SHM —
# 48.7 ms per 25 MB fp32 all-reduce (56% of prefill!), identical with
# NCCL_P2P_DISABLE=0. Forcing P2P_LEVEL=SYS selects P2P/CUMEM: 9.3x faster
# all-reduces (job 143998: prefill 75.6 -> 151.5 tok/s), and the upload stays
# NFS-bound (CUMEM lazy connect avoids the legacy peer-access cudaMalloc
# penalty that motivated the old NCCL_P2P_DISABLE=1 default).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
# Skip the 20-min up-front prefault: upload_to_gpu prefetches its mmap source
# a few layers ahead (GLMSERVE_UPLOAD_PREFETCH) and evicts consumed ranges a
# few layers behind (GLMSERVE_EVICT_LAG) — a single sliding ~50 GiB window, so
# the 343 GB payload never hits the 515 GB node's page-cache ceiling and the
# reclaim churn spiral (jobs 143730..143840) cannot start. Note the eviction
# means every phase re-reads the payload cold from NFS (~30 min at ~200 MB/s).
export GLMSERVE_NO_PREFAULT="${GLMSERVE_NO_PREFAULT:-1}"

run_tp8() {
    export GLMSERVE_TP_SIZE=8
    export GLMSERVE_PP_SIZE=1
    export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_gguf8x_${SLURM_JOB_ID}.id"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
    srun --ntasks=8 --ntasks-per-node=8 "$@"
    rm -f "$GLMSERVE_NCCL_ID_FILE"
}

if [[ "$PHASES" == *A* ]]; then
    echo "=== phase A: single-GPU 4-layer tokgen (reference) ==="
    GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 \
      srun --ntasks=1 --cpus-per-task=8 \
      build/glmserve tokgen --model "$GGUF" --gpu --max-layers 4 \
      --ctx 512 --tokens "$TOKGEN_TOKENS" --max-tokens 16 --temp 0
fi

if [[ "$PHASES" == *P* ]]; then
    # Upload-scaling probe: single GPU, no NCCL, enough layers (~4 GiB each
    # unsharded) to cross the ~14-16 GiB mark where the TP=8 upload collapses.
    # If this also collapses, the slowdown is driver/allocator scaling, not
    # NCCL/multi-process interaction.
    PROBE_LAYERS="${GLMSERVE_PROBE_LAYERS:-8}"
    echo "=== phase P: single-GPU ${PROBE_LAYERS}-layer upload-scaling probe ==="
    GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 \
      srun --ntasks=1 --cpus-per-task=8 \
      build/glmserve tokgen --model "$GGUF" --gpu --max-layers "$PROBE_LAYERS" \
      --ctx 512 --tokens "$TOKGEN_TOKENS" --max-tokens 4 --temp 0
fi

if [[ "$PHASES" == *B* ]]; then
    echo "=== phase B: TP=8 4-layer tokgen (must match phase A) ==="
    run_tp8 build/glmserve tokgen --model "$GGUF" --gpu --max-layers 4 \
        --ctx 512 --tokens "$TOKGEN_TOKENS" --max-tokens 16 --temp 0
fi

if [[ "$PHASES" == *C* ]]; then
    echo "=== phase C: TP=8 full-model bench ==="
    ARGS=(--model "$GGUF" --gpu --ctx "$CTX")
    if [ "$MAX_LAYERS" != "-1" ]; then ARGS+=(--max-layers "$MAX_LAYERS"); fi
    run_tp8 build/glmserve bench "${ARGS[@]}" --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN"
fi

if [[ "$PHASES" == *D* ]]; then
    # "The capital of France is" tokenized by the llama.cpp GLM-5.2 tokenizer
    # (BOS prepended); greedy continuation should name Paris. Detokenize the
    # printed ids via the llama.cpp server /detokenize to read the text.
    echo "=== phase D: TP=8 full-model semantic tokgen ==="
    run_tp8 build/glmserve tokgen --model "$GGUF" --gpu --ctx 256 \
        --tokens "151331 785 6722 315 9621 374" --max-tokens 32 --temp 0
fi

echo "=== gguf_8x done $(date -Iseconds) ==="
