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
# Phases: A) single-GPU 4-layer tokgen (int8-MMQ off = historic reference,
#            then MMQ on = the new activation-quantized reference)
#         B) TP=8 4-layer tokgen (must produce the same tokens as A/MMQ-on)
#         C) TP=8 full 78-layer bench (the llama.cpp comparison number)
#         D) TP=8 full-model semantic tokgen ("capital of France" coherence)
#         F) TP=8 full-model spec-decode tokgen (--mtp-draft-k; token stream
#            must equal phase D — the real-model speculative-correctness gate)
#         S) TP=8 full-model spec-decode bench (--mtp-draft-k throughput)
#         E) TP=8 long-ctx probe (prompt 8K/16K decode-at-depth, latent cache)
#
# Usage:
#   sbatch scripts/gguf_8x.sbatch                        # A+B+C
#   GLMSERVE_PHASES=CDFSE sbatch scripts/gguf_8x.sbatch
#   GLMSERVE_PROMPT_LEN=2048 GLMSERVE_GEN_LEN=256 sbatch scripts/gguf_8x.sbatch
#
# NOTE each full-model phase re-reads the 343 GB payload cold (~30 min): the
# upload evicts consumed ranges to dodge the page-cache ceiling.
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
if [ "${GLMSERVE_SKIP_BUILD:-0}" = 1 ]; then
    # Caller guarantees build/glmserve is current. Avoids the in-job rebuild
    # racing an interactive login-node build of the same NFS build/ dir
    # (write-write on build/glmserve corrupted the staged binary, job 145319).
    echo "=== build skipped (GLMSERVE_SKIP_BUILD=1) ==="
else
    echo "=== build GPU=1 ==="
    make GPU=1 -j"$JOBS"
fi

# Run from a node-local copy: the NFS build/glmserve is demand-paged, so a
# login-node rebuild mid-job truncates the mapped ELF -> SIGBUS on all ranks
# (killed job 144510 at phase D). /tmp is job-scoped on compute nodes.
BIN="/tmp/glmserve_bin_${SLURM_JOB_ID}"
cp build/glmserve "$BIN"
chmod +x "$BIN"
echo "staged binary -> $BIN"

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

TOKS_A="/tmp/glmserve_phaseA_${SLURM_JOB_ID}.tokens"

if [[ "$PHASES" == *A* ]]; then
    echo "=== phase A: single-GPU 4-layer tokgen, MMQ off (historic reference) ==="
    GLMSERVE_MMQ=0 GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 \
      srun --ntasks=1 --cpus-per-task=8 \
      "$BIN" tokgen --model "$GGUF" --gpu --max-layers 4 \
      --ctx 512 --tokens "$TOKGEN_TOKENS" --max-tokens 16 --temp 0 \
      | tee "$TOKS_A.full"
    grep -m1 "tokens:" "$TOKS_A.full" > "$TOKS_A" || true
    # MMQ-on stream is informational: int8-activation greedy streams
    # legitimately drift between TP splits (all-reduce vs serial fp summation)
    # on this truncated 4-layer stub, so the exact cross-TP gate stays MMQ-off;
    # phases D/F gate MMQ-on on the full model, where margins are real.
    echo "=== phase A: single-GPU 4-layer tokgen, MMQ on (informational) ==="
    GLMSERVE_WORLD_SIZE=1 GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=1 \
      srun --ntasks=1 --cpus-per-task=8 \
      "$BIN" tokgen --model "$GGUF" --gpu --max-layers 4 \
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
      "$BIN" tokgen --model "$GGUF" --gpu --max-layers "$PROBE_LAYERS" \
      --ctx 512 --tokens "$TOKGEN_TOKENS" --max-tokens 4 --temp 0
fi

if [[ "$PHASES" == *B* ]]; then
    echo "=== phase B: TP=8 4-layer tokgen, MMQ off (must match phase A) ==="
    TOKS_B="/tmp/glmserve_phaseB_${SLURM_JOB_ID}.tokens"
    run_tp8 env GLMSERVE_MMQ=0 "$BIN" tokgen --model "$GGUF" --gpu --max-layers 4 \
        --ctx 512 --tokens "$TOKGEN_TOKENS" --max-tokens 16 --temp 0 \
        | tee "$TOKS_B.full"
    grep -m1 "tokens:" "$TOKS_B.full" > "$TOKS_B" || true
    if [ -s "$TOKS_A" ] && [ -s "$TOKS_B" ]; then
        if diff -q "$TOKS_A" "$TOKS_B" >/dev/null; then
            echo "phase A==B token gate: PASS"
        else
            echo "phase A==B token gate: FAIL"
            echo "A: $(cat "$TOKS_A")"; echo "B: $(cat "$TOKS_B")"
        fi
    fi
fi

if [[ "$PHASES" == *C* ]]; then
    echo "=== phase C: TP=8 full-model bench ==="
    ARGS=(--model "$GGUF" --gpu --ctx "$CTX")
    if [ "$MAX_LAYERS" != "-1" ]; then ARGS+=(--max-layers "$MAX_LAYERS"); fi
    run_tp8 "$BIN" bench "${ARGS[@]}" --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN"
fi

SEM_TOKENS="151331 785 6722 315 9621 374"
TOKS_D="/tmp/glmserve_phaseD_${SLURM_JOB_ID}.tokens"

if [[ "$PHASES" == *D* ]]; then
    # "The capital of France is" tokenized by the llama.cpp GLM-5.2 tokenizer
    # (BOS prepended); greedy continuation should name Paris. Detokenize the
    # printed ids via the llama.cpp server /detokenize to read the text.
    echo "=== phase D: TP=8 full-model semantic tokgen ==="
    run_tp8 "$BIN" tokgen --model "$GGUF" --gpu --ctx 256 \
        --tokens "$SEM_TOKENS" --max-tokens 32 --temp 0 \
        | tee "$TOKS_D.full"
    grep -m1 "tokens:" "$TOKS_D.full" > "$TOKS_D" || true
fi

if [[ "$PHASES" == *F* ]]; then
    # Speculative-correctness gate on the REAL model: the MTP-drafted greedy
    # stream must be byte-identical to phase D's plain greedy stream.
    echo "=== phase F: TP=8 spec-decode tokgen (must match phase D) ==="
    TOKS_F="/tmp/glmserve_phaseF_${SLURM_JOB_ID}.tokens"
    run_tp8 "$BIN" tokgen --model "$GGUF" --gpu --ctx 256 \
        --tokens "$SEM_TOKENS" --max-tokens 32 --temp 0 \
        --mtp-draft-k "${GLMSERVE_DRAFT_K:-4}" \
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

if [[ "$PHASES" == *S* ]]; then
    # GLMSERVE_DRAFT_KS="4 6" sweeps acceptance-vs-depth; NOTE each draft_k
    # re-uploads the model cold (~10 min) — keep the list short.
    for K in ${GLMSERVE_DRAFT_KS:-${GLMSERVE_DRAFT_K:-4}}; do
        echo "=== phase S: TP=8 spec-decode bench (draft_k=$K) ==="
        run_tp8 "$BIN" bench --model "$GGUF" --gpu --ctx "$CTX" \
            --prompt-len "$PROMPT_LEN" --gen-len "$GEN_LEN" \
            --mtp-draft-k "$K"
    done
fi

if [[ "$PHASES" == *R* ]]; then
    # Real-prompt spec-decode bench: acceptance is meaningless on the synthetic
    # pseudo-random prompt (phase S falls back after the probe window), so this
    # is the number that answers llama.cpp's ngram-mod serving throughput.
    # diverse = README text (honest case), repetitive = looped paragraph
    # (llama.cpp measures ~120 tok/s here via 100% ngram acceptance).
    for PF in ${GLMSERVE_PROMPT_FILES:-bench/prompts/diverse_1024.ids bench/prompts/repetitive_1024.ids}; do
        echo "=== phase R: TP=8 real-prompt spec bench ($PF, draft_k=${GLMSERVE_DRAFT_K:-4}) ==="
        run_tp8 "$BIN" bench --model "$GGUF" --gpu --ctx "$CTX" \
            --prompt-file "$PF" --gen-len "$GEN_LEN" \
            --mtp-draft-k "${GLMSERVE_DRAFT_K:-4}"
    done
fi

if [[ "$PHASES" == *E* ]]; then
    # Long-context decode-at-depth probe: the fp16 latent cache + fp16 indexer
    # cache keep 16K ctx at ~1.7 GiB/rank; decode beyond index_topk=2048 runs
    # the learned DSA selector (parallel score + radix top-k) every token.
    ELEN="${GLMSERVE_LONGCTX_PROMPT:-16000}"
    echo "=== phase E: TP=8 long-ctx probe (prompt=$ELEN gen=128) ==="
    run_tp8 "$BIN" bench --model "$GGUF" --gpu --ctx 16384 \
        --prompt-len "$ELEN" --gen-len 128
fi

echo "=== gguf_8x done $(date -Iseconds) ==="
