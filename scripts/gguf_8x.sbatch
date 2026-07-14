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
# (TP=8, PP=1). Each rank mmaps the shared payload, parallel-packs its 1/8
# quantized weight shard into reusable host staging, bulk-uploads it, and runs
# the dequant-on-the-fly forward (qgemm/moe_expert_ffn_q) in lockstep; rank 0
# prints throughput.
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
# Full phases reuse the shared GGUF page cache by default. Use one phase per job
# for stable timing; GLMSERVE_STAGE_LOCAL=1 adds a persistent node-local NVMe
# cache when the node's NFS client is the bottleneck.
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

# Optional node-local cache. The 319-GiB GGUF is read through NFS in a highly
# strided order by TP ranks; on some nodes that path collapses to ~14 MB/s.
# A one-time sequential copy to the 14-TB local /tmp NVMe turns every later
# restart into a local-file upload. The marker is written only after all nine
# shards are complete, so a killed copy cannot be mistaken for a valid cache.
if [ "${GLMSERVE_STAGE_LOCAL:-0}" = 1 ]; then
    LOCAL_ROOT="${GLMSERVE_LOCAL_ROOT:-/tmp/glmserve_gguf_cache}"
    LOCAL_NAME="$(basename "$GGUF")"
    LOCAL_GGUF="$LOCAL_ROOT/$LOCAL_NAME"
    MARKER="$LOCAL_GGUF/.complete"
    if [ -f "$MARKER" ] && [ -f "$LOCAL_GGUF/$(ls "$LOCAL_GGUF" | grep -m1 '00001-of')" ]; then
        echo "=== reusing node-local GGUF cache: $LOCAL_GGUF ==="
    else
        PARTIAL="$LOCAL_GGUF.partial.$$"
        rm -rf "$PARTIAL"
        mkdir -p "$PARTIAL"
        # Parallel per-shard copies: a single sequential cp is capped by the
        # per-stream NFS read rate (measured 8.5 MB/s sustained on itiger03 =
        # 10+ h for the full payload), while eight concurrent streams reach the
        # client's aggregate ceiling (~66-112 MB/s).
        echo "=== staging GGUF to local NVMe (parallel): $GGUF -> $LOCAL_GGUF ==="
        STAGE_T0=$SECONDS
        for SHARD in "$GGUF"/*.gguf; do
            echo "[stage] $(basename "$SHARD")"
            cp "$SHARD" "$PARTIAL/" &
        done
        wait
        echo "[stage] all shards copied in $((SECONDS - STAGE_T0)) s"
        for SHARD in "$GGUF"/*.gguf; do
            SRC_B=$(stat -c %s "$SHARD"); DST_B=$(stat -c %s "$PARTIAL/$(basename "$SHARD")")
            if [ "$SRC_B" != "$DST_B" ]; then
                echo "FAIL: staged $(basename "$SHARD") size $DST_B != source $SRC_B"; exit 3
            fi
        done
        touch "$PARTIAL/.complete"
        rm -rf "$LOCAL_GGUF"
        mv "$PARTIAL" "$LOCAL_GGUF"
        echo "=== local GGUF stage complete ==="
    fi
    GGUF="$LOCAL_GGUF"
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
BIN_SOURCE="${GLMSERVE_BIN_PATH:-build/glmserve}"
cp "$BIN_SOURCE" "$BIN"
chmod +x "$BIN"
echo "staged binary $BIN_SOURCE -> $BIN"

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
# Skip the up-front whole-model prefault. upload_to_gpu touches a few layers
# ahead (GLMSERVE_UPLOAD_PREFETCH), while the default no-eviction policy keeps
# those pages available to all TP ranks and later server restarts. Nodes under
# unusual memory pressure can opt back into a bounded window with
# GLMSERVE_EVICT_LAG=8.
export GLMSERVE_NO_PREFAULT="${GLMSERVE_NO_PREFAULT:-1}"
# Keep the shared GGUF page cache across requests/restarts. Set
# GLMSERVE_EVICT_LAG=8 for the old bounded-window cold-stream behavior, or
# GLMSERVE_EVICT_ALL=1 GLMSERVE_EVICT_LAG=-1 for an explicit cold-start A/B.
export GLMSERVE_EVICT_LAG="${GLMSERVE_EVICT_LAG:--1}"

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
    # Confidence-gated arm: truncated drafts must not change the greedy stream,
    # and under TP this exercises the sharded top-1-prob combine (the gate
    # decision must be identical on every rank or the all-reduces deadlock).
    echo "=== phase F2: TP=8 spec-decode tokgen, GLMSERVE_SPEC_CONF=0.6 (must match phase D) ==="
    TOKS_F2="/tmp/glmserve_phaseF2_${SLURM_JOB_ID}.tokens"
    run_tp8 env GLMSERVE_SPEC_CONF=0.6 "$BIN" tokgen --model "$GGUF" --gpu --ctx 256 \
        --tokens "$SEM_TOKENS" --max-tokens 32 --temp 0 \
        --mtp-draft-k "${GLMSERVE_DRAFT_K:-4}" \
        | tee "$TOKS_F2.full"
    grep -m1 "tokens:" "$TOKS_F2.full" > "$TOKS_F2" || true
    if [ -s "$TOKS_D" ] && [ -s "$TOKS_F2" ]; then
        if diff -q "$TOKS_D" "$TOKS_F2" >/dev/null; then
            echo "phase D==F2 conf-gated spec token gate: PASS"
        else
            echo "phase D==F2 conf-gated spec token gate: FAIL"
            echo "D: $(cat "$TOKS_D")"; echo "F2: $(cat "$TOKS_F2")"
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
    # (llama.cpp best-of-two is 126.22 tok/s here with 100% acceptance).
    for PF in ${GLMSERVE_PROMPT_FILES:-bench/prompts/diverse_1024.ids bench/prompts/repetitive_1024.ids}; do
        # MTP remains most efficient at a short depth on unmatched prose. A
        # repetitive prompt can fill a 64-token lookup draft and amortize one
        # trunk pass across the entire group, matching llama.cpp's ngram-mod
        # operating point. GLMSERVE_DRAFT_K still overrides both cases.
        RK="${GLMSERVE_DRAFT_K:-4}"
        if [[ -z "${GLMSERVE_DRAFT_K:-}" && "$PF" == *repetitive* ]]; then RK=64; fi
        # Diverse prose sweeps the draft-confidence gate (GLMSERVE_SPEC_CONF):
        # conf=0 disables the adaptive fallback so every group drafts — that arm
        # is the acceptance-histogram calibration, not a throughput claim. The
        # repetitive gate arm stays at conf=0 (ngram lookup drafts, gate moot).
        CONFS="0"
        if [[ "$PF" != *repetitive* ]]; then CONFS="${GLMSERVE_SPEC_CONF_SWEEP:-0 0.4 0.6}"; fi
        for CONF in $CONFS; do
        CONF_ENV=(GLMSERVE_SPEC_CONF="$CONF")
        if [[ "$CONF" == 0 && "$PF" != *repetitive* ]]; then CONF_ENV+=(GLMSERVE_SPEC_ADAPTIVE=0); fi
        echo "=== phase R: TP=8 real-prompt spec bench ($PF, draft_k=$RK, conf=$CONF) ==="
        RLOG="/tmp/glmserve_phaseR_${SLURM_JOB_ID}_$(basename "$PF")_conf${CONF}.log"
        run_tp8 env "${CONF_ENV[@]}" "$BIN" bench --model "$GGUF" --gpu --ctx "$CTX" \
            --prompt-file "$PF" --gen-len "$GEN_LEN" \
            --mtp-draft-k "$RK" | tee "$RLOG"
        # The exact workload used for the llama.cpp ngram-mod comparison is a
        # regression gate, not just an informational number.
        if [[ "$PF" == *repetitive_1024.ids && "$RK" == 64 && "$GEN_LEN" == 128 ]]; then
            TPS="$(awk '$1 == "decode" { print $9 }' "$RLOG" | tail -n1)"
            if [[ -z "$TPS" ]] || ! awk -v got="$TPS" 'BEGIN { exit !(got > 126.22) }'; then
                echo "phase R llama.cpp throughput gate: FAIL (glmserve=${TPS:-missing}, reference=126.22 tok/s)"
                exit 1
            fi
            if ! grep -Eq 'speculative[[:space:]]*: 2 groups \(2 ngram\), 64\.00 tok/group accepted' "$RLOG"; then
                echo "phase R llama.cpp acceptance gate: FAIL"
                exit 1
            fi
            echo "phase R llama.cpp gate: PASS (${TPS} > 126.22 tok/s, 2x64 accepted)"
        fi
        done
    done
fi

if [[ "$PHASES" == *M* ]]; then
    # mma real-model correctness gate: greedy continuation of a real
    # 1024-token prompt with the tensor-core prefill on vs off must produce
    # the same tokens (phases D/F use 6-token prompts and never reach the
    # n>=64 mma engagement, so they don't cover it). Tolerance-free: mma
    # changes fp accumulation, so we gate on the greedy TOKEN stream, which
    # should survive rounding-level logit shifts; a mismatch beyond the last
    # few tokens means a real bug.
    PF_M="${GLMSERVE_MMA_GATE_PROMPT:-bench/prompts/diverse_1024.ids}"
    TOKS_M0="/tmp/glmserve_phaseM0_${SLURM_JOB_ID}.tokens"
    TOKS_M1="/tmp/glmserve_phaseM1_${SLURM_JOB_ID}.tokens"
    echo "=== phase M: TP=8 mma-vs-dp4a real-prompt token gate ($PF_M) ==="
    run_tp8 env GLMSERVE_MMA=0 GLMSERVE_SMEM=0 "$BIN" tokgen --model "$GGUF" --gpu \
        --ctx 1280 --tokens-file "$PF_M" --max-tokens 64 --temp 0 \
        | tee "$TOKS_M0.full"
    grep -m1 "tokens:" "$TOKS_M0.full" > "$TOKS_M0" || true
    run_tp8 env GLMSERVE_MMA=1 "$BIN" tokgen --model "$GGUF" --gpu \
        --ctx 1280 --tokens-file "$PF_M" --max-tokens 64 --temp 0 \
        | tee "$TOKS_M1.full"
    grep -m1 "tokens:" "$TOKS_M1.full" > "$TOKS_M1" || true
    if [ -s "$TOKS_M0" ] && [ -s "$TOKS_M1" ]; then
        if diff -q "$TOKS_M0" "$TOKS_M1" >/dev/null; then
            echo "phase M mma token gate: PASS"
        else
            echo "phase M mma token gate: FAIL"
            echo "M0: $(cat "$TOKS_M0")"; echo "M1: $(cat "$TOKS_M1")"
        fi
    fi
fi

if [[ "$PHASES" == *E* ]]; then
    # Long-context decode-at-depth bench on real prose: the fp16 latent cache +
    # fp16 indexer cache keep 16K ctx at ~1.7 GiB/rank; decode beyond
    # index_topk=2048 runs the learned DSA selector (parallel score + radix
    # top-k) every token. Plain arm is stage-profiled (where does the depth
    # cost land); spec arm is the serving configuration (conf-gated MTP).
    # llama.cpp ngram-mod reference on the same prompts, itiger06 prod,
    # cache_prompt=false, cold ngram cache (2026-07-11): prefill 363 / 320
    # tok/s at 8K/16K, decode ~25.0 / ~25.5 tok/s.
    # VRAM at 16K ctx is tight: weights + latent/indexer KV leave ~0.95 GiB
    # free (job 145677 OOM'd with the default 2048-token scratch arena), so
    # long-ctx arms run a 1024-token prefill chunk (mma still engages: 1024x8
    # dispatch rows) and each arm sizes --ctx to its own prompt. The 16K spec
    # arm additionally carries the resident MTP block (~0.65 GiB/rank) and a
    # 512-token arena, and is guarded: an OOM there must not kill the job.
    ECHUNK="${GLMSERVE_LONGCTX_CHUNK:-1024}"
    for PF in ${GLMSERVE_LONGCTX_FILES:-bench/prompts/diverse_8192.ids bench/prompts/diverse_16384.ids}; do
        NTOK=$(wc -w < "$PF")
        ECTX=$((NTOK + 192))
        echo "=== phase E: TP=8 long-ctx bench ($PF, ctx=$ECTX, plain, profiled) ==="
        run_tp8 env GLMSERVE_PROF=1 GLMSERVE_PREFILL_CHUNK="$ECHUNK" \
            "$BIN" bench --model "$GGUF" --gpu --ctx "$ECTX" \
            --prompt-file "$PF" --gen-len 128
        if [[ "$NTOK" -le 8192 ]]; then
            # Same-binary suffix-width A/B: old 64-wide absorbed suffix chunks.
            echo "=== phase E: TP=8 long-ctx bench ($PF, WIDE_SUFFIX=0 control) ==="
            run_tp8 env GLMSERVE_WIDE_SUFFIX=0 GLMSERVE_PREFILL_CHUNK="$ECHUNK" \
                "$BIN" bench --model "$GGUF" --gpu --ctx "$ECTX" \
                --prompt-file "$PF" --gen-len 128
        fi
        EK="${GLMSERVE_DRAFT_K:-4}"
        SCHUNK="$ECHUNK"
        if [[ "$NTOK" -gt 8192 ]]; then SCHUNK=512; fi
        echo "=== phase E: TP=8 long-ctx bench ($PF, draft_k=$EK, chunk=$SCHUNK) ==="
        run_tp8 env GLMSERVE_PREFILL_CHUNK="$SCHUNK" \
            "$BIN" bench --model "$GGUF" --gpu --ctx "$ECTX" \
            --prompt-file "$PF" --gen-len 128 --mtp-draft-k "$EK" \
            || echo "phase E spec arm ($PF): FAILED (VRAM-tight arm, non-fatal)"
    done
fi

if [[ "$PHASES" == *Y* ]]; then
    # DSA selector GEMM-vs-scalar real-model token gate at 4K depth (the only
    # phases that run with the selector active are long-ctx ones: D/F/M all
    # sit below index_topk=2048). The GEMM path runs in CUBLAS_PEDANTIC_MATH
    # (true fp32, no TF32); greedy continuation must survive the SGEMM
    # summation-order change. A token-1 divergence means a real selection bug
    # (pre-fix: cuBLAS TF32's 10-bit mantissa rotated the per-head dots enough
    # to flip top-k from token 1); a late-token flip is rounding-level noise.
    PF_Y="/tmp/glmserve_phaseY_${SLURM_JOB_ID}.ids"
    awk '{for(i=1;i<=4096;i++) printf "%s ", $i; print ""}' \
        bench/prompts/diverse_8192.ids > "$PF_Y"
    TOKS_Y0="/tmp/glmserve_phaseY0_${SLURM_JOB_ID}.tokens"
    TOKS_Y1="/tmp/glmserve_phaseY1_${SLURM_JOB_ID}.tokens"
    echo "=== phase Y: TP=8 DSA gemm-vs-scalar token gate (4K prompt) ==="
    run_tp8 env GLMSERVE_DSA_GEMM=0 GLMSERVE_PREFILL_CHUNK=1024 \
        "$BIN" tokgen --model "$GGUF" --gpu --ctx 4352 \
        --tokens-file "$PF_Y" --max-tokens 32 --temp 0 | tee "$TOKS_Y0.full"
    grep -m1 "tokens:" "$TOKS_Y0.full" > "$TOKS_Y0" || true
    run_tp8 env GLMSERVE_DSA_GEMM=1 GLMSERVE_PREFILL_CHUNK=1024 \
        "$BIN" tokgen --model "$GGUF" --gpu --ctx 4352 \
        --tokens-file "$PF_Y" --max-tokens 32 --temp 0 | tee "$TOKS_Y1.full"
    grep -m1 "tokens:" "$TOKS_Y1.full" > "$TOKS_Y1" || true
    if [ -s "$TOKS_Y0" ] && [ -s "$TOKS_Y1" ]; then
        if diff -q "$TOKS_Y0" "$TOKS_Y1" >/dev/null; then
            echo "phase Y DSA gemm token gate: PASS"
        else
            echo "phase Y DSA gemm token gate: DIVERGED (inspect: rounding-level"
            echo "selector flips move greedy paths; a mismatch from token 1 = bug)"
            echo "Y0: $(cat "$TOKS_Y0")"; echo "Y1: $(cat "$TOKS_Y1")"
        fi
    fi
fi

echo "=== gguf_8x done $(date -Iseconds) ==="
