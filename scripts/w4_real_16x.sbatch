#!/usr/bin/env bash
#SBATCH --job-name=glmserve-w4-16x
#SBATCH --partition=bigTiger
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/project/inniang/glmserve/scripts/w4_real_16x.%j.out
#
# Real-weight W4A16 distributed load/upload gate for the target topology:
#   TP=8 within each node, PP=2 across nodes, world=16.
#
# Required input:
#   GLMSERVE_REAL_MODEL=/path/to/full/GLM-5.2
#
# Optional:
#   GLMSERVE_W4_MODEL=/path/to/GLM-5.2-w4a16
#   GLMSERVE_W4_MODEL_STAGE0=/path/to/GLM-5.2-w4a16-stage0
#   GLMSERVE_W4_MODEL_STAGE1=/path/to/GLM-5.2-w4a16-stage1
#   GLMSERVE_W4_GROUP=128
#   GLMSERVE_DIST_LOAD_CTX=4096
#   GLMSERVE_FETCH_REPO=zai-org/GLM-5.2   # opt-in; may download ~1.4 TB
#   GLMSERVE_REQUIRE_FREE_GIB=1800        # optional fetch-space guard
#   GLMSERVE_FETCH_JOBS=4                 # optional parallel shard fetch
#   GLMSERVE_VERIFY_PRESENT=1             # optional HEAD size check for existing shards
#
# Prefer fetching on a login/data-transfer node before submitting this GPU job:
#   bash scripts/fetch_glm52_shards.sh --probe 1 --max-files 1
#   bash scripts/fetch_glm52_shards.sh --require-free-gib 1800
set -euo pipefail
cd /project/inniang/glmserve
source scripts/env.sh

REAL_MODEL="${GLMSERVE_REAL_MODEL:-/project/inniang/hf-cache/glm52-real}"
W4_MODEL="${GLMSERVE_W4_MODEL:-/project/inniang/hf-cache/glm52-real-w4a16}"
GROUP="${GLMSERVE_W4_GROUP:-128}"
PY="${GLMSERVE_PYTHON:-python3}"

echo "=== nodes ==="
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "=== GPUs on first node view ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== source checkpoint check: $REAL_MODEL ==="
if [[ -n "${GLMSERVE_FETCH_REPO:-}" ]]; then
  echo "=== fetching missing safetensors shards from $GLMSERVE_FETCH_REPO ==="
  echo "NOTE: if this compute node has no internet, run scripts/fetch_glm52_shards.sh on a login/data-transfer node first."
  FREE_ARGS=()
  if [[ -n "${GLMSERVE_REQUIRE_FREE_GIB:-}" ]]; then
    FREE_ARGS+=(--require-free-gib "$GLMSERVE_REQUIRE_FREE_GIB")
  fi
  FETCH_ARGS=(--jobs "${GLMSERVE_FETCH_JOBS:-1}")
  if [[ "${GLMSERVE_VERIFY_PRESENT:-0}" == "1" ]]; then
    FETCH_ARGS+=(--verify-present)
  fi
  "$PY" tools/fetch_safetensors_shards.py --model "$REAL_MODEL" \
    --repo "$GLMSERVE_FETCH_REPO" --revision "${GLMSERVE_FETCH_REVISION:-main}" \
    "${FREE_ARGS[@]}" "${FETCH_ARGS[@]}"
fi

if ! "$PY" tools/convert_hf_to_glmserve.py --model "$REAL_MODEL" --check; then
  echo "FAIL: source checkpoint is not loadable. The index may be present, but all referenced safetensors shards must exist before the real W4 gate can run."
  echo "To fetch public HF shards explicitly, run:"
  echo "  bash scripts/fetch_glm52_shards.sh --probe 1 --max-files 1"
  echo "  bash scripts/fetch_glm52_shards.sh --require-free-gib 1800"
  echo "or rerun this job with GLMSERVE_FETCH_REPO=zai-org/GLM-5.2 if compute nodes have internet."
  exit 2
fi

if [[ ! -f "$W4_MODEL/model.safetensors.index.json" ]]; then
  echo "=== quantize real checkpoint -> $W4_MODEL (group=$GROUP) ==="
  "$PY" tools/convert_hf_to_glmserve.py --model "$REAL_MODEL" \
    --quantize-int4 "$W4_MODEL" --group "$GROUP"
fi

echo "=== W4 checkpoint check: $W4_MODEL ==="
"$PY" tools/convert_hf_to_glmserve.py --model "$W4_MODEL" --check

echo "=== build GPU tests ==="
JOBS="${GLMSERVE_BUILD_JOBS:-8}"
make GPU=1 -j"$JOBS"
make tests GPU=1 -j"$JOBS"

export GLMSERVE_REQUIRE_CUDA=1
export GLMSERVE_MODEL_CKPT="$W4_MODEL"
if [[ -n "${GLMSERVE_W4_MODEL_STAGE0:-}" ]]; then
  export GLMSERVE_MODEL_CKPT_STAGE0="$GLMSERVE_W4_MODEL_STAGE0"
fi
if [[ -n "${GLMSERVE_W4_MODEL_STAGE1:-}" ]]; then
  export GLMSERVE_MODEL_CKPT_STAGE1="$GLMSERVE_W4_MODEL_STAGE1"
fi
export GLMSERVE_QUANT_ONLY=1
export GLMSERVE_TP_SIZE=8
export GLMSERVE_PP_SIZE=2
export GLMSERVE_DIST_LOAD_CTX="${GLMSERVE_DIST_LOAD_CTX:-4096}"
export GLMSERVE_DIST_LOAD_UPLOAD="${GLMSERVE_DIST_LOAD_UPLOAD:-0}"
export GLMSERVE_NCCL_ID_FILE="/tmp/glmserve_nccl_w4_16x_${SLURM_JOB_ID}.id"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
rm -f "$GLMSERVE_NCCL_ID_FILE"

echo "=== 16-rank TP=8/PP=2 W4 distributed load smoke ==="
srun --ntasks=16 --ntasks-per-node=8 bash -lc \
  'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exec build/gpu/tests/test_dist_load'

rm -f "$GLMSERVE_NCCL_ID_FILE"
echo "=== w4_real_16x done ==="
