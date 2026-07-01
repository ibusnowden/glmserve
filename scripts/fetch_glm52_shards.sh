#!/usr/bin/env bash
# Fetch or probe the GLM-5.2 safetensors shards named by the local HF index.
#
# Run this on a login or data-transfer node with internet access. The compute
# nodes used for CUDA jobs may not have outbound network access.
#
# Common usage:
#   GLMSERVE_REAL_MODEL=/project/inniang/hf-cache/glm52-real \
#   bash scripts/fetch_glm52_shards.sh --dry-run
#
#   GLMSERVE_REAL_MODEL=/project/inniang/hf-cache/glm52-real \
#   bash scripts/fetch_glm52_shards.sh --probe 1
#
#   HF_TOKEN=... GLMSERVE_REAL_MODEL=/project/inniang/hf-cache/glm52-real \
#   bash scripts/fetch_glm52_shards.sh --require-free-gib 1800 --verify-present --jobs 4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${GLMSERVE_PYTHON:-python3}"
REAL_MODEL="${GLMSERVE_REAL_MODEL:-/project/inniang/hf-cache/glm52-real}"
REPO="${GLMSERVE_FETCH_REPO:-zai-org/GLM-5.2}"
REVISION="${GLMSERVE_FETCH_REVISION:-main}"

exec "$PY" tools/fetch_safetensors_shards.py \
  --model "$REAL_MODEL" \
  --repo "$REPO" \
  --revision "$REVISION" \
  "$@"
