#!/usr/bin/env bash
# Validate the real GLM-5.2 llama.cpp GGUF weight set used by ../inference.
#
# This proves the split UD-Q3_K_XL checkpoint is present and that the glmserve
# binary can parse its GGUF metadata/tensor tables, load mmap-backed quant tensor
# views into GLM52Model, and dequantize a real block from each observed GGML type.
# It does not run generation yet; forward() still consumes safetensors and
# glmserve W4A16 repacks.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${GLMSERVE_PYTHON:-/project/inniang/.venv/bin/python}"
GGUF_PATH="${GLMSERVE_GGUF_PATH:-/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL}"
BIN="${GLMSERVE_BIN:-./build/glmserve}"

if [ -x "$BIN" ]; then
  "$BIN" inspect --model "$GGUF_PATH" --require-glm52 --check-glmserve-map --touch-gguf-payloads
  "$BIN" load-gguf --model "$GGUF_PATH" --touch-payloads --dequant-smoke --require-dequant-checksums --linear-smoke --require-linear-checksums
else
  printf 'glmserve binary not found at %s; falling back to Python GGUF checker\n' "$BIN" >&2
  "$PY" tools/inspect_gguf.py "$GGUF_PATH" --require-glm52 --require-quant "${GLMSERVE_REQUIRE_QUANT:-IQ3_XXS}"
fi
