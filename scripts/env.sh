#!/usr/bin/env bash
# Shared toolchain environment for glmserve — the GLM-5.2 C++/CUDA inference engine.
# Source this before building or running:  source scripts/env.sh
#
# Reuses the self-consistent CUDA 12.8 toolkit (.cudaenv) that the sibling mmllm
# project ships, because the system PATH has no nvcc and the pip cmake/nvcc
# wheels are broken on this cluster. Compute nodes have no internet, so we rely
# entirely on these node-local libraries.

export GLMSERVE_ROOT="${GLMSERVE_ROOT:-/project/inniang/glmserve}"

# --- CUDA toolkit (nvcc, cuBLAS/cuBLASLt, cudart, NCCL 2.30) ------------------
export CUDAENV="${CUDAENV:-/project/inniang/entropy/.cudaenv}"
export NVCC="${CUDAENV}/bin/nvcc"

# Host C++ compiler. Compute nodes have no /usr/bin/g++, so use the conda gcc
# that ships in .cudaenv (also what nvcc uses as -ccbin). Works on the login
# node too. Override CXX before sourcing to use a system compiler instead.
if [ -z "${CXX:-}" ] || ! command -v "${CXX}" >/dev/null 2>&1; then
  export CXX="${CUDAENV}/bin/x86_64-conda-linux-gnu-g++"
fi

export CUDA_INC="${CUDAENV}/targets/x86_64-linux/include"
export CUDA_INC2="${CUDAENV}/include"
export CUDA_LIB="${CUDAENV}/targets/x86_64-linux/lib"
export CUDA_LIB2="${CUDAENV}/lib"

# --- Runtime library path -----------------------------------------------------
# Our toolkit libs first, then the system driver (libcuda.so).
export LD_LIBRARY_PATH="${CUDA_LIB}:${CUDA_LIB2}:/usr/lib64:${LD_LIBRARY_PATH:-}"

# --- Default GPU architecture -------------------------------------------------
# RTX 6000 Ada Generation == sm_89.  Override with GPU_ARCH=90 for H100, etc.
export GPU_ARCH="${GPU_ARCH:-89}"

# Convenience: expose nvcc on PATH for this shell.
export PATH="${CUDAENV}/bin:${PATH}"

if [ -x "${NVCC}" ]; then
  echo "[glmserve] toolchain ready: $(${NVCC} --version | tail -1)  (sm_${GPU_ARCH})"
else
  echo "[glmserve] WARNING: nvcc not found at ${NVCC} — CPU-only build available." >&2
fi
