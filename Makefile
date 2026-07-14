# glmserve — GLM-5.2 C++/CUDA inference engine.
#
# Two build modes:
#   make            CPU-only reference engine (works on a login node, no GPU).
#                   Compiles the full control plane + the CPU correctness path.
#   make GPU=1      Full CUDA build: compiles cuda/*.cu with nvcc (sm_$(GPU_ARCH))
#                   and links cuBLAS/cuBLASLt + NCCL. Requires `source scripts/env.sh`.
#
# Other targets: tests, bench, clean. Output binaries go in build/.

CXX       ?= g++
NVCC      ?= $(CUDAENV)/bin/nvcc
GPU_ARCH  ?= 89

BUILD     := build
MODE      := $(if $(GPU),gpu,cpu)
OBJDIR    := $(BUILD)/$(MODE)
INCLUDES  := -Iinclude -Icuda
CXXFLAGS  := -std=c++17 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -pthread -MMD -MP $(INCLUDES)
LDFLAGS   := -pthread

# ---- core C++ sources (always compiled) -------------------------------------
CPP_SRCS := \
	src/config.cpp \
	src/tensor.cpp \
	src/gguf.cpp \
	src/gguf_quant.cpp \
	src/safetensors.cpp \
	src/tokenizer.cpp \
	src/kv_cache.cpp \
	src/sampler.cpp \
	src/model_glm52.cpp \
	src/model_gguf.cpp \
	src/scheduler.cpp \
	src/http_server.cpp \
	src/server.cpp \
	src/nccl_comm.cpp

# The GPU forward path (device-resident weights + kernel calls) is host C++ that
# calls the launch wrappers in cuda/*.cu. It is entirely under #ifdef GLMSERVE_CUDA,
# so it is only meaningful — and only compiled — in the GPU build. In a CPU build
# the GPU entry points are provided as stubs in model_glm52.cpp instead.
ifeq ($(GPU),1)
  CPP_SRCS += src/model_gpu.cpp
endif

CPP_OBJS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))

# ---- CUDA sources (only with GPU=1) -----------------------------------------
CU_SRCS := $(wildcard cuda/*.cu)
CU_OBJS := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SRCS))

ifeq ($(GPU),1)
  CUDA_INC  ?= $(CUDAENV)/targets/x86_64-linux/include
  CUDA_LIB  ?= $(CUDAENV)/targets/x86_64-linux/lib
  CXXFLAGS += -DGLMSERVE_CUDA -I$(CUDA_INC) -I$(CUDAENV)/include
  INCLUDES  += -I$(CUDA_INC) -I$(CUDAENV)/include
  # -ccbin points nvcc at the conda host compiler (compute nodes have no system g++).
  CCBIN     := $(if $(CXX),-ccbin $(CXX),)
  NVCCFLAGS := -std=c++17 -O3 -DGLMSERVE_CUDA -arch=sm_$(GPU_ARCH) $(CCBIN) \
               --expt-relaxed-constexpr -Xcompiler -fPIC $(INCLUDES)
  LDFLAGS  += -L$(CUDA_LIB) -L$(CUDAENV)/lib \
              -lcudart -lcublas -lcublasLt -lnccl -lstdc++
  ALL_OBJS := $(CPP_OBJS) $(CU_OBJS)
  LINKER    := $(NVCC)
  LINKFLAGS := -arch=sm_$(GPU_ARCH) $(CCBIN) -L$(CUDA_LIB) -L$(CUDAENV)/lib \
               -lcudart -lcublas -lcublasLt -lnccl -Xcompiler -pthread
else
  ALL_OBJS := $(CPP_OBJS)
  LINKER    := $(CXX)
  LINKFLAGS := $(LDFLAGS)
endif

.PHONY: all cpu gpu tests bench run-tests clean dirs

all: $(BUILD)/glmserve
cpu: all
gpu:
	$(MAKE) GPU=1

$(BUILD)/glmserve: dirs $(ALL_OBJS) $(OBJDIR)/src/main.o
	$(LINKER) $(ALL_OBJS) $(OBJDIR)/src/main.o -o $@ $(LINKFLAGS)
	@echo "[glmserve] built $@  (mode: $(if $(GPU),CUDA sm_$(GPU_ARCH),CPU-reference))"

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# auto-generated header dependencies (-MMD)
-include $(CPP_OBJS:.o=.d) $(OBJDIR)/src/main.d

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

dirs:
	@mkdir -p $(OBJDIR)/src $(OBJDIR)/cuda $(OBJDIR)/tests $(OBJDIR)/bench

# ---- tests ------------------------------------------------------------------
TEST_SRCS := $(wildcard tests/*.cpp)
TEST_BINS := $(patsubst tests/%.cpp,$(OBJDIR)/tests/%,$(TEST_SRCS))
# Exclude integration test from regular test suite (requires GPU)
TEST_BINS_EXCLUDE := $(OBJDIR)/tests/integration_test_cpu_gpu
TEST_BINS_RUN := $(filter-out $(TEST_BINS_EXCLUDE), $(TEST_BINS))

tests: dirs $(ALL_OBJS) $(TEST_BINS)
	@echo "[glmserve] built tests: $(TEST_BINS)"

$(OBJDIR)/tests/%: tests/%.cpp $(ALL_OBJS)
	@mkdir -p $(OBJDIR)/_obj
	$(CXX) $(CXXFLAGS) -c $< -o $(OBJDIR)/_obj/test_$*.o
	$(LINKER) $(OBJDIR)/_obj/test_$*.o $(ALL_OBJS) -o $@ $(LINKFLAGS)

# Run every built test/Python gate; nonzero exit if any fails.
run-tests: tests
	@rc=0; for t in $(TEST_BINS_RUN); do echo "--- $$t"; $$t || rc=1; done; \
	echo "--- tests/test_logits_match.py"; \
	python3 tests/test_logits_match.py --bin $(BUILD)/glmserve || rc=1; \
	echo "--- tests/test_mtp_logits.py"; \
	python3 tests/test_mtp_logits.py --bin $(BUILD)/glmserve || rc=1; \
	echo "--- tests/test_mtp_speculative.py"; \
	python3 tests/test_mtp_speculative.py --bin $(BUILD)/glmserve || rc=1; \
	echo "--- tests/test_mtp_generate.py"; \
	python3 tests/test_mtp_generate.py --bin $(BUILD)/glmserve || rc=1; \
	echo "--- tests/test_w4_quantized.py"; \
	python3 tests/test_w4_quantized.py --bin $(BUILD)/glmserve || rc=1; \
	exit $$rc

# Run the CPU/GPU integration test (requires GPU=1 build + CUDA device)
run-integration-test: tests
	@echo "--- $(OBJDIR)/tests/integration_test_cpu_gpu ---"
	$(OBJDIR)/tests/integration_test_cpu_gpu --model $(GLMSERVE_MODEL)

# ---- benches ----------------------------------------------------------------
BENCH_SRCS := $(wildcard bench/*.cpp)
BENCH_BINS := $(patsubst bench/%.cpp,$(OBJDIR)/bench/%,$(BENCH_SRCS))

bench: dirs $(ALL_OBJS) $(BENCH_BINS)
	@echo "[glmserve] built benches: $(BENCH_BINS)"

$(OBJDIR)/bench/%: bench/%.cpp $(ALL_OBJS)
	@mkdir -p $(OBJDIR)/_obj
	$(CXX) $(CXXFLAGS) -c $< -o $(OBJDIR)/_obj/bench_$*.o
	$(LINKER) $(OBJDIR)/_obj/bench_$*.o $(ALL_OBJS) -o $@ $(LINKFLAGS)

clean:
	rm -rf $(BUILD)
