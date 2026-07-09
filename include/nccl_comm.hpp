// glmserve — distributed runtime interface for the 2-node topology.
//
// Target layout (spec §5/§10): TP=8 inside each node, PP=2 across nodes, so
// node 0 holds layers [0, split) and node 1 holds [split, L). This avoids
// cross-node tensor-parallel all-reduce on every layer. The single-process
// build provides a no-op Communicator so the engine runs unchanged on one GPU
// / CPU; the GLMSERVE_CUDA + NCCL path fills in real collectives & P2P.
#pragma once

#include <cstdint>
#include <vector>

namespace glmserve {

struct DistConfig {
    int rank = 0;          // global rank
    int world_size = 1;    // total GPUs
    int tp_size = 1;       // tensor-parallel group size (within node)
    int pp_size = 1;       // pipeline-parallel stages (across nodes)
    int local_rank = 0;    // GPU index within node

    int pp_stage() const { return tp_size ? rank / tp_size : 0; }
    int tp_rank()  const { return tp_size ? rank % tp_size : 0; }
    bool is_first_stage() const { return pp_stage() == 0; }
    bool is_last_stage()  const { return pp_stage() == pp_size - 1; }
};

// Build a DistConfig from common launcher env vars:
//   GLMSERVE_RANK / RANK / SLURM_PROCID
//   GLMSERVE_WORLD_SIZE / WORLD_SIZE / SLURM_NTASKS
//   GLMSERVE_LOCAL_RANK / LOCAL_RANK / SLURM_LOCALID
//   GLMSERVE_TP_SIZE, GLMSERVE_PP_SIZE
DistConfig dist_config_from_env();

// Inclusive-exclusive layer range owned by this pipeline stage.
struct LayerRange { int64_t begin = 0; int64_t end = 0; };
LayerRange partition_layers(int64_t num_layers, int pp_stage, int pp_size);

class Communicator {
public:
    explicit Communicator(DistConfig cfg);
    ~Communicator();

    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    const DistConfig& config() const { return cfg_; }
    bool active() const { return state_ != nullptr; }

    // Tensor-parallel all-reduce (sum) over the TP group, in place.
    void all_reduce_sum(float* data, int64_t count);

    // Row-wise variant: reduces each of the `rows` contiguous rows as its own
    // NCCL call (grouped, so they pipeline). A ring all-reduce's per-element
    // summation order depends on the total message size, so reducing an
    // [n, row_elems] chunk in one call gives bitwise-different sums than the
    // n [1, row_elems] calls a plain decode step issues. Verify/absorb chunks
    // use this so speculative decode stays bit-identical to plain greedy.
    void all_reduce_sum_rows(float* data, int64_t rows, int64_t row_elems);

    // Pipeline send/recv of a hidden-state buffer to the next/prev stage.
    void pipeline_send_next(const float* data, int64_t count);
    void pipeline_recv_prev(float* data, int64_t count);
    // Pipeline send/recv for integer side-channel metadata such as DSA masks.
    void pipeline_send_next_int(const int* data, int64_t count);
    void pipeline_recv_prev_int(int* data, int64_t count);

    // Broadcast a contiguous int array from root_rank to all ranks in the TP
    // group (used to keep TP ranks in lockstep on the sampled token / prompt
    // during distributed serving). Host pointers are staged through the device.
    void bcast_int(int* data, int64_t count, int root_rank);

    void barrier();

private:
    DistConfig cfg_;
    void* state_ = nullptr;  // NcclState* in CUDA builds, null in CPU/no-op builds
};

}  // namespace glmserve
