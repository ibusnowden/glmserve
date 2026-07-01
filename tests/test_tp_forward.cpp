// glmserve test — tensor-parallel forward correctness on the tiny checkpoint.
//
// Runs the GLM-5.2 reference forward sharded across a TP group (TP=2, PP=1) and
// checks each rank's logits match a single-process full forward of the same
// checkpoint. This is the end-to-end proof that the TP all-reduce primitive is
// wired into real layer execution: every rank attends its head slice + owns its
// FFN slice, o_proj / down_proj are row-parallel, and all_reduce reconstructs
// the residual stream each sub-layer. (TP reorders the float sums vs a single
// process, so the match is to ~1e-3, not bit-exact like the PP split.)
//
// Launch (see scripts/tp_forward_smoke.sbatch):
//   GLMSERVE_TINY_CKPT=<dir>  GLMSERVE_TP_SIZE=2 GLMSERVE_PP_SIZE=1
//   srun --ntasks=2 build/gpu/tests/test_tp_forward
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>

#include "cuda_test_utils.hpp"
#include "config.hpp"
#include "safetensors.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "nccl_comm.hpp"
#endif

int main() {
#ifdef GLMSERVE_CUDA
    const char* test_name = "test_tp_forward";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    const char* ckpt = std::getenv("GLMSERVE_TINY_CKPT");
    if (!ckpt || !*ckpt) {
        std::printf("test_tp_forward: SKIPPED (set GLMSERVE_TINY_CKPT to a tiny checkpoint dir)\n");
        return 0;
    }

    glmserve::DistConfig cfg = glmserve::dist_config_from_env();
    if (cfg.world_size <= 1 || cfg.tp_size != 2 || cfg.pp_size != 1) {
        std::printf("test_tp_forward: SKIPPED (requires world=2 TP=2 PP=1; got world=%d TP=%d PP=%d)\n",
                    cfg.world_size, cfg.tp_size, cfg.pp_size);
        return 0;
    }

    glmserve::GLM52Config gcfg = glmserve::load_config(ckpt);
    glmserve::SafeTensors st;
    st.load(ckpt);

    glmserve::Communicator comm(cfg);
    if (!comm.active()) {
        std::printf("test_tp_forward: FAIL (communicator inactive)\n");
        return 1;
    }

    const std::vector<int> prompt = {3, 1, 4, 1, 5, 9, 2, 6};

    auto make_kv = [&](glmserve::GLM52Model& m) {
        return glmserve::KVCache(m.num_layers(), m.local_kv_heads(),
                                 gcfg.kv_cache_head_dim(), /*block_size=*/16, /*num_blocks=*/64,
                                 gcfg.use_dsa ? gcfg.index_head_dim : 0);
    };

    // --- tensor-parallel forward: this rank owns a head / FFN slice ---
    glmserve::GLM52Model tp_model(gcfg);
    tp_model.set_distributed(&comm);
    tp_model.load(st);
    glmserve::KVCache tp_cache = make_kv(tp_model);
    glmserve::SequenceKV tp_kv = tp_cache.make_sequence(0);
    std::vector<float> tp_logits = tp_model.forward(prompt, 0, tp_kv);

    // --- reference: full single-process forward of the same checkpoint ---
    glmserve::GLM52Model ref_model(gcfg);   // no communicator => single full rank
    ref_model.load(st);
    glmserve::KVCache ref_cache = make_kv(ref_model);
    glmserve::SequenceKV ref_kv = ref_cache.make_sequence(1);
    std::vector<float> ref_logits = ref_model.forward(prompt, 0, ref_kv);

    comm.barrier();

    if (tp_logits.size() != ref_logits.size()) {
        std::printf("test_tp_forward: FAIL rank=%d (logit size %zu != ref %zu)\n",
                    cfg.rank, tp_logits.size(), ref_logits.size());
        return 1;
    }
    float max_diff = 0.0f;
    int tp_argmax = 0, ref_argmax = 0;
    for (size_t i = 0; i < tp_logits.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(tp_logits[i] - ref_logits[i]));
        if (tp_logits[i] > tp_logits[tp_argmax]) tp_argmax = static_cast<int>(i);
        if (ref_logits[i] > ref_logits[ref_argmax]) ref_argmax = static_cast<int>(i);
    }
    const float tol = 1e-3f;   // all_reduce reorders the float sums vs single-process
    if (tp_argmax != ref_argmax || max_diff > tol) {
        std::printf("test_tp_forward: FAIL rank=%d (tp_rank=%d) tp_argmax=%d ref_argmax=%d "
                    "max_diff=%.6g tol=%.1g\n",
                    cfg.rank, cfg.tp_rank(), tp_argmax, ref_argmax, max_diff, tol);
        return 1;
    }
    std::printf("test_tp_forward: PASS rank=%d (tp_rank=%d, %lld local kv-heads) argmax=%d max_diff=%.6g\n",
                cfg.rank, cfg.tp_rank(), (long long)tp_model.local_kv_heads(), tp_argmax, max_diff);
    return 0;
#else
    std::printf("test_tp_forward: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
