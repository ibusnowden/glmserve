// glmserve test — pipeline-parallel forward correctness on the tiny checkpoint.
//
// Runs the GLM-5.2 reference forward split across PP stages (TP=1, PP=2) using
// the NCCL hidden-state handoff, then checks that the last stage's logits match
// a single-process full forward of the same checkpoint. This is the end-to-end
// proof that the PP send/recv primitives are wired into real layer execution:
// stage 0 owns the lower layers (embed + first half), sends its hidden state to
// stage 1, which owns the upper layers (second half + final-norm + lm_head).
//
// Launch (see scripts/pp_forward_smoke.sbatch):
//   GLMSERVE_TINY_CKPT=<dir>  GLMSERVE_TP_SIZE=1 GLMSERVE_PP_SIZE=2
//   srun --ntasks=2 build/gpu/tests/test_pp_forward
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
    const char* test_name = "test_pp_forward";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    const char* ckpt = std::getenv("GLMSERVE_TINY_CKPT");
    if (!ckpt || !*ckpt) {
        std::printf("test_pp_forward: SKIPPED (set GLMSERVE_TINY_CKPT to a tiny checkpoint dir)\n");
        return 0;
    }

    glmserve::DistConfig cfg = glmserve::dist_config_from_env();
    if (cfg.world_size <= 1 || cfg.pp_size != 2 || cfg.tp_size != 1) {
        std::printf("test_pp_forward: SKIPPED (requires world=2 TP=1 PP=2; got world=%d TP=%d PP=%d)\n",
                    cfg.world_size, cfg.tp_size, cfg.pp_size);
        return 0;
    }

    glmserve::GLM52Config gcfg = glmserve::load_config(ckpt);
    glmserve::SafeTensors st;
    st.load(ckpt);

    glmserve::Communicator comm(cfg);
    if (!comm.active()) {
        std::printf("test_pp_forward: FAIL (communicator inactive)\n");
        return 1;
    }

    // A deterministic prompt (same ids the single-process gate uses).
    const std::vector<int> prompt = {3, 1, 4, 1, 5, 9, 2, 6};

    auto make_kv = [&](glmserve::GLM52Model& m) {
        return glmserve::KVCache(m.num_layers(), gcfg.num_attention_heads,
                                 gcfg.kv_cache_head_dim(), /*block_size=*/16, /*num_blocks=*/64);
    };

    // --- pipeline-sharded forward: this rank runs only its stage's layers ---
    glmserve::GLM52Model pp_model(gcfg);
    pp_model.set_distributed(&comm);
    pp_model.load(st);
    glmserve::LayerRange owned = pp_model.owned_layers();
    glmserve::KVCache pp_cache = make_kv(pp_model);
    glmserve::SequenceKV pp_kv = pp_cache.make_sequence(0);
    std::vector<float> pp_logits = pp_model.forward(prompt, 0, pp_kv);

    if (cfg.is_first_stage()) {
        // Stage 0 produced no logits; it embedded + ran its layers + sent the
        // hidden state (the recv on stage 1 below confirms the bytes arrived).
        comm.barrier();
        std::printf("test_pp_forward: PASS rank=%d (stage 0: layers [%lld,%lld), sent hidden)\n",
                    cfg.rank, (long long)owned.begin, (long long)owned.end);
        return 0;
    }

    // --- reference: full single-process forward of the same checkpoint ---
    glmserve::GLM52Model ref_model(gcfg);   // no communicator => single full stage
    ref_model.load(st);
    glmserve::KVCache ref_cache = make_kv(ref_model);
    glmserve::SequenceKV ref_kv = ref_cache.make_sequence(1);
    std::vector<float> ref_logits = ref_model.forward(prompt, 0, ref_kv);

    comm.barrier();

    if (pp_logits.size() != ref_logits.size()) {
        std::printf("test_pp_forward: FAIL rank=%d (logit size %zu != ref %zu)\n",
                    cfg.rank, pp_logits.size(), ref_logits.size());
        return 1;
    }
    float max_diff = 0.0f;
    int pp_argmax = 0, ref_argmax = 0;
    for (size_t i = 0; i < pp_logits.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(pp_logits[i] - ref_logits[i]));
        if (pp_logits[i] > pp_logits[pp_argmax]) pp_argmax = static_cast<int>(i);
        if (ref_logits[i] > ref_logits[ref_argmax]) ref_argmax = static_cast<int>(i);
    }
    const float tol = 1e-4f;
    if (pp_argmax != ref_argmax || max_diff > tol) {
        std::printf("test_pp_forward: FAIL rank=%d (stage %d: layers [%lld,%lld)) "
                    "pp_argmax=%d ref_argmax=%d max_diff=%.6g tol=%.1g\n",
                    cfg.rank, cfg.pp_stage(), (long long)owned.begin, (long long)owned.end,
                    pp_argmax, ref_argmax, max_diff, tol);
        return 1;
    }
    std::printf("test_pp_forward: PASS rank=%d (stage %d: layers [%lld,%lld), recv'd hidden) "
                "argmax=%d max_diff=%.6g\n",
                cfg.rank, cfg.pp_stage(), (long long)owned.begin, (long long)owned.end,
                pp_argmax, max_diff);
    return 0;
#else
    std::printf("test_pp_forward: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
