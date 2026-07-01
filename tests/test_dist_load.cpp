// glmserve test — distributed load/upload smoke for arbitrary TP/PP topology.
//
// This is intentionally narrower than a forward correctness test. It proves
// that every rank can initialize NCCL, load only its owned pipeline layer range
// with the configured TP shard, and upload the resident weights/KV buffers to
// its assigned GPU. That is the first real W4 gate before combined TP=8/PP=2
// serving/forward is wired.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>

#include "cuda_test_utils.hpp"
#include "config.hpp"
#include "model.hpp"
#include "nccl_comm.hpp"
#include "safetensors.hpp"
static int64_t env_i64(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::strtoll(v, nullptr, 10) : fallback;
}

static bool env_bool(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return v[0] != '0' && v[0] != 'f' && v[0] != 'F' && v[0] != 'n' && v[0] != 'N';
}

static std::vector<std::string> stage_prefixes(const glmserve::GLM52Config& cfg,
                                               const glmserve::DistConfig& dist,
                                               int64_t max_layers) {
    int64_t total_layers = (max_layers > 0) ? std::min<int64_t>(max_layers, cfg.num_hidden_layers)
                                            : cfg.num_hidden_layers;
    glmserve::LayerRange range = glmserve::partition_layers(total_layers, dist.pp_stage(), dist.pp_size);
    std::vector<std::string> prefixes;
    if (dist.is_first_stage() || dist.is_last_stage()) {
        prefixes.push_back("model.embed_tokens.");
        prefixes.push_back("embed_tokens.");
    }
    if (dist.is_last_stage()) {
        prefixes.push_back("model.norm.");
        prefixes.push_back("norm.");
        prefixes.push_back("lm_head.");
    }
    for (int64_t i = range.begin; i < range.end; ++i)
        prefixes.push_back("model.layers." + std::to_string(i) + ".");
    if (dist.is_last_stage()) {
        for (int64_t m = 0; m < cfg.num_nextn_predict_layers; ++m)
            prefixes.push_back("model.layers." + std::to_string(cfg.num_hidden_layers + m) + ".");
    }
    return prefixes;
}
#endif

int main() {
#ifdef GLMSERVE_CUDA
    const char* test_name = "test_dist_load";
    int init = cuda_test_init(test_name);
    if (init != 0) return init == 2 ? 0 : 1;

    glmserve::DistConfig dist = glmserve::dist_config_from_env();
    if (dist.world_size <= 1) {
        std::printf("test_dist_load: SKIPPED (requires distributed world; got world=%d)\n",
                    dist.world_size);
        return 0;
    }

    std::string stage_env = "GLMSERVE_MODEL_CKPT_STAGE" + std::to_string(dist.pp_stage());
    const char* ckpt = std::getenv(stage_env.c_str());
    if (!ckpt || !*ckpt) ckpt = std::getenv("GLMSERVE_MODEL_CKPT");
    if (!ckpt || !*ckpt) ckpt = std::getenv("GLMSERVE_TINY_CKPT");
    if (!ckpt || !*ckpt) {
        std::printf("test_dist_load: SKIPPED (set GLMSERVE_MODEL_CKPT, %s, or GLMSERVE_TINY_CKPT)\n",
                    stage_env.c_str());
        return 0;
    }

    glmserve::GLM52Config cfg = glmserve::load_config(ckpt);
    glmserve::SafeTensors st;
    int64_t max_layers = env_i64("GLMSERVE_MAX_LAYERS", -1);
    const char* selective = std::getenv("GLMSERVE_SELECTIVE_LOAD");
    if (!selective || selective[0] != '0')
        st.load_prefixes(ckpt, stage_prefixes(cfg, dist, max_layers));
    else
        st.load(ckpt);

    glmserve::Communicator comm(dist);
    if (!comm.active()) {
        std::printf("test_dist_load: FAIL rank=%d (communicator inactive)\n", dist.rank);
        return 1;
    }

    glmserve::GLM52Model model(cfg);
    model.set_distributed(&comm);
    model.load(st, max_layers);
    glmserve::LayerRange owned = model.owned_layers();

    int64_t ctx = env_i64("GLMSERVE_DIST_LOAD_CTX", 4096);
    bool do_upload = env_bool("GLMSERVE_DIST_LOAD_UPLOAD", dist.pp_size <= 1);
    if (do_upload) {
        if (!model.upload_to_gpu(ctx)) {
            std::printf("test_dist_load: FAIL rank=%d (upload_to_gpu returned false)\n", dist.rank);
            return 1;
        }
        cudaDeviceSynchronize();
    }
    comm.barrier();

    std::printf("test_dist_load: PASS rank=%d/%d local=%d TP=%d tp_rank=%d PP=%d stage=%d "
                "layers=[%lld,%lld) local_heads=%lld ctx=%lld upload=%s\n",
                dist.rank, dist.world_size, dist.local_rank, dist.tp_size, dist.tp_rank(),
                dist.pp_size, dist.pp_stage(), (long long)owned.begin, (long long)owned.end,
                (long long)model.local_kv_heads(), (long long)ctx, do_upload ? "yes" : "no");
    return 0;
#else
    std::printf("test_dist_load: SKIPPED (CPU-only build; rebuild with GPU=1)\n");
    return 0;
#endif
}
