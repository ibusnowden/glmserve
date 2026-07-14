// glmserve — CPU/GPU integration test: verify both paths produce identical outputs.
//
// This test:
// 1. Loads a tiny checkpoint (same architecture, small weights)
// 2. Runs the full forward on CPU path
// 3. Runs the full forward on GPU path
// 4. Compares logits to verify they match within tolerance
//
// Usage: ./build/tests/integration_test_cpu_gpu --model /tmp/glm52_tiny
//
// This catches any divergence between CPU reference and GPU kernels.

#include "model.hpp"
#include "config.hpp"
#include "safetensors.hpp"
#include "kv_cache.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace glmserve;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s --model DIR\n", argv[0]);
        return 1;
    }

    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model required\n");
        return 1;
    }

    // Load config
    GLM52Config cfg = load_config(model_path);
    cfg.summarize();

    // Load weights
    SafeTensors st;
    st.load(model_path);

    // Create CPU model
    GLM52Model cpu_model(cfg);
    cpu_model.load(st);

    // Create GPU model (if CUDA available)
    bool gpu_available = false;
    GLM52Model* gpu_model = nullptr;
    #ifdef GLMSERVE_CUDA
    gpu_model = new GLM52Model(cfg);
    gpu_model->load(st);
    if (gpu_model->upload_to_gpu(256)) {
        gpu_available = true;
    } else {
        fprintf(stderr, "GPU upload failed, skipping GPU test\n");
        delete gpu_model;
        gpu_model = nullptr;
    }
    #endif

    if (!gpu_available) {
        fprintf(stderr, "GPU not available, only testing CPU path\n");
        // Just run CPU forward to verify it works
        KVCache cpu_cache(1, 64, 256, 16, 32);
        SequenceKV cpu_kv = cpu_cache.make_sequence(0);
        std::vector<int> tokens = {1, 3, 15, 257};  // BOS + some tokens
        std::vector<float> cpu_logits = cpu_model.forward(tokens, 0, cpu_kv);
        printf("CPU forward completed, logits[%d] = %.6f\n", 
               static_cast<int>(cpu_logits.size()), cpu_logits[0]);
        return 0;
    }

    // Prepare test input
    std::vector<int> tokens = {1, 3, 15, 257};  // BOS + some tokens

    // Run CPU forward
    KVCache cpu_cache(1, 64, 256, 16, 32);
    SequenceKV cpu_kv = cpu_cache.make_sequence(0);
    std::vector<float> cpu_logits = cpu_model.forward(tokens, 0, cpu_kv);

    // Run GPU forward
    std::vector<float> gpu_logits = gpu_model->forward_gpu_prefill(tokens);

    // Compare logits
    int vocab_size = cfg.vocab_size;
    double max_diff = 0.0;
    int max_diff_idx = 0;

    for (int i = 0; i < vocab_size; ++i) {
        double diff = std::fabs(cpu_logits[i] - gpu_logits[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    printf("CPU/GPU logits comparison:\n");
    printf("  Vocab size: %d\n", vocab_size);
    printf("  Max diff: %.6e at idx %d\n", max_diff, max_diff_idx);
    printf("  CPU logits[0]: %.6f\n", cpu_logits[0]);
    printf("  GPU logits[0]: %.6f\n", gpu_logits[0]);

    // Tolerance: 1e-4 should be sufficient for fp32 kernels
    if (max_diff > 1e-4) {
        printf("FAIL: CPU/GPU logits differ by %.6e (tolerance 1e-4)\n", max_diff);
        printf("  CPU logits[%d] = %.6f\n", max_diff_idx, cpu_logits[max_diff_idx]);
        printf("  GPU logits[%d] = %.6f\n", max_diff_idx, gpu_logits[max_diff_idx]);
        return 1;
    }

    printf("PASS: CPU/GPU logits match within tolerance\n");
    return 0;
}
