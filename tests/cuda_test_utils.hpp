#pragma once

#ifdef GLMSERVE_CUDA
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static bool cuda_required() {
    const char* v = std::getenv("GLMSERVE_REQUIRE_CUDA");
    return v && v[0] && v[0] != '0';
}

// Returns 0 when CUDA is usable, 1 on hard failure, 2 when skipped.
static int cuda_test_init(const char* test_name) {
    int ndev = 0;
    cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess || ndev <= 0) {
        bool required = cuda_required();
        std::printf("%s: %s (no CUDA device: %s)\n", test_name,
                    required ? "FAIL" : "SKIPPED", cudaGetErrorString(e));
        return required ? 1 : 2;
    }
    e = cudaSetDevice(0);
    if (e != cudaSuccess) {
        std::printf("%s: FAIL (cudaSetDevice: %s)\n", test_name, cudaGetErrorString(e));
        return 1;
    }
    cudaGetLastError();
    return 0;
}

#define CUDA_TEST_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::printf("%s: FAIL (%s: %s)\n", test_name, #call,                \
                        cudaGetErrorString(_e));                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#endif
