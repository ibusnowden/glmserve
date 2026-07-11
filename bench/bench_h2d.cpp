// Host-to-device upload diagnostic: separate CUDA pinning overhead from steady
// copy bandwidth. Usage: bench_h2d [MiB] [repetitions].
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

static void time_copies(const char* label, void* dst, const void* src,
                        size_t bytes, int reps) {
    check(cudaDeviceSynchronize(), "pre-copy sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), label);
    check(cudaDeviceSynchronize(), "post-copy sync");
    const double sec = seconds_since(t0);
    std::printf("%-12s %.3f s/copy  %.2f GiB/s\n", label, sec / reps,
                (bytes * reps / 1073741824.0) / sec);
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::max(1, std::atoi(argv[1])) : 512;
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3;
    const size_t bytes = mib << 20;
    std::printf("H2D diagnostic: %zu MiB x %d\n", mib, reps);

    void* device = nullptr;
    check(cudaMalloc(&device, bytes), "cudaMalloc device");

    const auto p0 = std::chrono::steady_clock::now();
    std::vector<unsigned char> pageable(bytes, 0x5a);
    std::printf("pageable alloc+touch %.3f s\n", seconds_since(p0));
    time_copies("pageable", device, pageable.data(), bytes, reps);

    void* pinned = nullptr;
    const auto h0 = std::chrono::steady_clock::now();
    check(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    std::fill_n(static_cast<unsigned char*>(pinned), bytes, 0xa5);
    std::printf("pinned alloc+touch   %.3f s\n", seconds_since(h0));
    time_copies("pinned", device, pinned, bytes, reps);

    cudaFreeHost(pinned);
    cudaFree(device);
    return 0;
}
