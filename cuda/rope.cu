// glmserve — partial-rotary RoPE (NeoX rotate-half on the first `rot` dims).
#include "common.cuh"
#include "kernels.cuh"

namespace glmserve {
namespace cuda {

// grid: (n * n_heads); each thread handles one (i in [0,half)) rotation pair.
__global__ void rope_kernel(float* __restrict__ x, const int64_t* __restrict__ pos,
                            int64_t n_heads, int64_t head_dim, int64_t rot, double theta) {
    int64_t row = blockIdx.x / n_heads;
    int64_t head = blockIdx.x % n_heads;
    int64_t half = rot / 2;
    float* v = x + (row * n_heads + head) * head_dim;
    int64_t p = pos[row];

    for (int64_t i = threadIdx.x; i < half; i += blockDim.x) {
        double freq = pow(theta, (2.0 * i) / rot);
        double ang = p / freq;
        float c = (float)cos(ang);
        float sn = (float)sin(ang);
        float x0 = v[i];
        float x1 = v[i + half];
        v[i]        = x0 * c - x1 * sn;
        v[i + half] = x1 * c + x0 * sn;
    }
}

void rope(float* x, const int64_t* pos, int64_t n, int64_t n_heads, int64_t head_dim,
          int64_t rot, double theta, cudaStream_t s) {
    int threads = 64;
    rope_kernel<<<(unsigned)(n * n_heads), threads, 0, s>>>(x, pos, n_heads, head_dim,
                                                            rot, theta);
}

}  // namespace cuda
}  // namespace glmserve
