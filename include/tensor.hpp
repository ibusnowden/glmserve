// glmserve — host tensor abstraction + dequantization helpers.
//
// A Tensor is a typed, shaped view over a block of bytes. It can either own its
// storage (heap) or be a non-owning view into a memory-mapped safetensors file
// (keeping the mapping alive via a shared keepalive handle).
//
// The CPU reference path computes in float32. dequant_to_f32() materializes any
// supported on-disk dtype (BF16/FP16/FP8/F32) into a float32 vector. Quantized
// integer weights (I8/U8-packed-int4) require companion scale tensors and are
// handled by dequant_block() given the scales.
#pragma once

#include "common.hpp"

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace glmserve {

class Tensor {
public:
    Tensor() = default;

    Tensor(DType dtype, std::vector<int64_t> shape, void* data,
           std::shared_ptr<void> keepalive = nullptr)
        : dtype_(dtype), shape_(std::move(shape)), data_(data),
          keepalive_(std::move(keepalive)) {}

    // Allocate an owning F32 tensor.
    static Tensor zeros_f32(std::vector<int64_t> shape) {
        int64_t n = num_elements(shape);
        auto buf = std::shared_ptr<void>(new float[n](), [](void* p) {
            delete[] static_cast<float*>(p);
        });
        return Tensor(DType::kF32, std::move(shape), buf.get(), buf);
    }

    DType dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t dim(int i) const { return shape_[i]; }
    int64_t numel() const { return num_elements(shape_); }
    size_t  nbytes() const {
        return static_cast<size_t>(numel()) * dtype_size_bits(dtype_) / 8;
    }
    void*       data()       { return data_; }
    const void* data() const { return data_; }
    bool valid() const { return data_ != nullptr; }

    template <typename T> T* as() { return static_cast<T*>(data_); }
    template <typename T> const T* as() const { return static_cast<const T*>(data_); }

    static int64_t num_elements(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                               std::multiplies<int64_t>());
    }

    std::string shape_str() const {
        std::string s = "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            s += std::to_string(shape_[i]);
            if (i + 1 < shape_.size()) s += ",";
        }
        return s + "]";
    }

    // Materialize this tensor as float32 (dequantizing scale-free dtypes).
    std::vector<float> dequant_to_f32() const;

private:
    DType dtype_ = DType::kUnknown;
    std::vector<int64_t> shape_;
    void* data_ = nullptr;
    std::shared_ptr<void> keepalive_;  // keeps mmap/heap alive
};

// Dequantize raw bytes of a known dtype into out (length = count).
void dequant_to_f32(DType dtype, const void* src, float* out, int64_t count);

// Dequantize a packed-int4 (U8, two nibbles/byte) or int8 (I8) weight block
// using per-group float scales. group_size is the number of weight columns
// sharing one scale; scales has length ceil(count/group_size). Symmetric.
void dequant_int4_to_f32(const uint8_t* packed, const float* scales,
                         int64_t count, int64_t group_size, float* out);
void dequant_int8_to_f32(const int8_t* q, const float* scales,
                         int64_t count, int64_t group_size, float* out);

}  // namespace glmserve
