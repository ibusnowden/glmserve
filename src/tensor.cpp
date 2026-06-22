#include "tensor.hpp"

namespace glmserve {

void dequant_to_f32(DType dtype, const void* src, float* out, int64_t count) {
    switch (dtype) {
        case DType::kF32: {
            std::memcpy(out, src, static_cast<size_t>(count) * sizeof(float));
            break;
        }
        case DType::kBF16: {
            const uint16_t* p = static_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < count; ++i) out[i] = bf16_to_f32(p[i]);
            break;
        }
        case DType::kF16: {
            const uint16_t* p = static_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < count; ++i) out[i] = f16_to_f32(p[i]);
            break;
        }
        case DType::kF8E4M3: {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            for (int64_t i = 0; i < count; ++i) out[i] = f8e4m3_to_f32(p[i]);
            break;
        }
        case DType::kF8E5M2: {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            for (int64_t i = 0; i < count; ++i) out[i] = f8e5m2_to_f32(p[i]);
            break;
        }
        case DType::kI32: {
            const int32_t* p = static_cast<const int32_t*>(src);
            for (int64_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DType::kI64: {
            const int64_t* p = static_cast<const int64_t*>(src);
            for (int64_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        default:
            GLM_CHECK(false, "dequant_to_f32: dtype %s needs companion scales "
                             "(use dequant_int4/int8)", dtype_name(dtype));
    }
}

std::vector<float> Tensor::dequant_to_f32() const {
    std::vector<float> out(static_cast<size_t>(numel()));
    glmserve::dequant_to_f32(dtype_, data_, out.data(), numel());
    return out;
}

void dequant_int4_to_f32(const uint8_t* packed, const float* scales,
                         int64_t count, int64_t group_size, float* out) {
    // Two signed 4-bit values per byte: low nibble first. Symmetric, zero-point 8.
    for (int64_t i = 0; i < count; ++i) {
        uint8_t byte = packed[i >> 1];
        int q = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        int signed_q = q - 8;  // [-8, 7]
        float scale = scales[i / group_size];
        out[i] = static_cast<float>(signed_q) * scale;
    }
}

void dequant_int8_to_f32(const int8_t* q, const float* scales,
                         int64_t count, int64_t group_size, float* out) {
    for (int64_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(q[i]) * scales[i / group_size];
    }
}

}  // namespace glmserve
