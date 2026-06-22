// glmserve — common utilities: logging, error handling, dtype enum, half/bf16/fp8
// conversions, timing.  Header-only, no CUDA dependency unless GLMSERVE_CUDA is set.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <stdexcept>

namespace glmserve {

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
enum class LogLevel { kError = 0, kWarn = 1, kInfo = 2, kDebug = 3 };

inline LogLevel& log_level() {
    static LogLevel lvl = [] {
        const char* e = std::getenv("GLMSERVE_LOG");
        if (e) {
            if (!std::strcmp(e, "error")) return LogLevel::kError;
            if (!std::strcmp(e, "warn"))  return LogLevel::kWarn;
            if (!std::strcmp(e, "debug")) return LogLevel::kDebug;
        }
        return LogLevel::kInfo;
    }();
    return lvl;
}

#define GLM_LOG(lvl, tag, ...)                                                  \
    do {                                                                        \
        if (static_cast<int>(lvl) <= static_cast<int>(::glmserve::log_level())) {\
            std::fprintf(stderr, "[glmserve][" tag "] ");                       \
            std::fprintf(stderr, __VA_ARGS__);                                  \
            std::fprintf(stderr, "\n");                                         \
        }                                                                       \
    } while (0)

#define GLM_INFO(...)  GLM_LOG(::glmserve::LogLevel::kInfo,  "info",  __VA_ARGS__)
#define GLM_WARN(...)  GLM_LOG(::glmserve::LogLevel::kWarn,  "warn",  __VA_ARGS__)
#define GLM_ERROR(...) GLM_LOG(::glmserve::LogLevel::kError, "error", __VA_ARGS__)
#define GLM_DEBUG(...) GLM_LOG(::glmserve::LogLevel::kDebug, "debug", __VA_ARGS__)

#define GLM_CHECK(cond, ...)                                                    \
    do {                                                                        \
        if (!(cond)) {                                                          \
            char _buf[1024];                                                    \
            std::snprintf(_buf, sizeof(_buf), __VA_ARGS__);                     \
            throw std::runtime_error(std::string("glmserve check failed: ") +  \
                                     _buf + " (at " + __FILE__ + ":" +          \
                                     std::to_string(__LINE__) + ")");           \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Dtypes.  These mirror the on-disk safetensors dtype strings.
// ---------------------------------------------------------------------------
enum class DType {
    kF32,    // 32-bit float
    kF16,    // IEEE half
    kBF16,   // bfloat16
    kF8E4M3, // FP8 e4m3 (weights / kv-cache)
    kF8E5M2, // FP8 e5m2
    kI32,    // int32
    kI64,    // int64
    kI8,     // int8 (quantized weights, with separate scales)
    kU8,     // uint8 (packed int4: two nibbles per byte)
    kUnknown
};

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::kF32:    return "F32";
        case DType::kF16:    return "F16";
        case DType::kBF16:   return "BF16";
        case DType::kF8E4M3: return "F8_E4M3";
        case DType::kF8E5M2: return "F8_E5M2";
        case DType::kI32:    return "I32";
        case DType::kI64:    return "I64";
        case DType::kI8:     return "I8";
        case DType::kU8:     return "U8";
        default:             return "UNKNOWN";
    }
}

inline DType dtype_from_string(const std::string& s) {
    if (s == "F32")  return DType::kF32;
    if (s == "F16")  return DType::kF16;
    if (s == "BF16") return DType::kBF16;
    if (s == "F8_E4M3") return DType::kF8E4M3;
    if (s == "F8_E5M2") return DType::kF8E5M2;
    if (s == "I32")  return DType::kI32;
    if (s == "I64")  return DType::kI64;
    if (s == "I8")   return DType::kI8;
    if (s == "U8")   return DType::kU8;
    return DType::kUnknown;
}

inline size_t dtype_size_bits(DType d) {
    switch (d) {
        case DType::kF32: case DType::kI32: return 32;
        case DType::kI64: return 64;
        case DType::kF16: case DType::kBF16: return 16;
        case DType::kF8E4M3: case DType::kF8E5M2: case DType::kI8: case DType::kU8: return 8;
        default: return 0;
    }
}

// ---------------------------------------------------------------------------
// Scalar conversions (host side, no intrinsics — portable reference path).
// ---------------------------------------------------------------------------

// bfloat16 (top 16 bits of an fp32) -> float
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // round-to-nearest-even
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// IEEE half -> float
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/- zero
        } else {
            // subnormal: normalize
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x400u));
            mant &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13); // inf / nan
    } else {
        bits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant += 1; // round
        return static_cast<uint16_t>(sign | half_mant);
    } else if (exp >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00u);   // overflow -> inf
    }
    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    if ((mant >> 12) & 1u) half += 1;                    // round-to-nearest
    return half;
}

// FP8 e4m3 (1-4-3, no inf, bias 7) -> float
inline float f8e4m3_to_f32(uint8_t v) {
    uint32_t sign = (v & 0x80u) << 24;
    uint32_t exp  = (v >> 3) & 0x0fu;
    uint32_t mant = v & 0x07u;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x08u));
            mant &= 0x07u;
            bits = sign | (static_cast<uint32_t>(127 - 7 - e) << 23) | (mant << 20);
        }
    } else {
        bits = sign | (static_cast<uint32_t>(exp - 7 + 127) << 23) | (mant << 20);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// FP8 e5m2 (1-5-2, bias 15) -> float
inline float f8e5m2_to_f32(uint8_t v) {
    uint32_t sign = (v & 0x80u) << 24;
    uint32_t exp  = (v >> 2) & 0x1fu;
    uint32_t mant = v & 0x03u;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x04u));
            mant &= 0x03u;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mant << 21);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 21);
    } else {
        bits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 21);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    void reset() { t0 = std::chrono::high_resolution_clock::now(); }
};

}  // namespace glmserve
