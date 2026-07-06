#include "gguf_quant.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace glmserve {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;

static uint16_t u16le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t u32le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static float fp16le(const uint8_t* p) {
    return f16_to_f32(u16le(p));
}

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static void dequant_f32(const uint8_t* src, float* dst, uint64_t n) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

static void dequant_f16(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) dst[i] = fp16le(src + 2 * i);
}

static void dequant_q8_0_block(const uint8_t* x, float* y) {
    const float d = fp16le(x);
    const int8_t* qs = reinterpret_cast<const int8_t*>(x + 2);
    for (int j = 0; j < QK8_0; ++j) y[j] = d * qs[j];
}

static void dequant_q3_K_block(const uint8_t* x, float* y) {
    const uint8_t* hmask = x;
    const uint8_t* q = x + QK_K / 8;
    const uint8_t* scales_b = q + QK_K / 4;
    const float d_all = fp16le(scales_b + 12);

    uint32_t aux[4] = {};
    std::memcpy(aux, scales_b, 12);
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * (static_cast<int8_t>((q[l] >> shift) & 3) -
                             ((hmask[l] & m) ? 0 : 4));
            }
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * (static_cast<int8_t>((q[l + 16] >> shift) & 3) -
                             ((hmask[l + 16] & m) ? 0 : 4));
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static void dequant_q4_K_block(const uint8_t* x, float* y) {
    const float d = fp16le(x);
    const float dmin = fp16le(x + 2);
    const uint8_t* scales = x + 4;
    const uint8_t* q = scales + 12;
    int is = 0;
    uint8_t sc = 0, m = 0;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static void dequant_q5_K_block(const uint8_t* x, float* y) {
    const float d = fp16le(x);
    const float dmin = fp16le(x + 2);
    const uint8_t* scales = x + 4;
    const uint8_t* qh = scales + 12;
    const uint8_t* ql = qh + QK_K / 8;
    int is = 0;
    uint8_t sc = 0, m = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

static void dequant_q6_K_block(const uint8_t* x, float* y) {
    const uint8_t* ql = x;
    const uint8_t* qh = ql + QK_K / 2;
    const int8_t* sc = reinterpret_cast<const int8_t*>(qh + QK_K / 4);
    const float d = fp16le(reinterpret_cast<const uint8_t*>(sc + QK_K / 16));
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0x0F) |
                                                  (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0x0F) |
                                                  (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) |
                                                  (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) |
                                                  (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void dequant_iq3_xxs_block(const uint8_t* x, float* y) {
    const float d = fp16le(x);
    const uint8_t* qs = x + 2;
    const uint8_t* scales_and_signs = qs + QK_K / 4;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32 = u32le(scales_and_signs + 4 * ib32);
        const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
            const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                y[j + 4] = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
    }
}

static void dequant_iq4_xs_block(const uint8_t* x, float* y) {
    const float d = fp16le(x);
    const uint16_t scales_h = u16le(x + 2);
    const uint8_t* scales_l = x + 4;
    const uint8_t* qs = scales_l + QK_K / 64;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const uint8_t lo = (ib32 & 1) ? (scales_l[ib32 / 2] >> 4) : (scales_l[ib32 / 2] & 0x0F);
        const uint8_t hi = (scales_h >> (2 * ib32)) & 0x03;
        const int8_t scale = static_cast<int8_t>(lo | (hi << 4)) - 32;
        const float dl = d * static_cast<float>(scale);
        for (int j = 0; j < 16; ++j) y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0x0F];
        for (int j = 0; j < 16; ++j) y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        y += 32;
        qs += 16;
    }
}

static void dequant_one_block(uint32_t type, const uint8_t* src, float* dst) {
    switch (type) {
        case 8:  dequant_q8_0_block(src, dst); break;
        case 11: dequant_q3_K_block(src, dst); break;
        case 12: dequant_q4_K_block(src, dst); break;
        case 13: dequant_q5_K_block(src, dst); break;
        case 14: dequant_q6_K_block(src, dst); break;
        case 18: dequant_iq3_xxs_block(src, dst); break;
        case 23: dequant_iq4_xs_block(src, dst); break;
        default:
            GLM_CHECK(false, "unsupported GGUF dequant type %u", type);
    }
}

}  // namespace

bool gguf_type_can_dequantize(uint32_t type) {
    switch (type) {
        case 0: case 1: case 8: case 11: case 12: case 13: case 14: case 18: case 23:
            return true;
        default:
            return false;
    }
}

uint64_t gguf_type_block_elements(uint32_t type) {
    switch (type) {
        case 0: case 1: return 1;
        case 8: return QK8_0;
        case 11: case 12: case 13: case 14: case 18: case 23: return QK_K;
        default:
            GLM_CHECK(false, "unsupported GGUF dequant type %u", type);
    }
}

uint64_t gguf_type_block_bytes(uint32_t type) {
    switch (type) {
        case 0: return 4;
        case 1: return 2;
        case 8: return 34;
        case 11: return 110;
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        case 18: return 98;
        case 23: return 136;
        default:
            GLM_CHECK(false, "unsupported GGUF dequant type %u", type);
    }
}

std::vector<float> gguf_dequantize_prefix(uint32_t type, const uint8_t* data,
                                          uint64_t n_elements) {
    GLM_CHECK(data != nullptr, "cannot dequantize null GGUF data pointer");
    GLM_CHECK(gguf_type_can_dequantize(type), "unsupported GGUF dequant type %u", type);
    std::vector<float> out(static_cast<size_t>(n_elements));
    if (n_elements == 0) return out;
    if (type == 0) {
        dequant_f32(data, out.data(), n_elements);
        return out;
    }
    if (type == 1) {
        dequant_f16(data, out.data(), n_elements);
        return out;
    }

    const uint64_t block_elems = gguf_type_block_elements(type);
    const uint64_t block_bytes = gguf_type_block_bytes(type);
    std::vector<float> tmp(static_cast<size_t>(block_elems));
    uint64_t done = 0;
    while (done < n_elements) {
        const uint64_t block = done / block_elems;
        dequant_one_block(type, data + block * block_bytes, tmp.data());
        const uint64_t take = std::min<uint64_t>(block_elems, n_elements - done);
        std::memcpy(out.data() + done, tmp.data(), static_cast<size_t>(take) * sizeof(float));
        done += take;
    }
    return out;
}

std::vector<float> gguf_dequantize_row(uint32_t type, const uint8_t* data,
                                       uint64_t row_elements, uint64_t row_index) {
    GLM_CHECK(data != nullptr, "cannot dequantize null GGUF row pointer");
    GLM_CHECK(row_elements > 0, "cannot dequantize empty GGUF row");
    const uint64_t block_elems = gguf_type_block_elements(type);
    const uint64_t block_bytes = gguf_type_block_bytes(type);
    GLM_CHECK(row_elements % block_elems == 0,
              "GGUF row elements %llu not divisible by block size %llu for type %u",
              (unsigned long long)row_elements, (unsigned long long)block_elems, type);
    const uint64_t row_bytes = (row_elements / block_elems) * block_bytes;
    return gguf_dequantize_prefix(type, data + row_index * row_bytes, row_elements);
}

uint64_t gguf_f32_checksum(const float* data, size_t n) {
    uint64_t checksum = 1469598103934665603ull;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n * sizeof(float); ++i) {
        checksum ^= static_cast<uint64_t>(p[i]);
        checksum *= 1099511628211ull;
    }
    return checksum;
}

double gguf_row_dot(uint32_t type, const uint8_t* data, uint64_t row_elements,
                    uint64_t row_index, const float* x) {
    std::vector<float> row = gguf_dequantize_row(type, data, row_elements, row_index);
    double acc = 0.0;
    for (uint64_t i = 0; i < row_elements; ++i) {
        acc += static_cast<double>(row[static_cast<size_t>(i)]) * x[i];
    }
    return acc;
}

const uint8_t* gguf_iq3xxs_grid_table() {
    return reinterpret_cast<const uint8_t*>(iq3xxs_grid);
}
const uint8_t* gguf_ksigns_iq2xs_table() { return ksigns_iq2xs; }
const uint8_t* gguf_kmask_iq2xs_table() { return kmask_iq2xs; }
const int8_t* gguf_kvalues_iq4nl_table() { return kvalues_iq4nl; }

}  // namespace glmserve
