// glmserve — CPU reference dequantizers for GGML/GGUF block formats.
//
// These routines are intentionally small and dependency-free. They provide a
// correctness target for wiring mmap-backed GGUF quant weights into execution.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glmserve {

bool gguf_type_can_dequantize(uint32_t type);
uint64_t gguf_type_block_elements(uint32_t type);
uint64_t gguf_type_block_bytes(uint32_t type);

// Dequantize the first n_elements from an encoded GGUF tensor payload. For
// block formats, n_elements may stop inside a block; the final block is decoded
// and cropped.
std::vector<float> gguf_dequantize_prefix(uint32_t type, const uint8_t* data,
                                          uint64_t n_elements);
std::vector<float> gguf_dequantize_row(uint32_t type, const uint8_t* data,
                                       uint64_t row_elements, uint64_t row_index);

// Stable checksum over float bytes, useful for smoke tests without printing
// payload-sized vectors.
uint64_t gguf_f32_checksum(const float* data, size_t n);
double gguf_row_dot(uint32_t type, const uint8_t* data, uint64_t row_elements,
                    uint64_t row_index, const float* x);

// Pointers to the validated host dequant tables (identical bytes to the tables
// used by gguf_dequantize_prefix). Used to seed GPU __constant__ copies so the
// device dequant matches the CPU reference bit-for-bit.
const uint8_t* gguf_iq3xxs_grid_table();   // 256 * uint32_t
const uint8_t* gguf_ksigns_iq2xs_table();  // 128 * uint8_t
const uint8_t* gguf_kmask_iq2xs_table();   // 8 * uint8_t
const int8_t*  gguf_kvalues_iq4nl_table();  // 16 * int8_t

}  // namespace glmserve
