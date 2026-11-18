#pragma once

#include <cstdint>

namespace bitsnn_cpu {

constexpr int32_t kQ15One = 32768;

bool pack_bits(
    const uint8_t* input,
    int64_t rows,
    int64_t n,
    uint64_t* output);

void unpack_bits(
    const uint64_t* input,
    int64_t rows,
    int64_t n,
    uint8_t* output);

int32_t binary_dot_scalar(
    const uint64_t* spikes,
    const uint64_t* weights,
    int64_t n);

int32_t binary_dot_avx2(
    const uint64_t* spikes,
    const uint64_t* weights,
    int64_t n);

int32_t binary_dot(
    const uint64_t* spikes,
    const uint64_t* weights,
    int64_t n);

int32_t binary_dot_unpacked(
    const uint8_t* spikes,
    const uint8_t* weights,
    int64_t n);

void lif_step(
    const int32_t* input,
    const int32_t* membrane,
    int64_t rows,
    int64_t n,
    int32_t decay_q,
    int32_t threshold,
    int32_t v_reset,
    uint64_t* spikes,
    int32_t* next_membrane);

bool avx2_available();

}  // namespace bitsnn_cpu
