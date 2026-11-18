#include "bit_kernels.h"

#include <algorithm>
#include <bit>
#include <limits>

namespace bitsnn_cpu {
namespace {

inline uint64_t tail_mask(int64_t n) {
    const auto remainder = static_cast<unsigned>(n & 63);
    return remainder == 0 ? std::numeric_limits<uint64_t>::max()
                          : (uint64_t{1} << remainder) - 1;
}

inline int32_t saturate_int32(int64_t value) {
    return static_cast<int32_t>(std::clamp(
        value,
        static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
        static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
}

}  // namespace

bool pack_bits(
    const uint8_t* input,
    int64_t rows,
    int64_t n,
    uint64_t* output) {
    const int64_t words = (n + 63) / 64;
    for (int64_t row = 0; row < rows; ++row) {
        const uint8_t* source = input + row * n;
        uint64_t* destination = output + row * words;
        for (int64_t word = 0; word < words; ++word) {
            const int64_t offset = word * 64;
            const int64_t count = std::min<int64_t>(64, n - offset);
            uint64_t value = 0;
            for (int64_t bit = 0; bit < count; ++bit) {
                const uint8_t spike = source[offset + bit];
                if (spike > 1) {
                    return false;
                }
                value |= static_cast<uint64_t>(spike) << bit;
            }
            destination[word] = value;
        }
    }
    return true;
}

void unpack_bits(
    const uint64_t* input,
    int64_t rows,
    int64_t n,
    uint8_t* output) {
    const int64_t words = (n + 63) / 64;
    for (int64_t row = 0; row < rows; ++row) {
        const uint64_t* source = input + row * words;
        uint8_t* destination = output + row * n;
        for (int64_t word = 0; word < words; ++word) {
            const int64_t offset = word * 64;
            const int64_t count = std::min<int64_t>(64, n - offset);
            const uint64_t value = source[word];
            for (int64_t bit = 0; bit < count; ++bit) {
                destination[offset + bit] = static_cast<uint8_t>((value >> bit) & 1);
            }
        }
    }
}

int32_t binary_dot_scalar(
    const uint64_t* spikes,
    const uint64_t* weights,
    int64_t n) {
    const int64_t full_words = n / 64;
    int32_t sum = 0;
    for (int64_t word = 0; word < full_words; ++word) {
        sum += std::popcount(spikes[word] & weights[word]);
    }
    if ((n & 63) != 0) {
        sum += std::popcount((spikes[full_words] & weights[full_words]) & tail_mask(n));
    }
    return sum;
}

int32_t binary_dot(
    const uint64_t* spikes,
    const uint64_t* weights,
    int64_t n) {
    if (avx2_available()) {
        return binary_dot_avx2(spikes, weights, n);
    }
    return binary_dot_scalar(spikes, weights, n);
}

int32_t binary_dot_unpacked(
    const uint8_t* spikes,
    const uint8_t* weights,
    int64_t n) {
    int32_t sum = 0;
    for (int64_t index = 0; index < n; ++index) {
        sum += static_cast<int32_t>(spikes[index] & weights[index]);
    }
    return sum;
}

void lif_step(
    const int32_t* input,
    const int32_t* membrane,
    int64_t rows,
    int64_t n,
    int32_t decay_q,
    int32_t threshold,
    int32_t v_reset,
    uint64_t* spikes,
    int32_t* next_membrane) {
    const int64_t words = (n + 63) / 64;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t word = 0; word < words; ++word) {
            spikes[row * words + word] = 0;
        }
        for (int64_t index = 0; index < n; ++index) {
            const int64_t offset = row * n + index;
            const int64_t retained =
                (static_cast<int64_t>(membrane[offset]) - v_reset) * decay_q / kQ15One;
            const int64_t charged = static_cast<int64_t>(v_reset) + retained + input[offset];
            const int32_t bounded = saturate_int32(charged);
            const bool fired = bounded >= threshold;
            next_membrane[offset] = fired ? v_reset : bounded;
            if (fired) {
                spikes[row * words + index / 64] |= uint64_t{1} << (index & 63);
            }
        }
    }
}

}  // namespace bitsnn_cpu
