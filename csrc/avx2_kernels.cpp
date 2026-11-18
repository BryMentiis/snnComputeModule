#include "bit_kernels.h"

#include <bit>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define BITSNN_X86 1
#include <immintrin.h>
#else
#define BITSNN_X86 0
#endif

#if BITSNN_X86 && defined(_MSC_VER)
#include <intrin.h>
#endif

#if BITSNN_X86 && (defined(__GNUC__) || defined(__clang__))
#define BITSNN_AVX2_TARGET __attribute__((target("avx2")))
#else
#define BITSNN_AVX2_TARGET
#endif

namespace bitsnn_cpu {
namespace {

#if BITSNN_X86 && defined(_MSC_VER)
bool detect_avx2() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuid(info, 1);
    const bool os_avx = (info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0;
    if (!os_avx || (_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}
#elif BITSNN_X86 && (defined(__GNUC__) || defined(__clang__))
bool detect_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#else
bool detect_avx2() {
    return false;
}
#endif

inline uint64_t tail_mask(int64_t n) {
    const auto remainder = static_cast<unsigned>(n & 63);
    return remainder == 0 ? std::numeric_limits<uint64_t>::max()
                          : (uint64_t{1} << remainder) - 1;
}

}  // namespace

bool avx2_available() {
    static const bool available = detect_avx2();
    return available;
}

BITSNN_AVX2_TARGET int32_t binary_dot_avx2(
    const uint64_t* spikes,
    const uint64_t* weights,
    int64_t n) {
#if BITSNN_X86
    const int64_t full_words = n / 64;
    const __m256i low_nibble = _mm256_set1_epi8(0x0f);
    const __m256i popcount_lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i zero = _mm256_setzero_si256();
    __m256i accumulated = _mm256_setzero_si256();

    int64_t word = 0;
    for (; word + 4 <= full_words; word += 4) {
        const __m256i x = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(spikes + word));
        const __m256i w = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + word));
        const __m256i active = _mm256_and_si256(x, w);
        const __m256i low = _mm256_and_si256(active, low_nibble);
        const __m256i high = _mm256_and_si256(_mm256_srli_epi16(active, 4), low_nibble);
        const __m256i counts = _mm256_add_epi8(
            _mm256_shuffle_epi8(popcount_lut, low),
            _mm256_shuffle_epi8(popcount_lut, high));
        accumulated = _mm256_add_epi64(accumulated, _mm256_sad_epu8(counts, zero));
    }

    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), accumulated);
    int64_t sum = static_cast<int64_t>(lanes[0] + lanes[1] + lanes[2] + lanes[3]);
    for (; word < full_words; ++word) {
        sum += std::popcount(spikes[word] & weights[word]);
    }
    if ((n & 63) != 0) {
        sum += std::popcount((spikes[full_words] & weights[full_words]) & tail_mask(n));
    }
    _mm256_zeroupper();
    return static_cast<int32_t>(sum);
#else
    return binary_dot_scalar(spikes, weights, n);
#endif
}

}  // namespace bitsnn_cpu
