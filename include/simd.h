#ifndef SIMD_H_INCLUDED
#define SIMD_H_INCLUDED

#include <immintrin.h>

const __m256i _ZERO = _mm256_setzero_si256();
const __m256i _ONE = _mm256_set1_epi16(1);
const __m256i _HALF = _mm256_set1_epi16(32);

// horizontal sum of vector
// https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av

inline int hsum_epi32_avx(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

inline int hsum_8x32(__m256i v) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    return hsum_epi32_avx(sum128);
}

inline __m256i cvtepi16_epi8(__m256i low, __m256i high) {
    __m256i res = _mm256_packs_epi16(low, high);
    res = _mm256_permute4x64_epi64(res, 0b11011000);
    return res;
}

inline __m256i madd_epi8(__m256i a, __m256i b) {
    __m256i res = _mm256_maddubs_epi16(a, b);
    res = _mm256_madd_epi16(res, _ONE);
    return res;
}

#endif // SIMD_H_INCLUDED
