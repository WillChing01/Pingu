#ifndef NNUE_H_INCLUDED
#define NNUE_H_INCLUDED

#include <array>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <string>

#include "constants.h"
#include "bitboard.h"
#include "weights.h"

//activation is implicitly ReLU, except for L2 -> Out which is linear.

const __m256i _ZERO = _mm256_setzero_si256();
const __m256i _ONE = _mm256_set1_epi16(1);

//horizontal sum of vector
//https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av

inline int hsum_epi32_avx(__m128i x)
{
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

inline int hsum_8x32(__m256i v)
{
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(v),
        _mm256_extracti128_si256(v, 1)
    );
    return hsum_epi32_avx(sum128);
}

inline __m256i cvtepi16_epi8(__m256i low, __m256i high)
{
    __m256i res = _mm256_packs_epi16(low, high);
    res = _mm256_permute4x64_epi64(res, 0b11011000);
    return res;
}

inline __m256i madd_epi8(__m256i a, __m256i b)
{
    __m256i res = _mm256_maddubs_epi16(a, b);
    res = _mm256_madd_epi16(res, _ONE);
    return res;
}

class Accumulator
{
    public:
        __m256i l1[4] = {};
        __m256i cl1[2] = {};

        Accumulator() {}

        void setOne(int index)
        {
            for (size_t i=0;i<4;++i)
            {
                __m256i x = _mm256_loadu_si256((__m256i *)&w_0[index][i * 16]);
                l1[i] = _mm256_add_epi16(l1[i], x);
            }
        }

        void setZero(int index)
        {
            for (size_t i=0;i<4;++i)
            {
                __m256i x = _mm256_loadu_si256((__m256i *)&w_0[index][i * 16]);
                l1[i] = _mm256_sub_epi16(l1[i], x);
            }
        }

        void cReLU()
        {
            cl1[0] = _mm256_max_epi16(_ZERO, cvtepi16_epi8(l1[0], l1[1]));
            cl1[1] = _mm256_max_epi16(_ZERO, cvtepi16_epi8(l1[2], l1[3]));
        }
};

class NNUE
{
    private:
        const U64* pieces;
        const bool* side;

        int kingPos[2] = {};

        Accumulator white;
        Accumulator black;

    public:
        NNUE() {}

        NNUE(const U64* _pieces, const bool* _side): pieces(_pieces), side(_side) {}

        void makeMove(U32 move)
        {
            U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            if (pieceType == 0)
            {
                refreshWhite();
            }
            else if (pieceType == 1)
            {
                refreshBlack();
            }
            else
            {
                //TODO.
            }
        }

        void fullRefresh()
        {
            refreshWhite();
            refreshBlack();
        }

        void refreshWhite()
        {
            kingPos[0] = __builtin_ctzll(pieces[0]);
            std::copy(b_0, b_0 + 64, white.l1);

            white.setOne(704 * kingPos[0] + kingPos[1]);
            for (size_t i=2;i<12;++i)
            {
                U64 x = pieces[i];
                while (x) {setOne(704 * kingPos[0] + 64 * (i - 1) + popLSB(x));}
            }
            white.cReLU();
        }

        void refreshBlack(U32 square)
        {
            kingPos[1] = __builtin_ctzll(pieces[1]);
            std::copy(b_0, b_0 + 64, black.l1);

            black.setOne(704 * kingPos[1] + (kingPos[0] ^ 56));
            for (size_t i=2;i<12;++i)
            {
                U64 x = pieces[i];
                while (x) {setOne(704 * kingPos[1] + 64 * (i - 2 * (i & 1)) + (popLSB(x) ^ 56));}
            }
            black.cReLU();
        }

        int forward()
        {
            __m256i sum = _ZERO;
            for (size_t i=0;i<2;++i)
            {
                __m256i w = _mm256_loadu_si256((__m256i *)&w_1[0][64 * (*side) + 32 * i]);
                sum = _mm256_add_epi32(sum, madd_epi8(white.cl1[i], w));
            }
            for (size_t i=0;i<2;++i)
            {
                __m256i w = _mm256_loadu_si256((__m256i *)&w_1[0][64 * (!(*side)) + 32 * i]);
                sum = _mm256_add_epi32(sum, madd_epi8(black.cl1[i], w));
            }
            return hsum_8x32(sum) + b_1[0];
        }
};



// class NNUE
// {
//     private:
//         const U64* pieces;

//         short input_layer[INPUT_COUNT] = {};
//         short input_weights[INPUT_COUNT][L1_COUNT] = {};
//         short input_bias[L1_COUNT] = {};

//         short l1_layer[L1_COUNT] = {};
//         short l1_crelu[L1_COUNT] = {};
//         short l1_weights[L2_COUNT][L1_COUNT] = {};
//         short l1_bias[L2_COUNT] = {};

//         int l2_layer[L2_COUNT] = {};
//         int l2_weights[OUTPUT_COUNT][L2_COUNT] = {};
//         int l2_bias[OUTPUT_COUNT] = {};

//         int output_layer[OUTPUT_COUNT] = {};

//     public:
//         NNUE() {}

//         NNUE(const U64* _pieces)
//         {
//             pieces = _pieces;
//             readWeights();
//         }

//         void refreshInput()
//         {
//             //refresh the input layer.
//             for (int i=0;i<INPUT_COUNT;i++) {input_layer[i] = 0;}

//             for (int i=0;i<12;i++)
//             {
//                 U64 x = pieces[i];
//                 while (x) {input_layer[64 * i + popLSB(x)] = 1;}
//             }

//             //update l1 layer.
//             for (int i=0;i<L1_COUNT;i++)
//             {
//                 l1_layer[i] = input_bias[i];
//                 for (int j=0;j<INPUT_COUNT;j++)
//                 {
//                     l1_layer[i] += input_weights[j][i] * input_layer[j];
//                 }
//             }
//         }

//         void zeroInput(int idx)
//         {
//             //update first hidden layer, assuming input bit set to zero.
//             __m256i x, y;
//             for (int i=0;i<L1_COUNT;i+=16)
//             {
//                 x = _mm256_loadu_si256((__m256i *)&input_weights[idx][i]);
//                 y = _mm256_loadu_si256((__m256i *)&l1_layer[i]);
//                 y = _mm256_sub_epi16(y, x);
//                 _mm256_storeu_si256((__m256i *)&l1_layer[i], y);
//             }
//         }

//         void oneInput(int idx)
//         {
//             //update first hidden layer, assuming input bit set to one.
//             __m256i x, y;
//             for (int i=0;i<L1_COUNT;i+=16)
//             {
//                 x = _mm256_loadu_si256((__m256i *)&input_weights[idx][i]);
//                 y = _mm256_loadu_si256((__m256i *)&l1_layer[i]);
//                 y = _mm256_add_epi16(y, x);
//                 _mm256_storeu_si256((__m256i *)&l1_layer[i], y);
//             }
//         }

//         int forward()
//         {
//             //propagate from first hidden layer to output with AVX2.

//             __m256i x, y, z;

//             //L1 -> cReLU(L1)
//             for (int i=0;i<L1_COUNT;i+=16)
//             {
//                 x = _mm256_loadu_si256((__m256i *)&l1_layer[i]);
//                 x = _mm256_max_epi16(_ZERO, x);
//                 x = _mm256_min_epi16(_CRELU1, x);
//                 _mm256_storeu_si256((__m256i *)&l1_crelu[i], x);
//             }

//             //cReLU(L1) -> L2
//             for (int i=0;i<L2_COUNT;i++)
//             {
//                 z = _mm256_setzero_si256();
//                 for (int j=0;j<L1_COUNT;j+=16)
//                 {
//                     x = _mm256_loadu_si256((__m256i *)&l1_crelu[j]);
//                     y = _mm256_loadu_si256((__m256i *)&l1_weights[i][j]);
//                     z = _mm256_add_epi32(z, _mm256_madd_epi16(x, y));
//                 }
//                 l2_layer[i] = hsum_8x32(z) + l1_bias[i];
//             }

//             //L2 -> cReLU(L2)
//             for (int i=0;i<L2_COUNT;i+=8)
//             {
//                 x = _mm256_loadu_si256((__m256i *)&l2_layer[i]);
//                 x = _mm256_max_epi32(_ZERO, x);
//                 x = _mm256_min_epi32(_CRELU2, x);
//                 x = _mm256_srai_epi32(x, CRELU2_RSHIFT);
//                 _mm256_storeu_si256((__m256i *)&l2_layer[i], x);
//             }

//             //cReLU(L2) -> Output
//             z = _mm256_setzero_si256();
//             for (int i=0;i<L2_COUNT;i+=8)
//             {
//                 x = _mm256_loadu_si256((__m256i *)&l2_layer[i]);
//                 y = _mm256_loadu_si256((__m256i *)&l2_weights[0][i]);
//                 z = _mm256_add_epi32(z, _mm256_mullo_epi32(x, y));
//             }
//             output_layer[0] = (hsum_8x32(z) + l2_bias[0]) / OUTPUT_FACTOR;

//             return output_layer[0];
//         }
// };

#endif // NNUE_H_INCLUDED
