#ifndef NNUE_H_INCLUDED
#define NNUE_H_INCLUDED

#include <array>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <string>

#include "weights.h"

//activation is implicitly ReLU, except for L2 -> Out which is linear.

const int INPUT_COUNT = 768;
const int L1_COUNT = 256;
const int L2_COUNT = 32;
const int OUTPUT_COUNT = 1;

const int L1_SCALING = 64;
const int L2_SCALING = 64;
const int OUTPUT_SCALING = 32;
const int SCALING_FACTOR = L1_SCALING * L2_SCALING * OUTPUT_SCALING;

const __m256i _MASK = _mm256_set1_epi64x(-1);
const __m256i _ZERO = _mm256_setzero_si256();

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

//nnue definition.

class NNUE
{
    public:
        std::array<int, INPUT_COUNT> input_layer = {};
        std::array<std::array<int, INPUT_COUNT>, L1_COUNT> input_weights = {};
        std::array<int, L1_COUNT> input_bias = {};

        std::array<int, L1_COUNT> l1_layer = {};
        std::array<std::array<int, L1_COUNT>, L2_COUNT> l1_weights = {};
        std::array<int, L2_COUNT> l1_bias = {};

        std::array<int, L2_COUNT> l2_layer = {};
        std::array<std::array<int, L2_COUNT>, OUTPUT_COUNT> l2_weights = {};
        std::array<int, OUTPUT_COUNT> l2_bias = {};

        std::array<int, OUTPUT_COUNT> output_layer = {};

        NNUE()
        {
            //read weights and biases.

            //Input -> L1
            for (int i=0;i<L1_COUNT;i++)
            {
                for (int j=0;j<INPUT_COUNT;j++)
                {
                    input_weights[i][j] = std::lround(L1_SCALING * weights_256_768[i][j]);
                }
                input_bias[i] = std::lround(L1_SCALING * bias_256[i]);
            }

            //L1 -> L2
            for (int i=0;i<L2_COUNT;i++)
            {
                for (int j=0;j<L1_COUNT;j++)
                {
                    l1_weights[i][j] = std::lround(L2_SCALING * weights_32_256[i][j]);
                }
                l1_bias[i] = std::lround(L2_SCALING * L1_SCALING * bias_32[i]);
            }

            //L2 -> Output
            for (int i=0;i<OUTPUT_COUNT;i++)
            {
                for (int j=0;j<L2_COUNT;j++)
                {
                    l2_weights[i][j] = std::lround(OUTPUT_SCALING * weights_1_32[i][j]);
                }
                l2_bias[i] = std::lround(SCALING_FACTOR * bias_1[i]);
            }
        }

        void refreshInput(const std::string &fen)
        {
            //fully refresh input layer with fen string.
            input_layer.fill(0);

            int square = 56;
            std::string pieceTypes = "KkQqRrBbNnPp";

            for (int i=0;i<(int)fen.length();i++)
            {
                if (fen[i] == ' ') {break;}
                else if (fen[i] == '/') {square -= 16;}
                else if ((int)(fen[i] - '0') < 9) {square += (int)(fen[i] - '0');}
                else {input_layer[64*pieceTypes.find(fen[i]) + square++] = 1;}
            }

            //fully refresh L1 layer with AVX2.
            __m256i x,y,z;

            for (int i=0;i<L1_COUNT;i++)
            {
                z = _mm256_setzero_si256();
                for (int j=0;j<INPUT_COUNT;j+=8)
                {
                    x = _mm256_maskload_epi32(&input_weights[i][j], _MASK);
                    y = _mm256_maskload_epi32(&input_layer[j], _MASK);
                    z = _mm256_add_epi32(z, _mm256_mullo_epi32(x, y));
                }
                l1_layer[i] = hsum_8x32(z) + input_bias[i];
            }
        }

        void zeroInput(int idx)
        {
            //update first hidden layer, assuming input bit set to zero.
            __m256i x, y;
            for (int i=0;i<L1_COUNT;i+=8)
            {
                x = _mm256_set_epi32(
                    input_weights[i+7][idx],
                    input_weights[i+6][idx],
                    input_weights[i+5][idx],
                    input_weights[i+4][idx],
                    input_weights[i+3][idx],
                    input_weights[i+2][idx],
                    input_weights[i+1][idx],
                    input_weights[i+0][idx]
                );
                y = _mm256_sub_epi32(_mm256_maskload_epi32(&l1_layer[i], _MASK), x);
                _mm256_maskstore_epi32(&l1_layer[i], _MASK, y);
            }
        }

        void oneInput(int idx)
        {
            //update first hidden layer, assuming input bit set to one.
            __m256i x, y;
            for (int i=0;i<L1_COUNT;i+=8)
            {
                x = _mm256_set_epi32(
                    input_weights[i+7][idx],
                    input_weights[i+6][idx],
                    input_weights[i+5][idx],
                    input_weights[i+4][idx],
                    input_weights[i+3][idx],
                    input_weights[i+2][idx],
                    input_weights[i+1][idx],
                    input_weights[i+0][idx]
                );
                y = _mm256_add_epi32(_mm256_maskload_epi32(&l1_layer[i], _MASK), x);
                _mm256_maskstore_epi32(&l1_layer[i], _MASK, y);
            }
        }

        int forward()
        {
            //propagate from first hidden layer to output with AVX2.

            __m256i x, y, z;

            //ReLU(L1) -> L2
            for (int i=0;i<L2_COUNT;i++)
            {
                z = _mm256_setzero_si256();
                for (int j=0;j<L1_COUNT;j+=8)
                {
                    x = _mm256_maskload_epi32(&l1_weights[i][j], _MASK);
                    y = _mm256_max_epi32(_ZERO, _mm256_maskload_epi32(&l1_layer[j], _MASK));
                    z = _mm256_add_epi32(z, _mm256_mullo_epi32(x, y));
                }
                l2_layer[i] = hsum_8x32(z) + l1_bias[i];
            }

            //ReLU(L2) -> Output
            z = _mm256_setzero_si256();
            for (int i=0;i<L2_COUNT;i+=8)
            {
                x = _mm256_maskload_epi32(&l2_weights[0][i], _MASK);
                y = _mm256_max_epi32(_ZERO, _mm256_maskload_epi32(&l2_layer[i], _MASK));
                z = _mm256_add_epi32(z, _mm256_mullo_epi32(x, y));
            }
            output_layer[0] = (hsum_8x32(z) + l2_bias[0]) / SCALING_FACTOR;

            return output_layer[0];
        }
};

#endif // NNUE_H_INCLUDED