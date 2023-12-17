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

const int INPUT_SCALING = 128;
const int WEIGHT_SCALING = 64;
const int OUTPUT_SCALING = 512;
const int CRELU1_LSHIFT = 1; // INPUT_SCALING / WEIGHT_SCALING
const int CRELU2_RSHIFT = 6; // INPUT_SCALING / (INPUT_SCALING * WEIGHT_SCALING)
const int OUTPUT_RSHIFT = 4; // OUTPUT_SCALING / (INPUT_SCALING * WEIGHT_SCALING)

const __m256i _ZERO = _mm256_setzero_si256();
const __m256i _CRELU1 = _mm256_set1_epi32(WEIGHT_SCALING);
const __m256i _CRELU2 = _mm256_set1_epi32(INPUT_SCALING * WEIGHT_SCALING);

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
        int input_layer[INPUT_COUNT] = {};
        int input_weights[L1_COUNT][INPUT_COUNT] = {};
        int input_bias[L1_COUNT] = {};

        int l1_layer[L1_COUNT] = {};
        int l1_crelu[L1_COUNT] = {};
        int l1_weights[L2_COUNT][L1_COUNT] = {};
        int l1_bias[L2_COUNT] = {};

        int l2_layer[L2_COUNT] = {};
        int l2_weights[OUTPUT_COUNT][L2_COUNT] = {};
        int l2_bias[OUTPUT_COUNT] = {};

        int output_layer[OUTPUT_COUNT] = {};

        NNUE()
        {
            //read weights and biases.

            //Input -> L1
            for (int i=0;i<L1_COUNT;i++)
            {
                for (int j=0;j<INPUT_COUNT;j++)
                {
                    input_weights[i][j] = std::lround(WEIGHT_SCALING * weights_256_768[i][j]);
                }
                input_bias[i] = std::lround(WEIGHT_SCALING * bias_256[i]);
            }

            //L1 -> L2
            for (int i=0;i<L2_COUNT;i++)
            {
                for (int j=0;j<L1_COUNT;j++)
                {
                    l1_weights[i][j] = std::lround(WEIGHT_SCALING * weights_32_256[i][j]);
                }
                l1_bias[i] = std::lround(INPUT_SCALING * WEIGHT_SCALING * bias_32[i]);
            }

            //L2 -> Output
            for (int i=0;i<OUTPUT_COUNT;i++)
            {
                for (int j=0;j<L2_COUNT;j++)
                {
                    l2_weights[i][j] = std::lround(WEIGHT_SCALING * weights_1_32[i][j]);
                }
                l2_bias[i] = std::lround(INPUT_SCALING * WEIGHT_SCALING * bias_1[i]);
            }
        }

        void refreshInput(const std::string &fen)
        {
            //fully refresh input layer with fen string.
            for (int i=0;i<INPUT_COUNT;i++) {input_layer[i] = 0;}

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
                    x = _mm256_loadu_si256((__m256i *)&input_layer[j]);
                    y = _mm256_loadu_si256((__m256i *)&input_weights[i][j]);
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
                y = _mm256_loadu_si256((__m256i *)&l1_layer[i]);
                y = _mm256_sub_epi32(y, x);
                _mm256_storeu_si256((__m256i *)&l1_layer[i], y);
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
                y = _mm256_loadu_si256((__m256i *)&l1_layer[i]);
                y = _mm256_add_epi32(y, x);
                _mm256_storeu_si256((__m256i *)&l1_layer[i], y);
            }
        }

        int forward()
        {
            //propagate from first hidden layer to output with AVX2.

            __m256i x, y, z;

            //L1 -> cReLU(L1)
            for (int i=0;i<L1_COUNT;i+=8)
            {
                x = _mm256_loadu_si256((__m256i *)&l1_layer[i]);
                x = _mm256_max_epi32(_ZERO, x);
                x = _mm256_min_epi32(_CRELU1, x);
                x = _mm256_slli_epi32(x, CRELU1_LSHIFT);
                _mm256_storeu_si256((__m256i *)&l1_crelu[i], x);
            }

            //cReLU(L1) -> L2
            for (int i=0;i<L2_COUNT;i++)
            {
                z = _mm256_setzero_si256();
                for (int j=0;j<L1_COUNT;j+=8)
                {
                    x = _mm256_loadu_si256((__m256i *)&l1_crelu[j]);
                    y = _mm256_loadu_si256((__m256i *)&l1_weights[i][j]);
                    z = _mm256_add_epi32(z, _mm256_mullo_epi32(x, y));
                }
                l2_layer[i] = hsum_8x32(z) + l1_bias[i];
            }

            //L2 -> cReLU(L2)
            for (int i=0;i<L2_COUNT;i+=8)
            {
                x = _mm256_loadu_si256((__m256i *)&l2_layer[i]);
                x = _mm256_max_epi32(_ZERO, x);
                x = _mm256_min_epi32(_CRELU2, x);
                x = _mm256_srli_epi32(x, CRELU2_RSHIFT);
                _mm256_storeu_si256((__m256i *)&l2_layer[i], x);
            }

            //cReLU(L2) -> Output
            z = _mm256_setzero_si256();
            for (int i=0;i<L2_COUNT;i+=8)
            {
                x = _mm256_loadu_si256((__m256i *)&l2_layer[i]);
                y = _mm256_loadu_si256((__m256i *)&l2_weights[0][i]);
                z = _mm256_add_epi32(z, _mm256_mullo_epi32(x, y));
            }
            output_layer[0] = (hsum_8x32(z) + l2_bias[0]) >> OUTPUT_RSHIFT;

            return output_layer[0];
        }
};

#endif // NNUE_H_INCLUDED
