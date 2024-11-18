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

// activation is implicitly ReLU, except for L2 -> Out which is linear.

const __m256i _ZERO = _mm256_setzero_si256();
const __m256i _ONE = _mm256_set1_epi16(1);

// horizontal sum of vector
// https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av

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
        _mm256_extracti128_si256(v, 1));
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
    const U64 *pieces;
    bool side;
    int kingPos;
    __m256i l1[4] = {};
    __m256i cl1[2] = {};

    Accumulator() {}

    Accumulator(const U64 *_pieces, bool _side) : pieces(_pieces), side(_side) {}

    virtual U32 index(U32 pieceType, U32 square) = 0;

    void makeMove(U32 move)
    {
        _move(move, &Accumulator::setZero, &Accumulator::setOne);
    }

    void unmakeMove(U32 move)
    {
        _move(move, &Accumulator::setOne, &Accumulator::setZero);
    }

    void _move(U32 move, void (Accumulator::*_zero)(U32 index), void (Accumulator::*_one)(U32 index))
    {
        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        if (pieceType == side)
        {
            refresh();
            return;
        }

        U32 startSquare = (move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
        U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
        U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
        bool enPassant = move & MOVEINFO_ENPASSANT_MASK;

        (this->*_zero)(index(pieceType, startSquare));
        (this->*_one)(index(finishPieceType, finishSquare));
        if (capturedPieceType != 15)
        {
            (this->*_zero)(index(capturedPieceType, finishSquare + enPassant * (-8 + 16 * (pieceType & 1))));
        }

        // check for castles.
        if (pieceType == (!side))
        {
            if (finishSquare - startSquare == 2)
            {
                (this->*_zero)(index(_nRooks + !side, KING_ROOK_SQUARE[!side]));
                (this->*_one)(index(_nRooks + !side, KING_ROOK_SQUARE[!side] - 2));
            }
            else if (startSquare - finishSquare == 2)
            {
                (this->*_zero)(index(_nRooks + !side, QUEEN_ROOK_SQUARE[!side]));
                (this->*_one)(index(_nRooks + !side, QUEEN_ROOK_SQUARE[!side] + 3));
            }
        }
    }

    void refresh()
    {
        kingPos = __builtin_ctzll(pieces[side]);
        std::copy(b_0.begin(), b_0.end(), l1);

        setOne(index(!side, __builtin_ctzll(pieces[!side])));
        for (size_t i = 2; i < 12; ++i)
        {
            U64 x = pieces[i];
            while (x)
            {
                setOne(index(i, popLSB(x)));
            }
        }
        cReLU();
    }

    void setOne(U32 index)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            l1[i] = _mm256_add_epi16(l1[i], w_0[index][i]);
        }
        cReLU();
    }

    void setZero(U32 index)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            l1[i] = _mm256_sub_epi16(l1[i], w_0[index][i]);
        }
        cReLU();
    }

    void cReLU()
    {
        cl1[0] = _mm256_max_epi16(_ZERO, cvtepi16_epi8(l1[0], l1[1]));
        cl1[1] = _mm256_max_epi16(_ZERO, cvtepi16_epi8(l1[2], l1[3]));
    }
};

class White : public Accumulator
{
public:
    White() : Accumulator() {}

    White(const U64 *_pieces) : Accumulator(_pieces, 0) {}

    U32 index(U32 pieceType, U32 square)
    {
        return 704 * kingPos + 64 * (pieceType - 1) + square;
    }
};

class Black : public Accumulator
{
public:
    Black() : Accumulator() {}

    Black(const U64 *_pieces) : Accumulator(_pieces, 1) {}

    U32 index(U32 pieceType, U32 square)
    {
        return 704 * kingPos + 64 * (pieceType - 2 * (pieceType & 1)) + (square ^ 56);
    }
};

class NNUE
{
private:
    White white;
    Black black;
    const bool *side;

public:
    NNUE() {}

    NNUE(const U64 *_pieces, const bool *_side) : side(_side)
    {
        white = White(_pieces);
        black = Black(_pieces);
    }

    void makeMove(U32 move)
    {
        white.makeMove(move);
        black.makeMove(move);
    }

    void unmakeMove(U32 move)
    {
        white.unmakeMove(move);
        black.unmakeMove(move);
    }

    void fullRefresh()
    {
        white.refresh();
        black.refresh();
    }

    int forward()
    {
        __m256i sum = madd_epi8(white.cl1[0], w_1[0][2 * (*side)]);
        sum = _mm256_add_epi32(sum, madd_epi8(white.cl1[1], w_1[0][2 * (*side) + 1]));
        sum = _mm256_add_epi32(sum, madd_epi8(black.cl1[0], w_1[0][2 * (!(*side))]));
        sum = _mm256_add_epi32(sum, madd_epi8(black.cl1[1], w_1[0][2 * (!(*side)) + 1]));
        return hsum_8x32(sum) + b_1[0];
    }
};

#endif // NNUE_H_INCLUDED
