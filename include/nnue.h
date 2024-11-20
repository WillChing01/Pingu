#ifndef NNUE_H_INCLUDED
#define NNUE_H_INCLUDED

#include <array>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <string>

#include "constants.h"
#include "bitboard.h"
#include "simd.h"
#include "weights.h"

class alignas(32) Accumulator
{
public:
    alignas(32) short l1[64] = {};
    alignas(32) char cl1[64] = {};
    const U64 *pieces;
    bool side;
    int kingPos = 0;

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

    void _move(U32 move, void (Accumulator::*_zero)(U32 x, U32 y), void (Accumulator::*_one)(U32 x, U32 y))
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

        (this->*_zero)(pieceType, startSquare);
        (this->*_one)(finishPieceType, finishSquare);
        if (capturedPieceType != 15)
        {
            (this->*_zero)(capturedPieceType, finishSquare + enPassant * (-8 + 16 * (pieceType & 1)));
        }

        // check for castles.
        if (pieceType == (!side))
        {
            if (finishSquare - startSquare == 2)
            {
                (this->*_zero)(_nRooks + !side, KING_ROOK_SQUARE[!side]);
                (this->*_one)(_nRooks + !side, KING_ROOK_SQUARE[!side] - 2);
            }
            else if (startSquare - finishSquare == 2)
            {
                (this->*_zero)(_nRooks + !side, QUEEN_ROOK_SQUARE[!side]);
                (this->*_one)(_nRooks + !side, QUEEN_ROOK_SQUARE[!side] + 3);
            }
        }
    }

    void refresh()
    {
        kingPos = __builtin_ctzll(pieces[side]);
        std::copy(b_0.begin(), b_0.end(), l1);

        setOne(!side, __builtin_ctzll(pieces[!side]));
        for (size_t i = 2; i < 12; ++i)
        {
            U64 x = pieces[i];
            while (x)
            {
                setOne(i, popLSB(x));
            }
        }
        cReLU();
    }

    void setOne(U32 pieceType, U32 square)
    {
        U32 ind = index(pieceType, square);
        __m256i w;
        __m256i l;
        for (size_t i = 0; i < 64; i += 16)
        {
            w = _mm256_loadu_si256((__m256i *)&w_0[ind][i]);
            l = _mm256_loadu_si256((__m256i *)&l1[i]);
            _mm256_storeu_si256((__m256i *)&l1[i], _mm256_add_epi16(l, w));
        }
        cReLU();
    }

    void setZero(U32 pieceType, U32 square)
    {
        U32 ind = index(pieceType, square);
        __m256i w;
        __m256i l;
        for (size_t i = 0; i < 64; i += 16)
        {
            w = _mm256_loadu_si256((__m256i *)&w_0[ind][i]);
            l = _mm256_loadu_si256((__m256i *)&l1[i]);
            _mm256_storeu_si256((__m256i *)&l1[i], _mm256_sub_epi16(l, w));
        }
        cReLU();
    }

    void cReLU()
    {
        __m256i x, y;
        for (size_t i = 0; i < 64; i += 32)
        {
            x = _mm256_max_epi16(_ZERO, _mm256_loadu_si256((__m256i *)&l1[i]));
            y = _mm256_max_epi16(_ZERO, _mm256_loadu_si256((__m256i *)&l1[i + 16]));
            _mm256_storeu_si256((__m256i *)&cl1[i], cvtepi16_epi8(x, y));
        }
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
        __m256i sum = _ZERO;
        __m256i l, w;

        for (size_t i = 0; i < 64; i += 32)
        {
            l = _mm256_loadu_si256((__m256i *)&white.cl1[i]);
            w = _mm256_loadu_si256((__m256i *)&w_1[0][64 * (*side) + i]);
            sum = _mm256_add_epi32(sum, madd_epi8(l, w));
        }
        for (size_t i = 0; i < 64; i += 32)
        {
            l = _mm256_loadu_si256((__m256i *)&black.cl1[i]);
            w = _mm256_loadu_si256((__m256i *)&w_1[0][64 * (!(*side)) + i]);
            sum = _mm256_add_epi32(sum, madd_epi8(l, w));
        }
        return hsum_8x32(sum) + b_1[0];
    }
};

#endif // NNUE_H_INCLUDED
