#ifndef NNUE_H_INCLUDED
#define NNUE_H_INCLUDED

#include "constants.h"
#include "bitboard.h"
#include "simd.h"
#include "weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>
#include <string>

template <int (*index)(U32 kingPos, U32 pieceType, U32 square), bool side>
class alignas(32) Accumulator {
  public:
    alignas(32) short l1[32] = {};
    alignas(32) char cl1[32] = {};
    const U64* pieces;
    int kingPos = 0;

    Accumulator() {}

    Accumulator(const U64* _pieces) : pieces(_pieces) {}

    template <void (Accumulator::*_zero)(U32, U32), void (Accumulator::*_one)(U32, U32)>
    void _move(U32 move) {
        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        if (pieceType == side) {
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
        if (capturedPieceType != 15) {
            (this->*_zero)(capturedPieceType, finishSquare + enPassant * (-8 + 16 * (pieceType & 1)));
        }

        // check for castles.
        if (pieceType == (!side)) {
            if (finishSquare - startSquare == 2) {
                (this->*_zero)(_nRooks + !side, KING_ROOK_SQUARE[!side]);
                (this->*_one)(_nRooks + !side, KING_ROOK_SQUARE[!side] - 2);
            } else if (startSquare - finishSquare == 2) {
                (this->*_zero)(_nRooks + !side, QUEEN_ROOK_SQUARE[!side]);
                (this->*_one)(_nRooks + !side, QUEEN_ROOK_SQUARE[!side] + 3);
            }
        }

        cReLU();
    }

    void makeMove(U32 move) { _move<&Accumulator::setZero, &Accumulator::setOne>(move); }

    void unmakeMove(U32 move) { _move<&Accumulator::setOne, &Accumulator::setZero>(move); }

    void refresh() {
        kingPos = __builtin_ctzll(pieces[side]);
        std::copy(perspective_b0.begin(), perspective_b0.end(), l1);

        setOne(!side, __builtin_ctzll(pieces[!side]));
        for (size_t i = 2; i < 12; ++i) {
            U64 x = pieces[i];
            while (x) {
                setOne(i, popLSB(x));
            }
        }

        cReLU();
    }

    void setOne(U32 pieceType, U32 square) {
        U32 ind = index(kingPos, pieceType, square);
        __m256i w;
        __m256i l;
        for (size_t i = 0; i < 32; i += 16) {
            w = _mm256_loadu_si256((__m256i*)&perspective_w0[ind][i]);
            l = _mm256_loadu_si256((__m256i*)&l1[i]);
            _mm256_storeu_si256((__m256i*)&l1[i], _mm256_add_epi16(l, w));
        }
    }

    void setZero(U32 pieceType, U32 square) {
        U32 ind = index(kingPos, pieceType, square);
        __m256i w;
        __m256i l;
        for (size_t i = 0; i < 32; i += 16) {
            w = _mm256_loadu_si256((__m256i*)&perspective_w0[ind][i]);
            l = _mm256_loadu_si256((__m256i*)&l1[i]);
            _mm256_storeu_si256((__m256i*)&l1[i], _mm256_sub_epi16(l, w));
        }
    }

    void cReLU() {
        __m256i x, y;
        for (size_t i = 0; i < 32; i += 32) {
            x = _mm256_srai_epi16(
                _mm256_add_epi16(_mm256_max_epi16(_ZERO, _mm256_loadu_si256((__m256i*)&l1[i])), _HALF), 6);
            y = _mm256_srai_epi16(
                _mm256_add_epi16(_mm256_max_epi16(_ZERO, _mm256_loadu_si256((__m256i*)&l1[i + 16])), _HALF), 6);
            _mm256_storeu_si256((__m256i*)&cl1[i], cvtepi16_epi8(x, y));
        }
    }
};

inline int whiteIndex(U32 kingPos, U32 pieceType, U32 square) { return 704 * kingPos + 64 * (pieceType - 1) + square; }

inline int blackIndex(U32 kingPos, U32 pieceType, U32 square) {
    return 704 * (kingPos ^ 56) + 64 * (pieceType - 2 * (pieceType & 1)) + (square ^ 56);
}

typedef Accumulator<&whiteIndex, 0> White;
typedef Accumulator<&blackIndex, 1> Black;

class NNUE {
  private:
    White white;
    Black black;
    const bool* side;
    int pieceCount = 32;

  public:
    NNUE() {}

    NNUE(const U64* _pieces, const bool* _side) : side(_side) {
        white = White(_pieces);
        black = Black(_pieces);
    }

    void makeMove(U32 move) {
        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
        if (capturedPieceType != 15) {
            --pieceCount;
        }

        white.makeMove(move);
        black.makeMove(move);
    }

    void unmakeMove(U32 move) {
        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
        if (capturedPieceType != 15) {
            ++pieceCount;
        }

        white.unmakeMove(move);
        black.unmakeMove(move);
    }

    void fullRefresh() {
        pieceCount = 2;
        for (size_t i = 2; i < 12; ++i) {
            pieceCount += __builtin_popcountll(white.pieces[i]);
        }

        white.refresh();
        black.refresh();
    }

    int forward() {
        __m256i sum = _ZERO;
        __m256i l, w;

        int index = (pieceCount - 1) / 8;

        for (size_t i = 0; i < 32; i += 32) {
            l = _mm256_loadu_si256((__m256i*)&white.cl1[i]);
            w = _mm256_loadu_si256((__m256i*)&stacks_w0[index][32 * (*side) + i]);
            sum = _mm256_add_epi32(sum, madd_epi8(l, w));
        }
        for (size_t i = 0; i < 32; i += 32) {
            l = _mm256_loadu_si256((__m256i*)&black.cl1[i]);
            w = _mm256_loadu_si256((__m256i*)&stacks_w0[index][32 * (!(*side)) + i]);
            sum = _mm256_add_epi32(sum, madd_epi8(l, w));
        }
        return hsum_8x32(sum) + stacks_b0[index];
    }
};

#endif // NNUE_H_INCLUDED
