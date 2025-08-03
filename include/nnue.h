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
    alignas(32) std::vector<std::array<short, 32>> l1{256};
    alignas(32) char cl1[32] = {};
    const U64* pieces;
    int kingPos = 0;
    int ply = 0;

    Accumulator() {}

    Accumulator(const U64* _pieces) : pieces(_pieces) {}

    void makeMove(U32 move) {
        while ((int)l1.size() <= ply + 1) {
            l1.emplace_back();
        }

        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        if (pieceType == side) {
            ++ply;
            refresh();
            return;
        }

        __m256i buffer[2] = {
            _mm256_loadu_si256((__m256i*)(&l1[ply][0])),
            _mm256_loadu_si256((__m256i*)(&l1[ply][16])),
        };

        // for (int i = 0; i < 32; i += 16) {
        //     __m256i x = _mm256_loadu_si256((__m256i*)(&l1[ply][i]));
        //     _mm256_storeu_si256((__m256i*)(&l1[ply + 1][i]), x);
        // }
        ++ply;

        U32 startSquare = (move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
        U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
        U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
        bool enPassant = move & MOVEINFO_ENPASSANT_MASK;

        setZero(pieceType, startSquare, buffer);
        setOne(finishPieceType, finishSquare, buffer);
        if (capturedPieceType != 15) {
            setZero(capturedPieceType, finishSquare + enPassant * (-8 + 16 * (pieceType & 1)), buffer);
        }

        // check for castles.
        if (pieceType == (!side)) {
            if (finishSquare - startSquare == 2) {
                setZero(_nRooks + !side, KING_ROOK_SQUARE[!side], buffer);
                setOne(_nRooks + !side, KING_ROOK_SQUARE[!side] - 2, buffer);
            } else if (startSquare - finishSquare == 2) {
                setZero(_nRooks + !side, QUEEN_ROOK_SQUARE[!side], buffer);
                setOne(_nRooks + !side, QUEEN_ROOK_SQUARE[!side] + 3, buffer);
            }
        }

        storeBuffer(l1[ply], buffer);
    }

    void unmakeMove(U32 move) {
        --ply;
        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        if (pieceType == side) {
            kingPos = __builtin_ctzll(pieces[side]);
        }
    }

    void refresh() {
        kingPos = __builtin_ctzll(pieces[side]);

        __m256i buffer[2] = {
            _mm256_loadu_si256((__m256i*)(&perspective_b0[0])),
            _mm256_loadu_si256((__m256i*)(&perspective_b0[16])),
        };

        setOne(!side, __builtin_ctzll(pieces[!side]), buffer);
        for (size_t i = 2; i < 12; ++i) {
            U64 x = pieces[i];
            while (x) {
                setOne(i, popLSB(x), buffer);
            }
        }

        storeBuffer(l1[ply], buffer);
    }

    void setOne(U32 pieceType, U32 square, __m256i* buffer) {
        U32 ind = index(kingPos, pieceType, square);
        __m256i w;
        for (size_t i = 0; i < 2; ++i) {
            w = _mm256_loadu_si256((__m256i*)&perspective_w0[ind][16 * i]);
            buffer[i] = _mm256_add_epi16(buffer[i], w);
        }
    }

    void setZero(U32 pieceType, U32 square, __m256i* buffer) {
        U32 ind = index(kingPos, pieceType, square);
        __m256i w;
        for (size_t i = 0; i < 2; ++i) {
            w = _mm256_loadu_si256((__m256i*)&perspective_w0[ind][16 * i]);
            buffer[i] = _mm256_sub_epi16(buffer[i], w);
        }
    }

    void storeBuffer(std::array<short, 32>& layer, __m256i* buffer) {
        for (size_t i = 0; i < 2; ++i) {
            _mm256_storeu_si256((__m256i*)&layer[16 * i], buffer[i]);
        }
    }

    void setOne(U32 pieceType, U32 square) {
        U32 ind = index(kingPos, pieceType, square);
        __m256i w;
        __m256i l;
        for (size_t i = 0; i < 32; i += 16) {
            w = _mm256_loadu_si256((__m256i*)&perspective_w0[ind][i]);
            l = _mm256_loadu_si256((__m256i*)&l1[ply][i]);
            _mm256_storeu_si256((__m256i*)&l1[ply][i], _mm256_add_epi16(l, w));
        }
    }

    void setZero(U32 pieceType, U32 square) {
        U32 ind = index(kingPos, pieceType, square);
        __m256i w;
        __m256i l;
        for (size_t i = 0; i < 32; i += 16) {
            w = _mm256_loadu_si256((__m256i*)&perspective_w0[ind][i]);
            l = _mm256_loadu_si256((__m256i*)&l1[ply][i]);
            _mm256_storeu_si256((__m256i*)&l1[ply][i], _mm256_sub_epi16(l, w));
        }
    }

    void cReLU() {
        __m256i x, y;
        for (size_t i = 0; i < 32; i += 32) {
            x = _mm256_srai_epi16(
                _mm256_add_epi16(_mm256_max_epi16(_ZERO, _mm256_loadu_si256((__m256i*)&l1[ply][i])), _HALF), 6);
            y = _mm256_srai_epi16(
                _mm256_add_epi16(_mm256_max_epi16(_ZERO, _mm256_loadu_si256((__m256i*)&l1[ply][i + 16])), _HALF), 6);
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
        white.cReLU();
        black.cReLU();

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
