#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include "constants.h"
#include "magic.h"
#include "knight.h"
#include "pawn.h"

namespace util
{
    inline bool isInCheck(bool side, const U64* pieces, const U64* occupied)
    {
        int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
        U64 b = occupied[0] | occupied[1];

        return
            (bool)(magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)])) ||
            (bool)(magicBishopAttacks(b,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)])) ||
            (bool)(knightAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKnights+(int)(!side)]) ||
            (bool)(pawnAttacks(pieces[_nKing+(int)(side)],side) & pieces[_nPawns+(int)(!side)]);
    }

    inline U32 isInCheckDetailed(bool side, const U64* pieces, const U64* occupied)
    {
        int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
        U64 b = occupied[0] | occupied[1];

        U64 inCheck = magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
        inCheck |= magicBishopAttacks(b,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
        inCheck |= knightAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKnights+(int)(!side)];
        inCheck |= pawnAttacks(pieces[_nKing+(int)(side)],side) & pieces[_nPawns+(int)(!side)];

        return __builtin_popcountll(inCheck);
    }

    inline U64 getCheckPiece(bool side, U32 square, const U64* pieces, const U64* occupied)
    {
        //assumes a single piece is giving check.
        U64 b = occupied[0] | occupied[1];

        if (U64 bishop = magicBishopAttacks(b,square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)])) {return bishop;}
        else if (U64 rook = magicRookAttacks(b,square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)])) {return rook;}
        else if (U64 knight = knightAttacks(1ull << square) & pieces[_nKnights+(int)(!side)]) {return knight;}
        else {return pawnAttacks(1ull << square,side) & pieces[_nPawns+(int)(!side)];}
    }

    inline U64 getBlockSquares(bool side, U32 square, const U64* pieces, const U64* occupied)
    {
        //assumes a single piece is giving check.
        U64 b = occupied[0] | occupied[1];

        if (U64 bishop = magicBishopAttacks(b, square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
        {
            return magicBishopAttacks(b, square) & magicBishopAttacks(b, __builtin_ctzll(bishop));
        }
        if (U64 rook = magicRookAttacks(b, square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
        {
            return magicRookAttacks(b, square) & magicRookAttacks(b, __builtin_ctzll(rook));
        }
        return 0;
    }
}

#endif // UTIL_H_INCLUDED
