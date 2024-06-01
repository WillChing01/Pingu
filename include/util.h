#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include "constants.h"
#include "magic.h"
#include "knight.h"
#include "pawn.h"

namespace util
{
    inline bool isInCheck(bool side, U64 *pieces, U64 *occupied)
    {
        int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
        U64 b = occupied[0] | occupied[1];

        return
            (bool)(magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)])) ||
            (bool)(magicBishopAttacks(b,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)])) ||
            (bool)(knightAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKnights+(int)(!side)]) ||
            (bool)(pawnAttacks(pieces[_nKing+(int)(side)],side) & pieces[_nPawns+(int)(!side)]);
    }
}

#endif // UTIL_H_INCLUDED
