#ifndef PAWN_H_INCLUDED
#define PAWN_H_INCLUDED

#include "constants.h"

inline U64 pawnAttacks(U64 b, int side)
{
    U64 res=0;

    if (side==0)
    {
        //white.
        res=((b & NOT_A_FILE) << 7) | ((b & NOT_H_FILE) << 9);
    }
    else
    {
        res=((b & NOT_A_FILE) >> 9) | ((b & NOT_H_FILE) >> 7);
    }

    return res;
}

#endif // PAWN_H_INCLUDED
