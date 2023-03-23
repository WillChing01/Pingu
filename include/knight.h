#ifndef KNIGHT_H_INCLUDED
#define KNIGHT_H_INCLUDED

#include "constants.h"

U64 knightAttacks(U64 b)
{
    U64 r1 = (b & NOT_A_FILE) >> 1;
    r1 |= (b & NOT_H_FILE) << 1;

    U64 r2 = (b & NOT_AB_FILE) >> 2;
    r2 |= (b & NOT_GH_FILE) << 2;

    U64 res = (r1 >> 16) | (r1 << 16);
    res |= (r2 >> 8) | (r2 << 8);

    return res;
}

#endif // KNIGHT_H_INCLUDED
