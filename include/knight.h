#ifndef KNIGHT_H_INCLUDED
#define KNIGHT_H_INCLUDED

#include "constants.h"

U64 _knightAttacks[64] = {};

U64 aggregateKnightAttacks(U64 b)
{
    U64 r1 = (b & NOT_A_FILE) >> 1;
    r1 |= (b & NOT_H_FILE) << 1;

    U64 r2 = (b & NOT_AB_FILE) >> 2;
    r2 |= (b & NOT_GH_FILE) << 2;

    U64 res = (r1 >> 16) | (r1 << 16);
    res |= (r2 >> 8) | (r2 << 8);

    return res;
}

void cacheKnightAttacks()
{
    for (int i=0;i<64;i++)
    {
        _knightAttacks[i] = aggregateKnightAttacks(1ull << i);
    }
}

inline U64 knightAttacks(int square) {return _knightAttacks[square];}

#endif // KNIGHT_H_INCLUDED
