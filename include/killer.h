#ifndef KILLER_H_INCLUDED
#define KILLER_H_INCLUDED

#include "constants.h"

U32 killerMoves[128][2] = {};

inline void updateKillers(U32 killerMove, int ply)
{
    if (killerMoves[ply][0] != killerMove)
    {
        killerMoves[ply][1] = killerMoves[ply][0];
        killerMoves[ply][0] = killerMove;
    }
}

#endif // KILLER_H_INCLUDED
