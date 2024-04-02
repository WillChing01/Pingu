#ifndef KILLER_H_INCLUDED
#define KILLER_H_INCLUDED

#include "constants.h"

const int MAX_KILLER_PLY = 128;

class Killer
{
    public:
        U32 killerMoves[MAX_KILLER_PLY][2] = {};

        Killer() {}

        void update(U32 killerMove, int ply)
        {
            if (killerMoves[ply][0] != killerMove)
            {
                killerMoves[ply][1] = killerMoves[ply][0];
                killerMoves[ply][0] = killerMove;
            }
        }

        void clear()
        {
            for (int i=0;i<MAX_KILLER_PLY;i++)
            {
                killerMoves[i][0] = 0;
                killerMoves[i][1] = 0;
            }
        }
};

#endif // KILLER_H_INCLUDED
