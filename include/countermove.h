#ifndef COUNTERMOVE_H_INCLUDED
#define COUNTERMOVE_H_INCLUDED

#include "constants.h"

class CounterMove
{
    public:
        U64 counterMoves[12][64] = {};

        CounterMove() {}

        void update(U64 move, U64 previousMove)
        {
            U32 pieceType = (previousMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 toSquare = (previousMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            counterMoves[pieceType][toSquare] = move;
        }

        void clear()
        {
            for (int i=0;i<12;i++)
            {
                for (int j=0;j<64;j++) {counterMoves[i][j] = 0;}
            }
        }
};

#endif // COUNTERMOVE_H_INCLUDED
