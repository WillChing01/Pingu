#ifndef PERFT_H_INCLUDED
#define PERFT_H_INCLUDED

#include <chrono>

#include "constants.h"
#include "board.h"

U64 childPerft(Board &b, int depth)
{
    if (depth == 0)
    {
        return 1;
    }
    else if (depth == 1)
    {
        b.generatePseudoMoves(b.moveHistory.size() & 1);
        return b.moveBuffer.size();
    }
    else
    {
        U64 total = 0;
        b.generatePseudoMoves(b.moveHistory.size() & 1);
        vector<U32> moveCache = b.moveBuffer;

        for (int i=0;i<(int)moveCache.size();i++)
        {
            b.makeMove(moveCache[i]);
            total += childPerft(b,depth-1);
            b.unmakeMove();
        }

        return total;
    }
}

U64 perft(Board &b, int depth, bool verbose = true)
{
    if (depth == 0)
    {
        return 1;
    }
    else if (depth == 1)
    {
        b.generatePseudoMoves(b.moveHistory.size() & 1);
        return b.moveBuffer.size();
    }
    else
    {
        U64 total = 0;
        b.generatePseudoMoves(b.moveHistory.size() & 1);
        vector<U32> moveCache = b.moveBuffer;

        for (int i=0;i<(int)moveCache.size();i++)
        {
            b.makeMove(moveCache[i]);
            U64 nodes = childPerft(b,depth-1);
            total += nodes;
            b.unmakeMove();
            if (verbose)
            {
                std::cout << toCoord((moveCache[i] & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET)
                          << toCoord((moveCache[i] & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET)
                          << " : " << nodes
                          << std::endl;
            }
        }

        return total;
    }
}

#endif // PERFT_H_INCLUDED
