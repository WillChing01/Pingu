#ifndef NODE_H_INCLUDED
#define NODE_H_INCLUDED

#include <array>
#include <vector>

#include "constants.h"

class BaseNode
{
    protected:
        std::vector<std::pair<U32,int> > moveCache;
        int mvv;
        int lva;
        U64 victimBB;
        U64 attackerBB;
    public:
        bool side;

        bool inCheck;
        int numChecks;

        int staticEval;

        int depth;

        int bestScore;
        int score;
        U32 bestMove;
        U32 move;
};

class Node : public BaseNode
{
    protected:
        std::vector<U32> quietsPlayed;
        int numMoves = 0;

    public:
        U64 zHash;
        bool hashHit;

        Node()
        {
            //initiate the node.
        }

        void reset()
        {
            //reset the node for a new position.
        }
};

class QNode : public BaseNode
{
    public:
        QNode() {}

        void reset() {}
};

#endif // NODE_H_INCLUDED
