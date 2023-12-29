#ifndef NODE_H_INCLUDED
#define NODE_H_INCLUDED

#include <array>
#include <vector>

#include "constants.h"

class Node
{
    public:
        bool inCheck;
        int staticEval;
        std::vector<std::pair<U32,int> > moveCache;
        std::vector<U32> quietsPlayed;
        int numMoves = 0;

        U64 zHash;

    Node()
    {
        //initiate the node.
    }

    void reset()
    {
        //reset the node for a new position.
    }
};

#endif // NODE_H_INCLUDED
