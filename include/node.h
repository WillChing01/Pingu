#ifndef NODE_H_INCLUDED
#define NODE_H_INCLUDED

#include <array>
#include <vector>

#include "constants.h"

class BaseNode
{
    protected:
        int mvv;
        int lva;
        U64 victimBB;
        U64 attackerBB;
    public:
        bool side;

        bool inCheck;

        int staticEval;

        std::vector<std::pair<U32,int> > moveCache;
};

class Node : public BaseNode
{
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
