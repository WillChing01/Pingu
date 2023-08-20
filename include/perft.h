#ifndef PERFT_H_INCLUDED
#define PERFT_H_INCLUDED

#include <chrono>

#include "constants.h"
#include "board.h"

const unsigned long long INITIAL_POSITION_NODES[7] = {1,20,400,8902,197281,4865609,119060324};
const unsigned long long KIWIPETE_POSITION_NODES[6] = {1,48,2039,97862,4085603,193690690};

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

bool testInitialPosition(int depth = 6)
{
    Board b;
    b.display();

    bool good=true;

    cout << "------------------------" << endl;
    cout << "PERFT - INITIAL POSITION" << endl;
    cout << "------------------------" << endl;

    for (int i=0;i<depth+1;i++)
    {
        cout << endl;
        cout << "DEPTH - " << i << endl;
        cout << "Expected - " << INITIAL_POSITION_NODES[i] << endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        unsigned long long result = i==depth ? perft(b,i) : childPerft(b,i);
        auto t2 = std::chrono::high_resolution_clock::now();
        cout << "Nodes searched - " << result << endl;
        cout << "Time (ms) - " << std::chrono::duration<double, std::milli>(t2-t1).count() << endl;

        if (result!=INITIAL_POSITION_NODES[i])
        {
            cout << endl;
            cout << "INCONSISTENT - stopping procedure" << endl;
            good=false;
            break;
        }
    }

    if (good)
    {
        cout << endl;
        cout << "-----------------" << endl;
        cout << "ALL CHECKS PASSED" << endl;
        cout << "-----------------" << endl;
    }
    return good;
}

bool testKiwipetePosition(int depth = 5)
{
    Board b;

    //set up the pieces.
    b.pieces[b._nQueens] = 1ull << (21);
    b.pieces[b._nQueens+1] = 1ull << (52);
    b.pieces[b._nBishops] = (1ull << (11)) + (1ull << (12));
    b.pieces[b._nBishops+1] = (1ull << (40)) + (1ull << (54));
    b.pieces[b._nKnights] = (1ull << (18)) + (1ull << (36));
    b.pieces[b._nKnights+1] = (1ull << (41)) + (1ull << (45));
    b.pieces[b._nPawns] += (1ull << (35)) - (1ull << (11))
    + (1ull << (28)) - (1ull << (12));
    b.pieces[b._nPawns+1] += (1ull << (25)) - (1ull << (49))
    + (1ull << (44)) - (1ull << (52))
    + (1ull << (46)) - (1ull << (54))
    + (1ull << (23)) - (1ull << (55));

    bool good=true;

    cout << "------------------------" << endl;
    cout << "PERFT - KIWIPETE POSITION" << endl;
    cout << "------------------------" << endl;

    for (int i=0;i<depth+1;i++)
    {
        cout << endl;
        cout << "DEPTH - " << i << endl;
        cout << "Expected - " << KIWIPETE_POSITION_NODES[i] << endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        unsigned long long result = i==depth ? perft(b,i) : childPerft(b,i);
        auto t2 = std::chrono::high_resolution_clock::now();
        cout << "Nodes searched - " << result << endl;
        cout << "Time (ms) - " << std::chrono::duration<double, std::milli>(t2-t1).count() << endl;

        if (result!=KIWIPETE_POSITION_NODES[i])
        {
            cout << endl;
            cout << "INCONSISTENT - stopping procedure" << endl;
            good=false;
            break;
        }
    }

    if (good)
    {
        cout << endl;
        cout << "-----------------" << endl;
        cout << "ALL CHECKS PASSED" << endl;
        cout << "-----------------" << endl;
    }
    return good;
}

#endif // PERFT_H_INCLUDED
