#ifndef TEST-SUITE_H_INCLUDED
#define TEST-SUITE_H_INCLUDED

#include <chrono>

#include "board.h"

const long long INITIAL_POSITION_NODES[7] = {1,20,400,8902,197281,4865609,119060324};
const long long KIWIPETE_POSITION_NODES[6] = {1,48,2039,97862,4085603,193690690};

long long perft(Board &b, int depth, bool display=false, bool rootNode=true)
{
    if (depth==0)
    {
        if (display) {b.display();}
        return 1;
    }
    else
    {
        long long total=0;

        b.moveBuffer.clear();
        b.generatePseudoMoves(b.turn);

        vector<moveInfo> moveCache = b.moveBuffer;

        for (int i=0;i<(int)moveCache.size();i++)
        {
            //execute the move.
            bool isLegal = b.makeMove(moveCache[i]);

            if (isLegal==true)
            {
                //go one deeper.
                long long res=perft(b,depth-1,display,false);
                total+=res;
                if (rootNode==true)
                {
                    cout << toCoord(moveCache[i].startSquare) << toCoord(moveCache[i].finishSquare) << " : " << res << endl;
                }

                //unmake the move.
                b.unmakeMove();
            }
        }

        return total;
    }
}

void testInitialPosition(int depth = 6)
{
    Board b;
    b.display();

    bool good=true;

    cout << "------------------------" << endl;
    cout << "PERFT - INITIAL POSITION" << endl;
    cout << "------------------------" << endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i=0;i<depth+1;i++)
    {
        cout << endl;
        cout << "DEPTH - " << i << endl;
        cout << "Expected - " << INITIAL_POSITION_NODES[i] << endl;

        long long result = perft(b,i,false,i==depth);
        cout << "Received - " << result << endl;

        if (result!=INITIAL_POSITION_NODES[i])
        {
            cout << endl;
            cout << "INCONSISTENT - stopping procedure" << endl;
            good=false;
            break;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    if (good)
    {
        cout << endl;
        cout << "-----------------" << endl;
        cout << "ALL CHECKS PASSED" << endl;
        cout << "-----------------" << endl;
    }

    cout << "-----------------" << endl;
    cout << "TIME ELAPSED (ms)" << endl;
    cout << std::chrono::duration<double, std::milli>(t2-t1).count() << endl;
    cout << "-----------------" << endl;
}

void testKiwipetePosition(int depth = 5)
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

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i=0;i<depth+1;i++)
    {
        cout << endl;
        cout << "DEPTH - " << i << endl;
        cout << "Expected - " << KIWIPETE_POSITION_NODES[i] << endl;

        long long result = perft(b,i,false,i==depth);
        cout << "Received - " << result << endl;

        if (result!=KIWIPETE_POSITION_NODES[i])
        {
            cout << endl;
            cout << "INCONSISTENT - stopping procedure" << endl;
            good=false;
            break;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    if (good)
    {
        cout << endl;
        cout << "-----------------" << endl;
        cout << "ALL CHECKS PASSED" << endl;
        cout << "-----------------" << endl;
    }

    cout << "-----------------" << endl;
    cout << "TIME ELAPSED (ms)" << endl;
    cout << std::chrono::duration<double, std::milli>(t2-t1).count() << endl;
    cout << "-----------------" << endl;
}

#endif // TEST-SUITE_H_INCLUDED
