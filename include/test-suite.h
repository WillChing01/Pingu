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

        b.updateOccupied();
        b.updateAttackTables(0);
        b.updateAttackTables(1);
        b.updateAttacked();

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
    b.pieces[b._nQueens] = convertToBitboard(21);
    b.pieces[b._nQueens+1] = convertToBitboard(52);
    b.pieces[b._nBishops] = convertToBitboard(11) + convertToBitboard(12);
    b.pieces[b._nBishops+1] = convertToBitboard(40) + convertToBitboard(54);
    b.pieces[b._nKnights] = convertToBitboard(18) + convertToBitboard(36);
    b.pieces[b._nKnights+1] = convertToBitboard(41) + convertToBitboard(45);
    b.pieces[b._nPawns] += convertToBitboard(35) - convertToBitboard(11)
    + convertToBitboard(28) - convertToBitboard(12);
    b.pieces[b._nPawns+1] += convertToBitboard(25) - convertToBitboard(49)
    + convertToBitboard(44) - convertToBitboard(52)
    + convertToBitboard(46) - convertToBitboard(54)
    + convertToBitboard(23) - convertToBitboard(55);

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
