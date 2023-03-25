#ifndef TEST-SUITE_H_INCLUDED
#define TEST-SUITE_H_INCLUDED

#include "board.h"

const long long INITIAL_POSITION_NODES[7] = {1,20,400,8902,197281,4865609,119060324};

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
        b.generatePseudoMoves(b.current.turn);
        b.testMoves();

        vector<moveInfo> moveCache = b.moveBuffer;

        for (int i=0;i<(int)moveCache.size();i++)
        {
            //execute the move.
            b.makeMove(moveCache[i]);

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

        return total;
    }
}

void testInitialPosition(int depth = 6)
{
    Board b;
    bool good=true;

    cout << "------------------------" << endl;
    cout << "PERFT - INITIAL POSITION" << endl;
    cout << "------------------------" << endl;

    for (int i=0;i<depth+1;i++)
    {
        cout << endl;
        cout << "DEPTH - " << i << endl;
        cout << "Expected - " << INITIAL_POSITION_NODES[i] << endl;

        long long result = perft(b,i);
        cout << "Received - " << result << endl;

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
}

#endif // TEST-SUITE_H_INCLUDED
