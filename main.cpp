#include <iostream>

#include "board.h"
#include "perft.h"
#include "game.h"

using namespace std;

int main()
{
    populateMagicTables();
    populateRandomNums();
//    testInitialPosition();
//    testKiwipetePosition();

    searchSpeedTest(6);

    playCPU(6);

//    Board b;
//    b.setPositionFen("1r4k1/2pQ2pp/p1r5/3b4/P1p3p1/8/2P2P1P/4R1K1 b - - 5 39");
//    b.display();
//
//    bestMoves.clear();
//    int res = alphaBetaRoot(b,-INT_MAX,INT_MAX,6);
//    b.makeMove(bestMoves[0]);
//    b.display();

    system("pause");
    return 0;
}
