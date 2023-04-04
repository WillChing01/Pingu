#include <iostream>

#include "board.h"
#include "perft.h"
#include "search.h"

using namespace std;

int main()
{
    populateMagicTables();
    testInitialPosition();
    testKiwipetePosition();

//    Board b; b.display();
//
//    int depth = 5;
//    while (true)
//    {
//        short res = alphaBetaRoot(b,-SHRT_MAX,SHRT_MAX,depth);
//        cout << "Evaluation: " << res << endl;
//        if (bestMoves.size()==0) {break;}
//        b.makeMove(bestMoves[0]);
//        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
//        b.display();
//    }

    system("pause");
    return 0;
}
