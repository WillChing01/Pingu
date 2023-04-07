#include <iostream>

#include "board.h"
#include "perft.h"
#include "search.h"

using namespace std;

int main()
{
    populateMagicTables();
//    testInitialPosition();
//    testKiwipetePosition();


//    searchSpeedTest(5);

    string temp;
    cout << "Pick a side (0/1): "; cin >> temp;

    bool side = temp == "0" ? false : true;

    Board b;
    int depth = 5; string startSquare, endSquare; int pieceType=0;

    if (side)
    {
        short res = alphaBetaRoot(b,-SHRT_MAX,SHRT_MAX,depth);
        cout << "Evaluation: " << res << endl;
        b.makeMove(bestMoves[0]);
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
    }
    b.display();

    while (true)
    {
        //player makes turn.
        cout << "Enter move: ";
        cin >> startSquare >> endSquare;

        for (int i=0;i<12;i++)
        {
            if ((b.pieces[i] & (1ull << toSquare(startSquare)))) {pieceType = i; break;}
        }

        b.moveBuffer.clear();
        b.updateOccupied();
        b.appendMove(pieceType, toSquare(startSquare), toSquare(endSquare));
        b.makeMove(b.moveBuffer[0]);
        b.moveBuffer.clear();
        b.display();

        short res = alphaBetaRoot(b,-SHRT_MAX,SHRT_MAX,depth);
        cout << "Evaluation: " << res << endl;
        if (bestMoves.size()==0) {break;}
        b.makeMove(bestMoves[0]);
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
        b.display();
    }

    system("pause");
    return 0;
}
