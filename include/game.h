#ifndef GAME_H_INCLUDED
#define GAME_H_INCLUDED

#include "board.h"
#include "search.h"

void playCPU(int depth)
{
    string temp;
    cout << "Pick a side (0/1): "; cin >> temp;

    bool side = temp == "0" ? false : true;

    Board b;
    string startSquare, endSquare; int pieceType=0;

    if (side)
    {
        int res = alphaBetaRoot(b,-INT_MAX,INT_MAX,depth);
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

        int res;
        if (b.shiftedPhase < 200)
        {
            res = alphaBetaRoot(b,-INT_MAX,INT_MAX,depth+1);
        }
        else if (b.shiftedPhase < 150)
        {
            res = alphaBetaRoot(b,-INT_MAX,INT_MAX,depth+2);
        }
        else if (b.shiftedPhase < 100)
        {
            res = alphaBetaRoot(b,-INT_MAX,INT_MAX,depth+4);
        }
        else
        {
            res = alphaBetaRoot(b,-INT_MAX,INT_MAX,depth);
        }
        cout << "Evaluation: " << res << endl;
        if (bestMoves.size()==0) {break;}
        b.makeMove(bestMoves[0]);
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
        b.display();
        cout << b.phase << " " << b.shiftedPhase << endl;
    }
}

#endif // GAME_H_INCLUDED
