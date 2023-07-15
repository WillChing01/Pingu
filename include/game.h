#ifndef GAME_H_INCLUDED
#define GAME_H_INCLUDED

#include "board.h"
#include "search.h"
#include "format.h"

void playCPU(double timePerMove)
{
    timeLeft = timePerMove;

    string temp;
    cout << "Pick a side (0/1): "; cin >> temp;

    bool side = temp == "0" ? false : true;

    Board b;
    string startSquare, endSquare;

    if (side)
    {
        isSearchAborted = false;
        int res = alphaBetaRoot(b,-MATE_SCORE,MATE_SCORE,64);
        cout << "Evaluation: " << res << endl;
        b.makeMove(storedBestMove);
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
    }
    b.display();

    while (true)
    {
        //player makes turn.
        b.generatePseudoMoves(side);

        if (b.moveBuffer.size() == 0) {break;}

        while (true)
        {
            cout << "Enter move: ";
            cin >> temp;
            U32 chessMove = stringToMove(b,temp);
            if (chessMove != 0) {b.makeMove(chessMove); break;}
            else {cout << "Illegal move." << endl;}
        }

        b.moveBuffer.clear();
        b.display();
        bestMoves.clear();

        int res;
        isSearchAborted = false; storedBestMove = 0;
        startTime = std::chrono::high_resolution_clock::now();
        res = alphaBetaRoot(b,-MATE_SCORE,MATE_SCORE,64);

        cout << "Evaluation: " << res << endl;
        if (storedBestMove==0)
        {
            b.moveBuffer.clear();
            b.generatePseudoMoves(!side);
            if (b.moveBuffer.size() == 0) {break;}
            b.makeMove(b.moveBuffer[0]);
        }
        else
        {
            b.makeMove(storedBestMove);
        }
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
        b.display();
        cout << b.phase << " " << b.shiftedPhase << endl;
    }
}

#endif // GAME_H_INCLUDED
