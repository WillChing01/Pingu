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
        int res = alphaBetaRoot(b,-INT_MAX,INT_MAX,64);
        cout << "Evaluation: " << res << endl;
        b.makeMove(storedBestMove);
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
    }
    b.display();

    while (true)
    {
        //player makes turn.
        b.generatePseudoMoves(side);

        //check if there are any legal moves left.
        bool movesLeft = false;
        for (int i=0;i<(int)b.moveBuffer.size();i++)
        {
            if (!(bool)(b.moveBuffer[i] & MOVEINFO_SHOULDCHECK_MASK)) {movesLeft=true; break;}
            else if (b.makeMove(b.moveBuffer[i]))
            {
                movesLeft=true;
                b.unmakeMove();
                break;
            }
        }

        if (!movesLeft) {break;}

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
        res = alphaBetaRoot(b,-INT_MAX,INT_MAX,64);

        cout << "Evaluation: " << res << endl;
        if (storedBestMove==0)
        {
            b.moveBuffer.clear();
            b.generatePseudoMoves(!side);
            bool gotMove=false;
            for (int i=0;i<(int)b.moveBuffer.size();i++)
            {
                if (b.makeMove(b.moveBuffer[i]))
                {
                    gotMove=true; break;
                }
            }

            if (!gotMove) {break;}
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
