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
    string startSquare, endSquare;

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
        b.generatePseudoMoves(side);
        while (true)
        {
            cout << "Enter move: ";
            cin >> startSquare >> endSquare;

            bool legal = false;

            for (int i=0;i<(int)b.moveBuffer.size();i++)
            {
                if (b.makeMove(b.moveBuffer[i]))
                {
                    b.unpackMove(b.moveBuffer[i]);
                    if (b.currentMove.startSquare == (U32)toSquare(startSquare) && b.currentMove.finishSquare == (U32)toSquare(endSquare))
                    {
                        legal = true; break;
                    }
                    else
                    {
                        b.unmakeMove();
                    }
                }
            }

            if (!legal) {cout << "Illegal move." << endl;}
            else {break;}
        }

        b.moveBuffer.clear();
        b.display();
        bestMoves.clear();

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
        if (bestMoves.size()==0)
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
            b.makeMove(bestMoves[0]);
        }
        cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
        b.display();
        cout << b.phase << " " << b.shiftedPhase << endl;
    }
}

#endif // GAME_H_INCLUDED
