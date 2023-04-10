#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "bitboard.h"

static vector<U32> bestMoves;

int alphaBetaQuiescence(Board &b, int alpha, int beta)
{
    bool inCheck = b.generatePseudoQMoves(b.moveHistory.size() & 1);

    bool movesLeft = false;
    for (int i=0;i<(int)(b.moveBuffer.size());i++)
    {
        if (!(bool)(b.moveBuffer[i] & MOVEINFO_SHOULDCHECK_MASK)) {movesLeft=true; break;}
        else if (b.makeMove(b.moveBuffer[i]))
        {
            movesLeft=true;
            b.unmakeMove();
            break;
        }
    }

    if (movesLeft)
    {
        b.updateOccupied(); b.orderMoves();
        vector<pair<U32,int> > moveCache = b.scoredMoves;

        int score;

        if (!inCheck)
        {
            //do stand-pat check.
            int standPat=b.regularEval();
            if (standPat >= beta) {return beta;}
            if (alpha < standPat) {alpha = standPat;}
        }

        for (int i=0;i<(int)(moveCache.size());i++)
        {
            if (b.makeMove(moveCache[i].first))
            {
                score = -alphaBetaQuiescence(b, -beta, -alpha);
                b.unmakeMove();

                if (score >= beta) {return beta;}
                if (score > alpha) {alpha = score;}
            }
        }

        return alpha;
    }
    else
    {
        //no captures left. evaluate normally.
        return b.evaluateBoard();
    }
}

int alphaBeta(Board &b, int alpha, int beta, int depth)
{
    if (depth == 0) {return alphaBetaQuiescence(b, alpha, beta);}
    else
    {
        bool inCheck = b.generatePseudoMoves(b.moveHistory.size() & 1);

        bool movesLeft=false;
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

        if (movesLeft)
        {
            b.updateOccupied(); b.orderMoves(depth);
            vector<pair<U32,int> > moveCache = b.scoredMoves;
            int score;
            for (int i=0;i<(int)(moveCache.size());i++)
            {
                if (b.makeMove(moveCache[i].first))
                {
                    score = -alphaBeta(b, -beta, -alpha, depth-1);
                    b.unmakeMove();
                    if (score >= beta)
                    {
                        //beta cut-off. add killer move.
                        if (b.currentMove.capturedPieceType == 15 &&
                            b.killerMoves[depth][0] != moveCache[i].first &&
                            b.killerMoves[depth][1] != moveCache[i].first)
                        {
                            b.killerMoves[depth][1] = b.killerMoves[depth][0];
                            b.killerMoves[depth][0] = moveCache[i].first;
                        }
                        return beta;
                    }
                    if (score > alpha) {alpha = score;}
                }
            }
            return alpha;
        }
        else if (inCheck)
        {
            //checkmate.
            return -INT_MAX;
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

int alphaBetaRoot(Board &b, int alpha, int beta, int depth)
{
    if (depth == 0) {return alphaBetaQuiescence(b, alpha, beta);}
    else
    {
        bool inCheck = b.generatePseudoMoves(b.moveHistory.size() & 1);

        bool movesLeft=false;
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

        if (movesLeft)
        {
            b.updateOccupied(); b.orderMoves();
            vector<pair<U32,int> > moveCache = b.scoredMoves;
            int pvIndex = 0;
            int score;
            for (int itDepth = 1; itDepth <= depth; itDepth++)
            {
                alpha = -INT_MAX; beta = INT_MAX; bestMoves.clear();
                //try pv first.
                if (b.makeMove(moveCache[pvIndex].first))
                {
                    score = -alphaBeta(b, -beta, -alpha, itDepth-1);
                    b.unmakeMove();
                    if (score > alpha) {alpha = score; bestMoves.clear(); bestMoves.push_back(moveCache[pvIndex].first);}
                }

                for (int i=0;i<(int)(moveCache.size());i++)
                {
                    if (i==pvIndex) {continue;}
                    if (b.makeMove(moveCache[i].first))
                    {
                        score = -alphaBeta(b, -beta, -alpha, itDepth-1);
                        b.unmakeMove();
                        if (score > alpha) {alpha = score; bestMoves.clear(); bestMoves.push_back(moveCache[i].first); pvIndex = i;}
                    }
                }
            }
            return alpha;
        }
        else if (inCheck)
        {
            //checkmate.
            return -INT_MAX;
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

void searchSpeedTest(int depth)
{
    //plays initial position for 10 ply.
    Board b; b.display();

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0;i<10;i++)
    {
        alphaBetaRoot(b,-INT_MAX,INT_MAX,depth);
        if (bestMoves.size()==0) {break;}
        b.makeMove(bestMoves[0]);
        cout << i+1 << " " << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    b.display();
    cout << "Time (ms) - " << std::chrono::duration<double, std::milli>(t2-t1).count() << endl;
}

#endif // SEARCH_H_INCLUDED
