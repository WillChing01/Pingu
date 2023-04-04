#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "bitboard.h"

static vector<U32> bestMoves;

short alphaBetaQuiescence(Board &b, short alpha, short beta)
{
    b.moveBuffer.clear();
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
        vector<U32> moveCache = b.moveBuffer;
        short score;

        if (!inCheck)
        {
            //do stand-pat check.
            short standPat=b.regularEval()*(1-2*(b.moveHistory.size() & 1));
            if (standPat >= beta) {return beta;}
            if (alpha < standPat) {alpha = standPat;}
        }

        for (int i=0;i<(int)(moveCache.size());i++)
        {
            if (b.makeMove(moveCache[i]))
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

short alphaBeta(Board &b, short alpha, short beta, int depth)
{
    if (depth == 0) {return alphaBetaQuiescence(b, alpha, beta);}
    else
    {
        b.moveBuffer.clear();
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
            vector<U32> moveCache = b.moveBuffer;
            short score;
            for (int i=0;i<(int)(moveCache.size());i++)
            {
                if (b.makeMove(moveCache[i]))
                {
                    score = -alphaBeta(b, -beta, -alpha, depth-1);
                    b.unmakeMove();
                    if (score >= beta) {return beta;}
                    if (score > alpha) {alpha = score;}
                }
            }
            return alpha;
        }
        else if (inCheck)
        {
            //checkmate.
            if (b.moveHistory.size() & 1) {return SHRT_MAX;}
            else {return -SHRT_MAX;}
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

short alphaBetaRoot(Board &b, short alpha, short beta, int depth)
{
    bestMoves.clear();
    if (depth == 0) {return alphaBetaQuiescence(b, alpha, beta);}
    else
    {
        b.moveBuffer.clear();
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
            vector<U32> moveCache = b.moveBuffer;
            short score;
            for (int i=0;i<(int)(moveCache.size());i++)
            {
                if (b.makeMove(moveCache[i]))
                {
                    cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << " ";
                    score = -alphaBeta(b, -beta, -alpha, depth-1);
                    cout << score << endl;
                    b.unmakeMove();
                    if (score > alpha) {alpha = score; bestMoves.clear(); bestMoves.push_back(moveCache[i]);}
                }
            }
            return alpha;
        }
        else if (inCheck)
        {
            //checkmate.
            if (b.moveHistory.size() & 1) {return SHRT_MAX;}
            else {return -SHRT_MAX;}
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

short negaMax(Board &b, int depth)
{
    if (depth == 0) {return b.evaluateBoard();}
    else
    {
        int top = -SHRT_MAX;
        b.moveBuffer.clear();
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
            vector<U32> moveCache = b.moveBuffer;
            short score;
            for (int i=0;i<(int)(moveCache.size());i++)
            {
                if (b.makeMove(moveCache[i]))
                {
                    score = -negaMax(b,depth-1);
                    b.unmakeMove();
                    if (score > top) {top=score;}
                }
            }
            return top;
        }
        else if (inCheck)
        {
            //checkmate.
            if (b.moveHistory.size() & 1) {return SHRT_MAX;}
            else {return -SHRT_MAX;}
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

short negaMaxRoot(Board &b, int depth)
{
    bestMoves.clear();
    if (depth == 0) {return b.evaluateBoard();}
    else
    {
        int top = -SHRT_MAX;
        b.moveBuffer.clear();
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
            vector<U32> moveCache = b.moveBuffer;
            short score;
            for (int i=0;i<(int)(moveCache.size());i++)
            {
                if (b.makeMove(moveCache[i]))
                {
                    cout << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << " ";
                    score = -negaMax(b,depth-1);
                    cout << score << endl;
                    b.unmakeMove();
                    if (score == top) {bestMoves.push_back(moveCache[i]);}
                    if (score > top) {top=score; bestMoves.clear(); bestMoves.push_back(moveCache[i]);}
                }
            }
            return top;
        }
        else if (inCheck)
        {
            //checkmate.
            if (b.moveHistory.size() & 1) {return SHRT_MAX;}
            else {return -SHRT_MAX;}
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

#endif // SEARCH_H_INCLUDED
