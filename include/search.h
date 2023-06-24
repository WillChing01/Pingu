#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <atomic>

#include "bitboard.h"
#include "transposition.h"

U32 storedBestMove = 0;
int storedBestScore = 0;
vector<U32> bestMoves;
vector<U32> pvMoves;

double timeLeft = 0; //milliseconds.
auto startTime = std::chrono::high_resolution_clock::now();
auto currentTime = std::chrono::high_resolution_clock::now();

std::atomic_bool isSearchAborted(false);
U32 totalNodes = 0;

inline bool checkTime()
{
    if (std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-startTime).count() > timeLeft)
    {
        isSearchAborted = true;
        return false;
    }
    else {return true;}
}

int alphaBetaQuiescence(Board &b, int alpha, int beta)
{
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0)
    {
        if (checkTime() == false) {return 0;}
    }
    if (isSearchAborted) {return 0;}

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
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0)
    {
        if (checkTime() == false) {return 0;}
    }
    if (isSearchAborted) {return 0;}

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
            //check transposition table for a previously calculated line.
            U64 bHash = b.zHashPieces ^ b.zHashState;
            if (ttProbe(bHash, tableEntry) == true)
            {
                if (tableEntry.depth >= depth)
                {
                    //PV node, score is exact.
                    if (tableEntry.isExact) {return tableEntry.evaluation;}
                    //score is a lower bound.
                    else if (tableEntry.isBeta) {if (tableEntry.evaluation >= beta) {return beta;}}
                    //all node, score is an upper bound.
                    else {if (tableEntry.evaluation < alpha) {return alpha;}}
                }

                b.updateOccupied(); b.orderMoves(depth, tableEntry.bestMove);
            }
            else
            {
                //no hash table hit.
                b.updateOccupied(); b.orderMoves(depth);
            }

            vector<pair<U32,int> > moveCache = b.scoredMoves;
            int score=alpha; bool isExact = false;
            int bestScore = -INT_MAX; U32 bestMove = 0;
            for (int i=0;i<(int)(moveCache.size());i++)
            {
                if (b.makeMove(moveCache[i].first))
                {
                    score = -alphaBeta(b, -beta, -alpha, depth-1);
                    b.unmakeMove();

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestMove = moveCache[i].first;
                    }

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

                        //update transposition table.
                        ttSave(bHash, depth, bestMove, bestScore, false, true);

                        return beta;
                    }
                    if (score > alpha) {alpha = score; isExact = true;}
                }
            }

            //update transposition table.
            ttSave(bHash, depth, bestMove, bestScore, isExact, false);

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
        rootCounter++;
        startTime = std::chrono::high_resolution_clock::now();

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

                //check if time is up.
                if (isSearchAborted) {break;}
                else
                {
                    storedBestMove = bestMoves[0];
                    storedBestScore = alpha;
                    b.unpackMove(storedBestMove);
                    cout << itDepth << " " << toCoord(b.currentMove.startSquare) << toCoord(b.currentMove.finishSquare) << " ";
                }
            }
            cout << endl;
            return storedBestScore;
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
    timeLeft = INT_MAX;
    for (int i=0;i<10;i++)
    {
        isSearchAborted = false;
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
