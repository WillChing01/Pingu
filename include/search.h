#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <iostream>
#include <atomic>
#include <algorithm>

#include "bitboard.h"
#include "transposition.h"
#include "evaluation.h"
#include "format.h"
#include "board.h"

const int nullMoveR = 2;
const int nullMoveDepthLimit = 3;

U32 storedBestMove = 0;
int storedBestScore = 0;
vector<U32> pvMoves;

double timeLeft = 0; //milliseconds.
auto startTime = std::chrono::high_resolution_clock::now();
auto currentTime = std::chrono::high_resolution_clock::now();

std::atomic_bool isSearchAborted(false);
U32 totalNodes = 0;

void collectPVChild(Board &b, int depth)
{
    U64 bHash = b.zHashPieces ^ b.zHashState;
    if (ttProbe(bHash) == true && depth > 0)
    {
        pvMoves.push_back(tableEntry.bestMove);
        b.makeMove(tableEntry.bestMove);
        collectPVChild(b, depth-1);
        b.unmakeMove();
    }
}

void collectPVRoot(Board &b, U32 bestMove, int depth)
{
    pvMoves.clear();
    pvMoves.push_back(bestMove);
    b.makeMove(bestMove);
    collectPVChild(b, depth-1);
    b.unmakeMove();
}

inline bool isDraw(Board &b)
{
    //check if current position has appeared in moveHistory.
    bool draw = false;
    U32 zHash = b.zHashPieces ^ b.zHashState;
    for (int i=(int)(b.moveHistory.size())-1;i>=0;i--)
    {
        if ((((b.moveHistory[i] & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET) >> 1) == (b._nPawns >> 1) ||
            ((b.moveHistory[i] & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET) != 15)
        {
            break;
        }
        else if (b.hashHistory[i] == zHash) {draw = true; break;}
    }
    return draw;
}

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

    if (b.moveBuffer.size() > 0)
    {
        int bestScore = -INT_MAX;

        if (!inCheck)
        {
            //do stand-pat check.
            bestScore=b.regularEval();
            if (bestScore >= beta) {return bestScore;}
            alpha = max(alpha,bestScore);
        }

        int score;
        b.updateOccupied();
        vector<pair<U32,int> > moveCache = b.orderQMoves();
        for (int i=0;i<(int)(moveCache.size());i++)
        {
            b.makeMove(moveCache[i].first);
            score = -alphaBetaQuiescence(b, -beta, -alpha);
            b.unmakeMove();

            if (score > bestScore)
            {
                if (score >= beta) {return score;}
                bestScore = score;
                alpha = max(alpha,score);
            }
        }

        return bestScore;
    }
    else
    {
        //no captures left. evaluate normally.
        return b.evaluateBoard();
    }
}

int alphaBeta(Board &b, int alpha, int beta, int depth, int ply, bool nullMoveAllowed)
{
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0)
    {
        if (checkTime() == false) {return 0;}
    }
    if (isSearchAborted) {return 0;}

    //check for draw.
    if (isDraw(b)) {return 0;}

    if (depth <= 0) {return alphaBetaQuiescence(b, alpha, beta);}

    bool inCheck = b.generatePseudoMoves(b.moveHistory.size() & 1);

    if (b.moveBuffer.size() > 0)
    {
        //check transposition table for a previously calculated line.
        U64 bHash = b.zHashPieces ^ b.zHashState;
        vector<pair<U32,int> > moveCache;
        if (ttProbe(bHash) == true)
        {
            if (tableEntry.depth >= depth)
            {
                //PV node, score is exact.
                if (tableEntry.isExact) {return tableEntry.evaluation;}
                //score is a lower bound.
                else if (tableEntry.isBeta) {if (tableEntry.evaluation >= beta) {return tableEntry.evaluation;}}
                //all node, score is an upper bound.
                else {if (tableEntry.evaluation <= alpha) {return tableEntry.evaluation;}}
            }
            b.updateOccupied();
            moveCache = b.orderMoves(ply, tableEntry.bestMove);
        }
        else
        {
            //no hash table hit.
            b.updateOccupied();
            moveCache = b.orderMoves(ply);
        }

        //null move pruning.
        if (nullMoveAllowed && !inCheck && depth >= nullMoveDepthLimit && b.phase > 0)
        {
            b.makeNullMove();
            int nullScore = -alphaBeta(b, -beta, -beta+1, depth-1-nullMoveR, ply+1, false);
            b.unmakeNullMove();

            //fail hard only for null move pruning.
            if (nullScore >= beta) {return beta;}
        }

        int score=alpha; bool isExact = false;
        int bestScore = -MATE_SCORE; U32 bestMove = 0;

        //pv search.
        //search first legal move with full window.
        b.makeMove(moveCache[0].first);
        bestScore = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);
        b.unmakeMove();
        if (bestScore >= beta)
        {
            //quiet beta cutoff.
            if (b.currentMove.capturedPieceType == 15)
            {
                //update killers.
                if (b.killerMoves[ply][0] != moveCache[0].first &&
                    b.killerMoves[ply][1] != moveCache[0].first)
                {
                    b.killerMoves[ply][1] = b.killerMoves[ply][0];
                    b.killerMoves[ply][0] = moveCache[0].first;
                }

                //increase history score.
                if (depth >= 5)
                {
                    b.history[b.currentMove.pieceType][b.currentMove.finishSquare] += depth*depth;
                    if (b.history[b.currentMove.pieceType][b.currentMove.finishSquare] > HISTORY_MAX) {b.ageHistory();}
                }
            }

            //update transposition table.
            if (score < MATE_SCORE) {ttSave(bHash, depth, moveCache[0].first, bestScore, false, true);}
            
            return bestScore;
        }
        if (bestScore > alpha) {alpha = bestScore; isExact = true;}
        bestMove = moveCache[0].first;

        //search all other moves with initial null window.
        for (int i=1;i<(int)(moveCache.size());i++)
        {
            b.makeMove(moveCache[i].first);
            if (depth >= 2)
            {
                //late move reductions.
                if (depth >= 3 && (alpha == (beta - 1)) && i >= 3 && !inCheck &&
                    b.currentMove.capturedPieceType == 15 &&
                    b.currentMove.pieceType == b.currentMove.finishPieceType)
                {
                    score = -alphaBeta(b, -beta, -alpha, depth-2, ply+1, true);
                }
                else {score = alpha + 1;}

                if (score > alpha)
                {
                    score = -alphaBeta(b, -alpha-1, -alpha, depth-1, ply+1, true);
                    if (score > alpha && score < beta)
                    {
                        score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);
                    }
                }
            }
            else {score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);}
            b.unmakeMove();

            if (score > bestScore)
            {
                if (score >= beta)
                {
                    //quiet beta cutoff.
                    if (b.currentMove.capturedPieceType == 15)
                    {
                        //update killers.
                        if (b.killerMoves[ply][0] != moveCache[i].first &&
                            b.killerMoves[ply][1] != moveCache[i].first)
                        {
                            b.killerMoves[ply][1] = b.killerMoves[ply][0];
                            b.killerMoves[ply][0] = moveCache[i].first;
                        }

                        if (depth >= 5)
                        {
                            bool shouldAge = false;

                            //increase history score.
                            b.history[b.currentMove.pieceType][b.currentMove.finishSquare] += depth * depth;
                            if (b.history[b.currentMove.pieceType][b.currentMove.finishSquare] > HISTORY_MAX) {shouldAge = true;}

                            //decrease history score of previous quiets.
                            for (int j=0;j<i;j++)
                            {
                                if ((moveCache[j].first & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET != 15) {continue;}
                                U32 pieceType = (moveCache[j].first & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                                U32 finishSquare = (moveCache[j].first & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                                b.history[pieceType][finishSquare] -= depth * depth;
                                if (b.history[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
                            }

                            if (shouldAge) {b.ageHistory();}
                        }
                    }

                    //update transposition table.
                    if (score < MATE_SCORE) {ttSave(bHash, depth, moveCache[i].first, score, false, true);}

                    return score;
                }
                if (score > alpha) {alpha = score; isExact = true;}
                bestScore = score;
                bestMove = moveCache[i].first;
            }
        }

        //update transposition table.
        if (bestScore != -MATE_SCORE) {ttSave(bHash, depth, bestMove, bestScore, isExact, false);}

        return bestScore;
    }
    else if (inCheck)
    {
        //checkmate.
        return -MATE_SCORE;
    }
    else
    {
        //stalemate.
        return 0;
    }
}

int alphaBetaRoot(Board &b, int alpha, int beta, int depth)
{
    if (depth == 0) {return alphaBetaQuiescence(b, alpha, beta);}
    else
    {
        rootCounter++;
        startTime = std::chrono::high_resolution_clock::now();

        //reset history at root.
        b.clearHistory();

        bool inCheck = b.generatePseudoMoves(b.moveHistory.size() & 1);

        if (b.moveBuffer.size() > 0)
        {
            b.updateOccupied();
            vector<pair<U32,int> > moveCache = b.orderMoves(0);
            int pvIndex = 0;
            int score;
            for (int itDepth = 1; itDepth <= depth; itDepth++)
            {
                auto iterationStartTime = std::chrono::high_resolution_clock::now();
                alpha = -MATE_SCORE-1; beta = MATE_SCORE; U32 bestMove = 0;
                //try pv first.
                b.makeMove(moveCache[pvIndex].first);
                score = -alphaBeta(b, -beta, -alpha, itDepth-1, 1, true);
                b.unmakeMove();
                if (score > alpha) {alpha = score; bestMove = moveCache[pvIndex].first;}

                for (int i=0;i<(int)(moveCache.size());i++)
                {
                    if (i==pvIndex) {continue;}
                    b.makeMove(moveCache[i].first);
                    score = -alphaBeta(b, -beta, -alpha, itDepth-1, 1, true);
                    b.unmakeMove();
                    if (score > alpha) {alpha = score; bestMove = moveCache[i].first; pvIndex = i;}
                }

                //check if time is up.
                if (isSearchAborted) {break;}
                else
                {
                    storedBestMove = bestMove;
                    storedBestScore = alpha;
                    b.unpackMove(storedBestMove);
                    cout << "info depth " << itDepth << " score cp " << storedBestScore << " pv";
                    collectPVRoot(b, storedBestMove, itDepth);
                    for (int i=0;i<(int)pvMoves.size();i++)
                    {
                        cout << " " << moveToString(pvMoves[i]);
                    } cout << endl;
                    //break if checkmate is reached.
                    if (storedBestScore == MATE_SCORE) {break;}
                }

                auto iterationFinishTime = std::chrono::high_resolution_clock::now();

                double iterationTime = std::chrono::duration<double, std::milli>(iterationFinishTime - iterationStartTime).count();
                double realTimeLeft = max(timeLeft - std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-startTime).count(), 0.);

                //assume each extra iteration is at least twice as long.
                if (iterationTime * 2. > realTimeLeft) {break;}

            }
            return storedBestScore;
        }
        else if (inCheck)
        {
            //checkmate.
            return -MATE_SCORE;
        }
        else
        {
            //stalemate.
            return 0;
        }
    }
}

#endif // SEARCH_H_INCLUDED
