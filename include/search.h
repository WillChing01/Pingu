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
std::vector<U32> pvMoves;

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
    if ((totalNodes & 2047) == 0) {if (!checkTime()) {return 0;}}
    if (isSearchAborted) {return 0;}

    b.updateOccupied();
    bool side = b.moveHistory.size() & 1;
    bool inCheck = b.isInCheck(side);

    int bestScore = -MATE_SCORE;

    if (!inCheck)
    {
        //do stand-pat check.
        bestScore = b.evaluateBoard();
        if (bestScore > alpha)
        {
            if (bestScore >= beta) {return bestScore;}
            alpha = bestScore;
        }

        //generate regular tactical moves.
        b.moveBuffer.clear();
        b.generateCaptures(side, 0);
    }
    else
    {
        //generate check evasion.
        U32 numChecks = b.isInCheckDetailed(side);
        b.moveBuffer.clear();
        b.generateCaptures(side, numChecks);
        b.generateQuiets(side, numChecks);
    }

    int score;
    if (b.moveBuffer.size() == 0) {return inCheck ? -MATE_SCORE : bestScore;}
    //no need to update occupied, just did move-gen.
    std::vector<std::pair<U32,int> > moveCache = b.orderQMoves();

    for (const auto &[move,moveScore]: moveCache)
    {
        b.makeMove(move);
        score = -alphaBetaQuiescence(b, -beta, -alpha);
        b.unmakeMove();

        if (score > bestScore)
        {
            if (score > alpha)
            {
                if (score >= beta) {return score;}
                alpha = score;
            }
            bestScore = score;
        }
    }

    return bestScore;
}

int alphaBeta(Board &b, int alpha, int beta, int depth, int ply, bool nullMoveAllowed)
{
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0) {if (!checkTime()) {return 0;}}
    if (isSearchAborted) {return 0;}

    //check for draw by repetition.
    if (isDraw(b)) {return 0;}

    //qSearch at horizon.
    if (depth <= 0) {totalNodes--; return alphaBetaQuiescence(b, alpha, beta);}

    //main search.
    bool side = b.moveHistory.size() & 1;
    b.updateOccupied();
    bool inCheck = b.isInCheck(side);

    //probe hash table.
    U64 bHash = b.zHashPieces ^ b.zHashState;
    bool hashHit = ttProbe(bHash);
    U32 hashMove = hashHit ? tableEntry.bestMove : 0;

    //check for early TT cutoff.
    if (hashHit && tableEntry.depth >= depth)
    {
        //PV node.
        if (tableEntry.isExact) {return tableEntry.evaluation;}
        //Cut node.
        else if (tableEntry.isBeta) {if (tableEntry.evaluation >= beta) {return tableEntry.evaluation;}}
        //All node.
        else {if (tableEntry.evaluation <= alpha) {return tableEntry.evaluation;}}
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

    //setup scoring variables.
    int score; bool isExact = false;
    int bestScore = -MATE_SCORE; U32 bestMove = 0;
    int numMoves = 0;

    //try hash move.
    if (hashHit && b.isValidMove(hashMove, inCheck))
    {
        b.makeMove(hashMove);
        bestScore = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);
        b.unmakeMove();
        numMoves++;

        if (bestScore >= beta)
        {
            //beta cutoff.
            if (b.currentMove.capturedPieceType == 15)
            {
                //update killers.
                if (b.killerMoves[ply][0] != hashMove)
                {
                    b.killerMoves[ply][1] = b.killerMoves[ply][0];
                    b.killerMoves[ply][0] = hashMove;
                }

                //increase history score.
                if (depth >= 5)
                {
                    b.history[b.currentMove.pieceType][b.currentMove.finishSquare] += depth * depth;
                    if (b.history[b.currentMove.pieceType][b.currentMove.finishSquare] > HISTORY_MAX) {b.ageHistory();}
                }
            }

            //update transposition table.
            if (bestScore != MATE_SCORE) {ttSave(bHash, depth, hashMove, bestScore, false, true);}
            return bestScore;
        }
        if (bestScore > alpha) {alpha = bestScore; isExact = true;}
        bestMove = hashMove;
    }

    //internal iterative reduction on hash miss.
    if (!hashHit && depth > 3) {depth--;}

    //get number of checks for move-gen.
    U32 numChecks = 0;
    if (inCheck) {b.updateOccupied(); numChecks = b.isInCheckDetailed(side);}

    //generate tactical moves and play them.
    b.moveBuffer.clear();
    b.generateCaptures(side, numChecks);
    std::vector<std::pair<U32,int> > moveCache = b.orderCaptures();

    //good captures and promotions.
    U32 move;
    int ind = moveCache.size();
    for (int i=0;i<(int)(moveCache.size());i++)
    {
        move = moveCache[i].first;
        //check that capture is not hash move.
        if (hashHit && (move == hashMove)) {continue;}
        //exit when we get to bad captures.
        if (moveCache[i].second < 0) {ind = i; break;}

        b.makeMove(move);
        if (depth >= 2 && numMoves > 0)
        {
            //PV search.
            score = -alphaBeta(b, -alpha-1, -alpha, depth-1, ply+1, true);
            if (score > alpha && score < beta)
            {
                //full window re-search.
                score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);
            }
        }
        else {score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);}
        b.unmakeMove();
        numMoves++;

        if (score > bestScore)
        {
            if (score >= beta)
            {
                //beta cutoff.
                //update transposition table.
                if (score != MATE_SCORE) {ttSave(bHash, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //try killers.
    U32 killers[2] = {0,0};
    for (int i=0;i<2;i++)
    {
        move = b.killerMoves[ply][i];
        //check that killer is not hash move.
        if (hashHit && move == hashMove) {continue;}
        //check that killers not identical.
        if (i == 1 && move == b.killerMoves[ply][0]) {continue;}
        //check if killer is valid.
        if (!b.isValidMove(move, inCheck)) {continue;}
        killers[i] = move;

        b.makeMove(move);
        if (depth >= 2 && numMoves > 0)
        {
            //late move reductions (non pv nodes).
            if (depth >= 3 && (alpha == (beta - 1)) && numMoves >= 3 && !inCheck)
            {
                score = -alphaBeta(b, -beta, -alpha, depth-2, ply+1, true);
            }
            else {score = alpha + 1;}

            //PV search.
            //if lmr above alpha then research at original depth.
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
        numMoves++;

        if (score > bestScore)
        {
            if (score >= beta)
            {
                //beta cutoff.
                //update killers.
                if (b.killerMoves[ply][0] != move)
                {
                    b.killerMoves[ply][1] = b.killerMoves[ply][0];
                    b.killerMoves[ply][0] = move;
                }

                //update history.
                if (depth >= 5)
                {
                    bool shouldAge = false;

                    b.history[b.currentMove.pieceType][b.currentMove.finishSquare] += depth * depth;
                    if (b.history[b.currentMove.pieceType][b.currentMove.finishSquare] > HISTORY_MAX) {shouldAge = true;}

                    //decrement hash move.
                    if (hashHit &&
                        (((hashMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET) == 15))
                    {
                        U32 pieceType = (hashMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 finishSquare = (hashMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                        b.history[pieceType][finishSquare] -= depth * depth;
                        if (b.history[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
                    }

                    //decrement previous killer.
                    if (i > 0 && killers[0] != 0 && killers[0] != hashMove)
                    {
                        U32 pieceType = (killers[0] & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 finishSquare = (killers[0] & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                        b.history[pieceType][finishSquare] -= depth * depth;
                        if (b.history[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
                    }

                    if (shouldAge) {b.ageHistory();}
                }

                //update transposition table.
                if (score != MATE_SCORE) {ttSave(bHash, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //bad captures.
    for (int i=ind;i<(int)(moveCache.size());i++)
    {
        move = moveCache[i].first;
        //check that capture is not hash move.
        if (hashHit && (move == hashMove)) {continue;}

        b.makeMove(move);
        if (depth >= 2 && numMoves > 0)
        {
            //PV search.
            score = -alphaBeta(b, -alpha-1, -alpha, depth-1, ply+1, true);
            if (score > alpha && score < beta)
            {
                //full window re-search.
                score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);
            }
        }
        else {score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);}
        b.unmakeMove();
        numMoves++;

        if (score > bestScore)
        {
            if (score >= beta)
            {
                //beta cutoff.
                //update transposition table.
                if (score != MATE_SCORE) {ttSave(bHash, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //generate quiets and try them.
    b.moveBuffer.clear();
    b.generateQuiets(side, numChecks);
    moveCache = b.orderQuiets();

    for (int i=0;i<(int)(moveCache.size());i++)
    {
        move = moveCache[i].first;
        if (hashHit && (move == hashMove)) {continue;}
        if ((move == killers[0]) || (move == killers[1])) {continue;}
        
        b.makeMove(move);
        if (depth >= 2 && numMoves > 0)
        {
            //late move reductions (non pv nodes).
            if (depth >= 3 && (alpha == (beta - 1)) && numMoves >= 3 && !inCheck)
            {
                score = -alphaBeta(b, -beta, -alpha, depth-2, ply+1, true);
            }
            else {score = alpha + 1;}

            //PV search.
            //if lmr above alpha then research at original depth.
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
        numMoves++;

        if (score > bestScore)
        {
            if (score >= beta)
            {
                //beta cutoff.
                //update killers.
                if (b.killerMoves[ply][0] != move)
                {
                    b.killerMoves[ply][1] = b.killerMoves[ply][0];
                    b.killerMoves[ply][0] = move;
                }

                //update history.
                if (depth >= 5)
                {
                    bool shouldAge = false;

                    b.history[b.currentMove.pieceType][b.currentMove.finishSquare] += depth * depth;
                    if (b.history[b.currentMove.pieceType][b.currentMove.finishSquare] > HISTORY_MAX) {shouldAge = true;}

                    //decrement hash move.
                    if (hashHit &&
                        (((hashMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET) == 15))
                    {
                        U32 pieceType = (hashMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 finishSquare = (hashMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                        b.history[pieceType][finishSquare] -= depth * depth;
                        if (b.history[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
                    }

                    //decrement previous killers.
                    for (int j=0;j<2;j++)
                    {
                        if (killers[j] == 0 || killers[j] == hashMove) {continue;}
                        U32 pieceType = (killers[j] & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 finishSquare = (killers[j] & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                        b.history[pieceType][finishSquare] -= depth * depth;
                        if (b.history[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
                    }

                    //decrement previous quiets.
                    for (int j=0;j<i;j++)
                    {
                        if (moveCache[j].first == hashMove || moveCache[j].first == killers[0] || moveCache[j].first == killers[1]) {continue;}
                        U32 pieceType = (moveCache[j].first & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 finishSquare = (moveCache[j].first & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                        b.history[pieceType][finishSquare] -= depth * depth;
                        if (b.history[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
                    }

                    if (shouldAge) {b.ageHistory();}
                }

                //update transposition table.
                if (score != MATE_SCORE) {ttSave(bHash, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //stalemate or checkmate.
    if (numMoves == 0) {return inCheck ? -MATE_SCORE : 0;}

    //update transposition table.
    if (bestScore != -MATE_SCORE) {ttSave(bHash, depth, bestMove, bestScore, isExact, false);}
    return bestScore;
}

int alphaBetaRoot(Board &b, int depth)
{
    //track start-time of search.
    startTime = std::chrono::high_resolution_clock::now();

    //update TT 'age' and reset nodes.
    rootCounter++;
    totalNodes = 0;

    //generate moves.
    bool side = b.moveHistory.size() & 1;
    bool inCheck = b.generatePseudoMoves(side);

    //checkmate or stalemate.
    if (b.moveBuffer.size() == 0) {return inCheck ? -MATE_SCORE : 0;}

    //order moves using old history.
    std::vector<std::pair<U32,int> > moveCache = b.orderMoves(0);

    //reset history at root.
    b.clearHistory();

    //reset best score and best move.
    storedBestScore = -MATE_SCORE; storedBestMove = 0;
    int pvIndex = 0;

    //iterative deepening.
    for (int itDepth = 1; itDepth <= depth; itDepth++)
    {
        auto iterationStartTime = std::chrono::high_resolution_clock::now();

        U32 startNodes = totalNodes;
        totalNodes++;

        int score;
        int alpha = -MATE_SCORE-1; int beta = MATE_SCORE;

        //order moves for later depths by nodes searched.
        if (itDepth > 1)
        {
            //put best move of previous iter to front.
            moveCache[pvIndex].second = INT_MAX;

            //sort rest of moves.
            sort(moveCache.begin(), moveCache.end(), [](const auto &a, const auto &b) {return a.second > b.second;});
        }

        //play moves.
        for (int i=0;i<(int)(moveCache.size());i++)
        {
            U32 startMoveNodes = totalNodes;
            b.makeMove(moveCache[i].first);
            if (itDepth >= 2 && i > 0)
            {
                //PV search.
                score = -alphaBeta(b, -alpha-1, -alpha, itDepth-1, 1, true);
                if (score > alpha)
                {
                    //full window re-search.
                    score = -alphaBeta(b, -beta, -alpha, itDepth-1, 1, true);
                }
            }
            else {score = -alphaBeta(b, -beta, -alpha, itDepth-1, 1, true);}
            b.unmakeMove();
            if (score > alpha) {alpha = score; pvIndex = i;}
            moveCache[i].second = totalNodes - startMoveNodes;
        }

        //check if time is up.
        if (isSearchAborted) {break;}

        auto iterationFinishTime = std::chrono::high_resolution_clock::now();
        
        double iterationTime = std::chrono::duration<double, std::milli>(iterationFinishTime - iterationStartTime).count();
        double realTimeLeft = std::max(timeLeft - std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-startTime).count(), 0.);

        //store best move and score.
        storedBestMove = moveCache[pvIndex].first;
        storedBestScore = alpha;

        //display info.
        std::cout << "info" <<
            " depth " << itDepth <<
            " score cp " << storedBestScore <<
            " time " << (U32)(iterationTime) <<
            " nodes " << totalNodes - startNodes <<
            " nps " << (U32)((double)(totalNodes - startNodes) / (iterationTime / 1000.)) <<
            " pv";
        collectPVRoot(b, storedBestMove, itDepth);
        for (const auto pvMove: pvMoves)
        {
            std::cout << " " << moveToString(pvMove);
        } std::cout << std::endl;

        //break if checkmate is reached.
        if (storedBestScore == MATE_SCORE) {break;}

        //early exit if insufficient time for next iteration.
        //assume a branching factor of 2.
        if (iterationTime * 2. > realTimeLeft) {break;}
    }
    return storedBestScore;
}

#endif // SEARCH_H_INCLUDED
