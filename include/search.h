#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <iostream>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "bitboard.h"
#include "transposition.h"
#include "evaluation.h"
#include "format.h"
#include "board.h"
#include "validate.h"
#include "movepicker.h"

const int maximumPruningDepth = 8;

const int inverseFutilityMargin = 120;
const int inverseFutilityDepthLimit = 8;

const int nullMoveR = 2;
const int nullMoveDepthLimit = 3;

static const int futilityDepthLimit = 2;
const std::array<int, futilityDepthLimit> futilityMargins = {150, 400};

static const int lateMovePruningDepthLimit = 4;
const std::array<int, lateMovePruningDepthLimit> lateMovePruningMargins = {6, 10, 14, 18};

const std::array<int, 3> aspirationDelta = {50, 200, 2 * MATE_SCORE};
const std::array<int, 4> betaDelta = {1, 50, 200, 2 * MATE_SCORE};

U32 storedBestMove = 0;
int storedBestScore = 0;
std::vector<U32> pvMoves;

double timeLeft = 0; //milliseconds.
auto startTime = std::chrono::high_resolution_clock::now();
auto currentTime = std::chrono::high_resolution_clock::now();

std::atomic_bool isSearchAborted(false);
U32 totalNodes = 0;

U32 lastIterTime = 0;
U32 lastIterNodes = 0;
U32 lastIterNps = 0;

bool isGameOver = false;

inline int formatScore(int score)
{
    return
        score > MATE_BOUND ? (MATE_SCORE - score + 1) / 2 :
        score < -MATE_BOUND ? (-MATE_SCORE - score) / 2 :
        score;
}

void collectPVChild(Board &b, int depth)
{
    U64 bHash = b.zHashPieces ^ b.zHashState;
    U64 hashInfo = ttProbe(bHash);
    if (hashInfo && depth > 0)
    {
        U32 hashMove = getHashMove(hashInfo);
        pvMoves.push_back(hashMove);
        b.makeMove(hashMove);
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

inline bool isDrawByRepetition(Board &b)
{
    //check if current position has appeared in moveHistory.
    U32 zHash = b.zHashPieces ^ b.zHashState;
    int finishInd = b.irrevMoveInd.size() ? b.irrevMoveInd.back() : -1;
    for (int i=(int)(b.hashHistory.size())-4;i>finishInd;i-=2)
    {
        if (b.hashHistory[i] == zHash) {return true;}
    }
    return false;
}

inline bool isDrawByMaterial(Board &b)
{
    return b.phase <= 1 && !(b.pieces[_nPawns] | b.pieces[_nPawns+1]);
}

inline bool isDrawByFifty(Board &b)
{
    //check if draw by fifty move rule.
    int lastIndex = b.moveHistory.size() - 1;
    int diff = b.irrevMoveInd.size() > 0 ? lastIndex - b.irrevMoveInd.back() : lastIndex;
    return diff >= 100;
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

inline int alphaBetaQuiescence(Board &b, int ply, int alpha, int beta)
{
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0) {if (!checkTime()) {return 0;}}
    if (isSearchAborted) {return 0;}

    //draw by insufficient material.
    if (isDrawByMaterial(b)) {return 0;}

    bool inCheck = util::isInCheck(b.side, b.pieces, b.occupied);
    int bestScore = -MATE_SCORE + ply;

    //stand pat.
    if (!inCheck)
    {
        bestScore = b.evaluateBoard();
        if (bestScore > alpha)
        {
            if (bestScore >= beta) {return bestScore;}
            alpha = bestScore;
        }
    }

    U32 numChecks = inCheck ? util::isInCheckDetailed(b.side, b.pieces, b.occupied) : 0;
    QMovePicker movePicker(&b, ply, numChecks);

    //loop through moves and search them.
    while (U32 move = movePicker.getNext())
    {
        b.makeMove(move);
        int score = -alphaBetaQuiescence(b, ply+1, -beta, -alpha);
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

inline int alphaBeta(Board &b, int alpha, int beta, int depth, int ply, bool nullMoveAllowed)
{
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0) {if (!checkTime()) {return 0;}}
    if (isSearchAborted) {return 0;}

    //check for draw.
    if (isDrawByRepetition(b) || isDrawByMaterial(b) || isDrawByFifty(b)) {return 0;}

    //probe hash table.
    U64 bHash = b.zHashPieces ^ b.zHashState;
    U64 hashInfo = ttProbe(bHash);

    //check for early TT cutoff.
    if (hashInfo && getHashDepth(hashInfo) >= depth)
    {
        //PV node.
        if (getHashExactFlag(hashInfo)) {return getHashEval(hashInfo, ply);}
        //Cut node.
        else if (getHashBetaFlag(hashInfo))
        {
            int hashEval = getHashEval(hashInfo, ply);
            if (hashEval >= beta) {return hashEval;}
        }
        //All node.
        else
        {
            int hashEval = getHashEval(hashInfo, ply);
            if (hashEval <= alpha) {return hashEval;}
        }
    }

    //qSearch at horizon.
    if (depth <= 0) {totalNodes--; return alphaBetaQuiescence(b, ply, alpha, beta);}

    //main search.
    bool inCheck = util::isInCheck(b.side, b.pieces, b.occupied);

    //get static evaluation.
    int staticEval = 0;
    bool canPrune = (alpha == beta - 1) && !inCheck && abs(alpha) < MATE_BOUND;
    if (canPrune && depth <= maximumPruningDepth) {staticEval = b.evaluateBoard();}

    //inverse futility pruning.
    if (canPrune && depth <= inverseFutilityDepthLimit)
    {
        int margin = inverseFutilityMargin * depth;
        if (staticEval - margin >= beta) {return beta;}
    }

    //null move pruning.
    if (hashInfo && getHashDepth(hashInfo) >= depth-nullMoveR-depth/6 && !getHashBetaFlag(hashInfo) && getHashEval(hashInfo, ply) < beta) {nullMoveAllowed = false;}
    if (nullMoveAllowed && !inCheck && depth >= nullMoveDepthLimit &&
        (b.occupied[b.side] ^ b.pieces[_nKing+b.side] ^ b.pieces[_nPawns+b.side]))
    {
        b.makeNullMove();
        int nullScore = -alphaBeta(b, -beta, -beta+1, depth-1-nullMoveR-depth/6, ply+1, false);
        b.unmakeNullMove();

        //fail hard only for null move pruning.
        if (nullScore >= beta) {return beta;}
    }

    //internal iterative reduction on hash miss.
    if (!hashInfo && depth > 3) {depth--;}

    //setup scoring variables.
    int score = 0; bool isExact = false;
    int bestScore = -MATE_SCORE; U32 bestMove = 0;
    int movesPlayed = 0; int quietsPlayed = 0;
    U32 numChecks = inCheck ? util::isInCheckDetailed(b.side, b.pieces, b.occupied) : 0;

    MovePicker movePicker(&b, ply, numChecks, getHashMove(hashInfo));

    bool canLateMovePrune = canPrune && depth <= lateMovePruningDepthLimit;

    bool canFutilityPrune = canPrune && depth <= futilityDepthLimit &&
                            staticEval + futilityMargins[depth-1] <= alpha;

    //loop through moves and search them.
    while (U32 move = movePicker.getNext())
    {
        //late move pruning.
        if (canLateMovePrune && quietsPlayed > lateMovePruningMargins[depth-1]) {break;}

        //futility pruning.
        if (canFutilityPrune && movesPlayed > 0 && movePicker.stage == QUIET_MOVES && !b.isCheckingMove(move)) {continue;}

        //search with PVS. Research if reductions do not fail low.
        b.makeMove(move);

        //late move reductions.
        int reduction = 0;
        bool canLateMoveReduce = !inCheck && alpha == beta-1;
        switch(movePicker.stage)
        {
            case HASH_MOVE:
                break;
            case GOOD_CAPTURES:
                break;
            case KILLER_MOVES:
                if (canLateMoveReduce && depth >= 3 && movesPlayed >= 3) {reduction = 1;}
                break;
            case BAD_CAPTURES:
                if (canLateMoveReduce && depth >= 3 && movesPlayed >= 3)
                {
                    score = -alphaBetaQuiescence(b, ply+1, -beta, -alpha);
                    if (score <= alpha) {reduction = 1;}
                }
                break;
            case QUIET_MOVES:
                if (canLateMoveReduce && movesPlayed > 0) {reduction = int(0.5 * std::log((double)depth) * std::log((double)(movesPlayed+1)));}
                break;
        }

        if (depth >= 2 && movesPlayed > 0)
        {
            if (reduction > 0) {score = -alphaBeta(b, -beta, -alpha, depth-1-reduction, ply+1, true);}
            else {score = alpha + 1;}

            if (score > alpha)
            {
                score = -alphaBeta(b, -alpha-1, -alpha, depth-1, ply+1, true);
                if (score > alpha && score < beta) {score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);}
            }
        }
        else {score = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);}
        b.unmakeMove();
        ++movesPlayed;
        if (movePicker.stage == QUIET_MOVES) {++quietsPlayed;}

        //update scores.
        if (score > bestScore)
        {
            if (score >= beta)
            {
                bool isQuiet = movePicker.stage == QUIET_MOVES || movePicker.singleQuiets.contains(move);
                if (isQuiet)
                {
                    //update killers.
                    b.killer.update(move, ply);

                    //update history.
                    if (depth >= 5)
                    {
                        if (movePicker.stage == QUIET_MOVES) {b.history.update(movePicker.singleQuiets, b.moveCache[ply], movePicker.moveIndex - 1, move, depth);}
                        else {b.history.update(movePicker.singleQuiets, move, depth);}
                    }
                }

                //update tt.
                if (!isSearchAborted) {ttSave(bHash, ply, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //stalemate or checkmate.
    if (movesPlayed == 0) {return inCheck ? -MATE_SCORE + ply : 0;}

    if (!isSearchAborted) {ttSave(bHash, ply, depth, bestMove, bestScore, isExact, false);}
    return bestScore;
}

int alphaBetaRoot(Board &b, int depth, bool gensfen = false)
{
    //track start-time of search.
    startTime = std::chrono::high_resolution_clock::now();

    //update TT 'age' and reset nodes.
    rootCounter++;
    totalNodes = 0;

    //reset gameover state.
    isGameOver = false;

    //generate moves.
    bool inCheck = b.generatePseudoMoves();

    //checkmate or stalemate.
    if (b.moveBuffer.size() == 0) {storedBestMove = 0; isGameOver = true; return inCheck ? -MATE_SCORE : 0;}

    //if only one move, return immediately.
    if (b.moveBuffer.size() == 1 && !gensfen) {storedBestMove = b.moveBuffer[0]; return 0;}

    //draw by insufficient material.
    if (b.phase <= 1 && !(b.pieces[_nPawns] | b.pieces[_nPawns+1])) {storedBestMove = b.moveBuffer[0]; isGameOver = true; return 0;}

    //create move cache.
    std::vector<std::pair<U32,int> > moveCache;
    for (const auto move: b.moveBuffer)
    {
        moveCache.push_back(std::pair<U32,int>(move, 0));
    }

    //age history at root.
    b.history.age(8);

    //reset best score and best move.
    storedBestScore = -MATE_SCORE; storedBestMove = 0;
    int pvIndex = 0;

    //iterative deepening.
    for (int itDepth = 1; itDepth <= std::min(depth, MAXDEPTH); itDepth++)
    {
        auto iterationStartTime = std::chrono::high_resolution_clock::now();

        U32 startNodes = totalNodes;
        totalNodes++;

        int score;
        int alpha = -MATE_SCORE; int beta = MATE_SCORE;

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
            if (i == 0)
            {
                //aspiration window for pv move.
                int alphaInd = itDepth >= 5 ? 0 : aspirationDelta.size() - 1;
                int betaInd = itDepth >= 5 ? 0 : aspirationDelta.size() - 1;
                while (true)
                {
                    alpha = std::max(storedBestScore - aspirationDelta[alphaInd], -MATE_SCORE);
                    beta = std::min(storedBestScore + aspirationDelta[betaInd], MATE_SCORE);
                    score = -alphaBeta(b, -beta, -alpha, itDepth-1, 1, true);
                    if (score <= alpha) {++alphaInd;}
                    else if (score >= beta) {++betaInd;}
                    else {break;}
                }
            }
            else
            {
                int betaInd = itDepth >= 2 ? 0 : betaDelta.size() - 1;
                while (true)
                {
                    beta = std::min(alpha + betaDelta[betaInd], MATE_SCORE);
                    score = -alphaBeta(b, -beta, -alpha, itDepth-1, 1, true);
                    if (score >= beta) {++betaInd;}
                    else {break;}
                }
            }
            b.unmakeMove();
            if (score > alpha && !isSearchAborted)
            {
                alpha = score;
                pvIndex = i;
                storedBestMove = moveCache[i].first;
                storedBestScore = score;
            }
            moveCache[i].second = totalNodes - startMoveNodes;
        }

        //check if time is up.
        if (isSearchAborted) {break;}

        auto iterationFinishTime = std::chrono::high_resolution_clock::now();
        
        double iterationTime = std::chrono::duration<double, std::milli>(iterationFinishTime - iterationStartTime).count();
        double realTimeLeft = std::max(timeLeft - std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-startTime).count(), 0.);

        lastIterTime = (U32)(iterationTime);
        lastIterNodes = totalNodes - startNodes;
        lastIterNps = (U32)((double)(lastIterNodes) / (iterationTime / 1000.));

        //display info.
        if (!gensfen)
        {
            std::cout << "info" <<
                " depth " << itDepth <<
                " score " << (abs(storedBestScore) > MATE_BOUND ? "mate " : "cp ") << formatScore(storedBestScore) <<
                " time " << lastIterTime <<
                " nodes " << lastIterNodes <<
                " nps " << lastIterNps <<
                " pv";
            collectPVRoot(b, storedBestMove, itDepth);
            for (const auto pvMove: pvMoves)
            {
                std::cout << " " << moveToString(pvMove);
            } std::cout << std::endl;
        }

        //break if checkmate is reached.
        if (storedBestScore > MATE_BOUND) {break;}

        //early exit if insufficient time for next iteration.
        //assume a branching factor of 2.
        if (iterationTime * 2. > realTimeLeft) {break;}
    }
    return storedBestScore;
}

#endif // SEARCH_H_INCLUDED
