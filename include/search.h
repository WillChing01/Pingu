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
#include "node.h"

const int maximumPruningDepth = 8;

const int inverseFutilityMargin = 120;
const int inverseFutilityDepthLimit = 8;

const int nullMoveR = 2;
const int nullMoveDepthLimit = 3;

static const int futilityDepthLimit = 2;
const std::array<int, futilityDepthLimit> futilityMargins = {150, 400};

static const int lateMovePruningDepthLimit = 4;
const std::array<int, lateMovePruningDepthLimit> lateMovePruningMargins = {8, 14, 20, 26};

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

static const int MAX_PLY = 64;
static const int MAX_Q_PLY = 64;

Node searchStack[MAX_PLY] = {};
QNode qStack[MAX_Q_PLY] = {};

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
    if (ttProbe(bHash, 0) == true && depth > 0)
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

inline int alphaBetaQuiescence(Board &b, int ply, int alpha, int beta)
{
    //check time.
    totalNodes++;
    if ((totalNodes & 2047) == 0) {if (!checkTime()) {return 0;}}
    if (isSearchAborted) {return 0;}

    //draw by insufficient material.
    if (b.phase <= 1 && !(b.pieces[b._nPawns] | b.pieces[b._nPawns+1])) {return 0;}

    bool side = b.moveHistory.size() & 1;
    bool inCheck = b.isInCheck(side);

    int bestScore = -MATE_SCORE + ply;

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

    std::vector<std::pair<U32,int> > moveCache = inCheck ? b.orderQMovesInCheck() : b.orderQMoves();

    for (const auto &[move,moveScore]: moveCache)
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

    //check for draw by repetition.
    if (isDraw(b)) {return 0;}

    //draw by insufficient material.
    if (b.phase <= 1 && !(b.pieces[b._nPawns] | b.pieces[b._nPawns+1])) {return 0;}

    //initialize pointer to node.
    Node* node = &searchStack[ply];

    //probe hash table.
    node->zHash = b.zHashPieces ^ b.zHashState;
    node->hashHit = ttProbe(node->zHash, ply);
    U32 hashMove = node->hashHit ? tableEntry.bestMove : 0;

    //check for early TT cutoff.
    if (node->hashHit && tableEntry.depth >= depth)
    {
        //PV node.
        if (tableEntry.isExact) {return tableEntry.evaluation;}
        //Cut node.
        else if (tableEntry.isBeta) {if (tableEntry.evaluation >= beta) {return tableEntry.evaluation;}}
        //All node.
        else {if (tableEntry.evaluation <= alpha) {return tableEntry.evaluation;}}
    }

    //qSearch at horizon.
    if (depth <= 0) {totalNodes--; return alphaBetaQuiescence(b, ply, alpha, beta);}

    //main search.
    node->side = b.moveHistory.size() & 1;
    node->inCheck = b.isInCheck(node->side);

    //get static evaluation.
    bool canPrune = (alpha == beta - 1) && !node->inCheck && abs(alpha) < MATE_BOUND;
    if (canPrune && depth <= maximumPruningDepth) {node->staticEval = b.regularEval();}

    //inverse futility pruning.
    if (canPrune && depth <= inverseFutilityDepthLimit)
    {
        int margin = inverseFutilityMargin * depth;
        if (node->staticEval - margin >= beta) {return beta;}
    }

    if (node->hashHit && tableEntry.depth >= depth-nullMoveR-depth/6 && !tableEntry.isBeta && tableEntry.evaluation < beta) {nullMoveAllowed = false;}

    //null move pruning.
    if (nullMoveAllowed && !node->inCheck && depth >= nullMoveDepthLimit &&
        (b.occupied[node->side] ^ b.pieces[b._nKing+node->side] ^ b.pieces[b._nPawns+node->side]))
    {
        b.makeNullMove();
        int nullScore = -alphaBeta(b, -beta, -beta+1, depth-1-nullMoveR-depth/6, ply+1, false);
        b.unmakeNullMove();

        //fail hard only for null move pruning.
        if (nullScore >= beta) {return beta;}
    }

    //setup scoring variables.
    int score; bool isExact = false;
    int bestScore = -MATE_SCORE; U32 bestMove = 0;
    int numMoves = 0;

    std::unordered_set<U32> singleQuiets;

    //try hash move.
    if (node->hashHit && b.isValidMove(hashMove, node->inCheck))
    {
        b.makeMove(hashMove);
        bestScore = -alphaBeta(b, -beta, -alpha, depth-1, ply+1, true);
        b.unmakeMove();
        numMoves++;

        if (bestScore >= beta)
        {
            //beta cutoff.
            bool isQuiet = (hashMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET == 15 &&
            (hashMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET == (hashMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            if (isQuiet)
            {
                b.updateKiller(hashMove, ply);
                if (depth >= 5) {b.updateHistory(singleQuiets, hashMove, depth);}
            }

            //update transposition table.
            if (!isSearchAborted) {ttSave(node->zHash, ply, depth, hashMove, bestScore, false, true);}
            return bestScore;
        }
        if (bestScore > alpha) {alpha = bestScore; isExact = true;}
        bestMove = hashMove;
        bool isQuiet = (hashMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET == 15 &&
        (hashMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET == (hashMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        if (isQuiet) {singleQuiets.insert(hashMove);}
    }

    //internal iterative reduction on hash miss.
    if (!node->hashHit && depth > 3) {depth--;}

    //get number of checks for move-gen.
    U32 numChecks = 0;
    if (node->inCheck) {numChecks = b.isInCheckDetailed(node->side);}

    //generate tactical moves and play them.
    b.moveBuffer.clear();
    b.generateCaptures(node->side, numChecks);
    node->moveCache = b.orderCaptures();

    //good captures and promotions.
    U32 move;
    int ind = node->moveCache.size();
    for (int i=0;i<(int)(node->moveCache.size());i++)
    {
        move = node->moveCache[i].first;
        //check that capture is not hash move.
        if (node->hashHit && (move == hashMove)) {continue;}
        //exit when we get to bad captures.
        if (node->moveCache[i].second < 0) {ind = i; break;}

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
                if (!isSearchAborted) {ttSave(node->zHash, ply, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //try killers.
    for (int i=0;i<2;i++)
    {
        move = b.killerMoves[ply][i];
        //check if move was played before.
        if (singleQuiets.contains(move)) {continue;}
        //check if killer is valid.
        if (!b.isValidMove(move, node->inCheck)) {continue;}

        b.makeMove(move);
        if (depth >= 2 && numMoves > 0)
        {
            //late move reductions (non pv nodes).
            if (depth >= 3 && (alpha == (beta - 1)) && numMoves >= 3 && !node->inCheck)
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
                b.updateKiller(move, ply);
                if (depth >= 5) {b.updateHistory(singleQuiets, move, depth);}

                //update transposition table.
                if (!isSearchAborted) {ttSave(node->zHash, ply, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
        singleQuiets.insert(move);
    }

    //bad captures.
    for (int i=ind;i<(int)(node->moveCache.size());i++)
    {
        move = node->moveCache[i].first;
        //check that capture is not hash move.
        if (node->hashHit && (move == hashMove)) {continue;}

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
                if (!isSearchAborted) {ttSave(node->zHash, ply, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //generate quiets and try them.
    b.moveBuffer.clear();
    b.generateQuiets(node->side, numChecks);
    node->moveCache = b.orderQuiets();

    int numQuiets = 0;

    bool canLateMovePrune = canPrune && depth <= lateMovePruningDepthLimit;

    bool canFutilityPrune = canPrune && depth <= futilityDepthLimit &&
                            node->staticEval + futilityMargins[depth-1] <= alpha;

    for (int i=0;i<(int)(node->moveCache.size());i++)
    {
        //late move pruning.
        if (canLateMovePrune && numQuiets > lateMovePruningMargins[depth-1]) {break;}

        move = node->moveCache[i].first;

        if (singleQuiets.contains(move)) {continue;}

        //futility pruning.
        if (canFutilityPrune && numMoves > 0 && !b.isCheckingMove(move)) {continue;}

        b.makeMove(move);
        if (depth >= 2 && numMoves > 0)
        {
            //late move reductions (non pv nodes).
            int LMR = int(0.5 * std::log((double)depth) * std::log((double)(numMoves+1)));
            if (LMR && (alpha == (beta - 1)) && !node->inCheck)
            {
                score = -alphaBeta(b, -beta, -alpha, depth-1-LMR, ply+1, true);
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
        numMoves++; numQuiets++;

        if (score > bestScore)
        {
            if (score >= beta)
            {
                //beta cutoff.
                b.updateKiller(move, ply);
                if (depth >= 5) {b.updateHistory(singleQuiets, node->moveCache, i, move, depth);}

                //update transposition table.
                if (!isSearchAborted) {ttSave(node->zHash, ply, depth, move, score, false, true);}
                return score;
            }
            if (score > alpha) {alpha = score; isExact = true;}
            bestScore = score;
            bestMove = move;
        }
    }

    //stalemate or checkmate.
    if (numMoves == 0) {return node->inCheck ? -MATE_SCORE + ply : 0;}

    //update transposition table.
    if (!isSearchAborted) {ttSave(node->zHash, ply, depth, bestMove, bestScore, isExact, false);}
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
    bool side = b.moveHistory.size() & 1;
    bool inCheck = b.generatePseudoMoves(side);

    //checkmate or stalemate.
    if (b.moveBuffer.size() == 0) {storedBestMove = 0; isGameOver = true; return inCheck ? -MATE_SCORE : 0;}

    //if only one move, return immediately.
    if (b.moveBuffer.size() == 1 && !gensfen) {storedBestMove = b.moveBuffer[0]; return 0;}

    //draw by insufficient material.
    if (b.phase <= 1 && !(b.pieces[b._nPawns] | b.pieces[b._nPawns+1])) {storedBestMove = b.moveBuffer[0]; isGameOver = true; return 0;}

    //create move cache.
    std::vector<std::pair<U32,int> > moveCache;
    for (const auto move: b.moveBuffer)
    {
        moveCache.push_back(std::pair<U32,int>(move, 0));
    }

    //age history at root.
    b.ageHistory(8);

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
