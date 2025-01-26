#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <chrono>

#include "board.h"
#include "movepicker.h"

const int maximumPruningDepth = 8;

const int inverseFutilityMargin = 120;
const int inverseFutilityDepthLimit = 8;

const int nullMoveR = 2;
const int nullMoveDepthLimit = 3;

const int futilityDepthLimit = 2;
const std::array<int, futilityDepthLimit> futilityMargins = {150, 400};

const int lateMovePruningDepthLimit = 4;
const std::array<int, lateMovePruningDepthLimit> lateMovePruningMargins = {6, 10, 14, 18};

const std::array<int, 3> aspirationDelta = {50, 200, 2 * MATE_SCORE};
const std::array<int, 4> betaDelta = {1, 50, 200, 2 * MATE_SCORE};

std::array<std::array<int, 256>, MAXDEPTH> lmrTable = {};

void populateLmrTable()
{
    for (int depth=1;depth<=MAXDEPTH;++depth)
    {
        for (int c=1;c<=256;++c)
        {
            lmrTable[depth-1][c-1] = int(0.5 * std::log((double)depth) * std::log((double)c));
        }
    }
}

inline int formatScore(int score)
{
    return
        score > MATE_BOUND ? (MATE_SCORE - score + 1) / 2 :
        score < -MATE_BOUND ? (-MATE_SCORE - score) / 2 :
        score;
}

inline void collectPV(std::vector<U32> &pvMoves, Board &b, int depth)
{
    if (depth == 0) {return;}

    U64 bHash = b.zHashPieces ^ b.zHashState;
    U64 hashInfo = ttProbe(bHash);
    if (!hashInfo) {return;}

    U32 hashMove = getHashMove(hashInfo);
    bool isInCheck = util::isInCheck(b.side, b.pieces, b.occupied);
    bool isValid = validate::isValidMove(hashMove, isInCheck, b.side, b.current, b.pieces, b.occupied);
    if (!isValid) {return;}

    pvMoves.push_back(hashMove);
    b.makeMove(hashMove);
    collectPV(pvMoves, b, depth-1);
    b.unmakeMove();
}

std::atomic<U64> globalNodeCount = 0;
std::atomic<U64> globalNodeLimit = ULLONG_MAX;

class Thread
{
    public:
        Board b;
        U32 nodeCount = 0;
        std::atomic<bool> isSearchAborted{true};
        std::atomic<bool> isSearchFinished{true};

        U32 bestMove = 0;
        int bestScore = 0;

        double searchTime = 0.; // milliseconds.
        std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<U32, int> > rootMoves;
        int bestMoveIndex = 0;

        int maxDepth = 0;
        bool analysisMode = false;

        std::vector<U32> pvMoves;

        Thread() {}

        bool isTimeUp()
        {
            if (std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-startTime).count() > searchTime)
            {
                return true;
            }
            return false;
        }

        bool isDrawByMaterial()
        {
            return b.phase <= 1 && !(b.pieces[_nPawns] | b.pieces[_nPawns+1]);
        }

        bool isDrawByRepetition()
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

        bool isDrawByFifty()
        {
            //check if draw by fifty move rule.
            int lastIndex = b.moveHistory.size() - 1;
            int diff = b.irrevMoveInd.size() > 0 ? lastIndex - b.irrevMoveInd.back() : lastIndex;
            return diff >= 100;
        }

        int qSearch(int ply, int alpha, int beta)
        {
            //check time.
            if (!(nodeCount & 2047u) && (isTimeUp() || globalNodeCount > globalNodeLimit))
            {
                isSearchAborted = true;
            }
            if (isSearchAborted) {return 0;}
            ++nodeCount; ++globalNodeCount;

            //draw by insufficient material.
            if (isDrawByMaterial()) {return 0;}

            bool inCheck = util::isInCheck(b.side, b.pieces, b.occupied);
            int nodeBestScore = -MATE_SCORE + ply;

            //stand pat.
            if (!inCheck)
            {
                nodeBestScore = b.evaluateBoard();
                if (nodeBestScore > alpha)
                {
                    if (nodeBestScore >= beta) {return nodeBestScore;}
                    alpha = nodeBestScore;
                }
            }

            U32 numChecks = inCheck ? util::isInCheckDetailed(b.side, b.pieces, b.occupied) : 0;
            QMovePicker movePicker(&b, ply, numChecks);

            //loop through moves and search them.
            while (U32 move = movePicker.getNext())
            {
                b.makeMove(move);
                int score = -qSearch(ply+1, -beta, -alpha);
                b.unmakeMove();

                if (score > nodeBestScore)
                {
                    if (score > alpha)
                    {
                        if (score >= beta) {return score;}
                        alpha = score;
                    }
                    nodeBestScore = score;
                }
            }

            return nodeBestScore;
        }

        int search(int alpha, int beta, int depth, int ply, bool nullMoveAllowed)
        {
            //check time.
            if (!(nodeCount & 2047u) && (isTimeUp() || globalNodeCount > globalNodeLimit))
            {
                isSearchAborted = true;
            }
            if (isSearchAborted) {return 0;}
            ++nodeCount; ++globalNodeCount;

            //check for draw.
            if (isDrawByRepetition() || isDrawByMaterial() || isDrawByFifty()) {return 0;}

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
            if (depth <= 0) {--nodeCount; --globalNodeCount; return qSearch(ply, alpha, beta);}

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
                int nullScore = -search(-beta, -beta+1, depth-1-nullMoveR-depth/6, ply+1, false);
                b.unmakeNullMove();

                //fail hard only for null move pruning.
                if (nullScore >= beta) {return beta;}
            }

            //internal iterative reduction on hash miss.
            if (!hashInfo && depth > 3) {--depth;}

            //setup scoring variables.
            int score = 0; bool isExact = false;
            int nodeBestScore = -MATE_SCORE; U32 nodeBestMove = 0;
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

                //SEE pruning.
                if (movePicker.stage == BAD_CAPTURES && movesPlayed > 0 && b.badCaptures[ply][movePicker.moveIndex - 1].second + 100 * depth < 0)
                {
                    movePicker.moveIndex = b.badCaptures[ply].size();
                    continue;
                }

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
                            score = -qSearch(ply+1, -beta, -alpha);
                            if (score <= alpha) {reduction = 1;}
                        }
                        break;
                    case QUIET_MOVES:
                        if (canLateMoveReduce && movesPlayed > 0) {reduction = lmrTable[depth-1][movesPlayed];}
                        break;
                }

                if (depth >= 2 && movesPlayed > 0)
                {
                    if (reduction > 0) {score = -search(-beta, -alpha, depth-1-reduction, ply+1, true);}
                    else {score = alpha + 1;}

                    if (score > alpha)
                    {
                        score = -search(-alpha-1, -alpha, depth-1, ply+1, true);
                        if (score > alpha && score < beta) {score = -search(-beta, -alpha, depth-1, ply+1, true);}
                    }
                }
                else {score = -search(-beta, -alpha, depth-1, ply+1, true);}
                b.unmakeMove();
                ++movesPlayed;
                if (movePicker.stage == QUIET_MOVES) {++quietsPlayed;}

                //update scores.
                if (score > nodeBestScore)
                {
                    if (score >= beta)
                    {
                        if (!isSearchAborted)
                        {
                            bool isQuiet = movePicker.stage == QUIET_MOVES || movePicker.singleQuiets.contains(move);
                            if (isQuiet)
                            {
                                //update killers.
                                b.killer.update(move, ply);

                                //update history.
                                if (depth >= 5)
                                {
                                    bool hasPrevMove = b.moveHistory.size() && b.moveHistory.back() != 0;
                                    if (movePicker.stage == QUIET_MOVES)
                                    {
                                        if (hasPrevMove)
                                        {
                                            b.history.update(b.moveHistory.back(), movePicker.singleQuiets, b.moveCache[ply], movePicker.moveIndex - 1, move, depth);
                                        }
                                        else
                                        {
                                            b.history.update(movePicker.singleQuiets, b.moveCache[ply], movePicker.moveIndex - 1, move, depth);
                                        }
                                    }
                                    else
                                    {
                                        if (hasPrevMove)
                                        {
                                            b.history.update(b.moveHistory.back(), movePicker.singleQuiets, move, depth);
                                        }
                                        else
                                        {
                                            b.history.update(movePicker.singleQuiets, move, depth);
                                        }
                                    }
                                }
                            }

                            //update tt.
                            ttSave(bHash, ply, depth, move, score, false, true);
                        }
                        return score;
                    }
                    if (score > alpha) {alpha = score; isExact = true;}
                    nodeBestScore = score;
                    nodeBestMove = move;
                }
            }

            //stalemate or checkmate.
            if (movesPlayed == 0) {return inCheck ? -MATE_SCORE + ply : 0;}

            if (!isSearchAborted) {ttSave(bHash, ply, depth, nodeBestMove, nodeBestScore, isExact, false);}
            return nodeBestScore;
        }

        void aspirationSearch(int depth)
        {
            int score = -MATE_SCORE;
            U32 startNodes;

            //aspiration search for pv move.
            startNodes = nodeCount;
            b.makeMove(rootMoves[0].first);
            int alphaInd = depth >= 5 ? 0 : aspirationDelta.size() - 1;
            int betaInd = depth >= 5 ? 0 : aspirationDelta.size() - 1;
            while (true)
            {
                int alpha = std::max(bestScore - aspirationDelta[alphaInd], -MATE_SCORE);
                int beta = std::min(bestScore + aspirationDelta[betaInd], MATE_SCORE);
                score = -search(-beta, -alpha, depth-1, 1, true);
                if (score <= alpha) {++alphaInd; continue;}
                if (score >= beta) {++betaInd; continue;}
                break;
            }
            b.unmakeMove();
            if (!isSearchAborted)
            {
                bestScore = score;
                bestMove = rootMoves[0].first;
                bestMoveIndex = 0;
            }
            rootMoves[0].second = nodeCount - startNodes;

            //pvs with aspiration for subsequent moves.
            for (int i=1;i<(int)(rootMoves.size());++i)
            {
                startNodes = nodeCount;
                b.makeMove(rootMoves[i].first);
                int betaInd = depth >= 2 ? 0 : betaDelta.size() - 1;
                while (true)
                {
                    int beta = std::min(bestScore + betaDelta[betaInd], MATE_SCORE);
                    score = -search(-beta, -bestScore, depth-1, 1, true);
                    if (score >= beta) {++betaInd; continue;}
                    break;
                }
                b.unmakeMove();
                if (score > bestScore && !isSearchAborted)
                {
                    bestScore = score;
                    bestMove = rootMoves[i].first;
                    bestMoveIndex = i;
                }
                rootMoves[i].second = nodeCount - startNodes;
            }
        }

        void rootSearch(bool verbose)
        {
            //root search with iterative deepening.
            bool inCheck = b.generatePseudoMoves();

            //checkmate or stalemate.
            if (b.moveBuffer.size() == 0) {bestScore = inCheck ? -MATE_SCORE : 0; isSearchFinished = true; return;}
            //if only one move return immediately.
            if (b.moveBuffer.size() == 1 && !analysisMode) {bestMove = b.moveBuffer[0]; isSearchFinished = true; return;}

            //set root moves.
            for (const U32 move: b.moveBuffer)
            {
                rootMoves.push_back(std::pair<U32, int>(move, 0));
            }

            for (int depth = 1; depth <= std::min(maxDepth, MAXDEPTH); ++depth)
            {
                //start of iteration book-keeping.
                auto iterationStartTime = std::chrono::high_resolution_clock::now();
                ++nodeCount; ++globalNodeCount;

                //order root moves by nodes searched.
                if (depth > 1)
                {
                    rootMoves[bestMoveIndex].second = INT_MAX;
                    std::sort(rootMoves.begin(), rootMoves.end(), [](const auto &a, const auto &b) {return a.second > b.second;});
                }

                //aspiration search.
                aspirationSearch(depth);
                if (isSearchAborted) {break;}

                //end of iteration book-keeping.
                auto iterationFinishTime = std::chrono::high_resolution_clock::now();
                double iterationTime = std::chrono::duration<double, std::milli>(iterationFinishTime - iterationStartTime).count();
                double totalTimeSpent = std::chrono::duration<double, std::milli>(iterationFinishTime - startTime).count();
                double timeLeft = std::max(searchTime - totalTimeSpent, 0.);

                if (verbose) {outputInfo(depth, totalTimeSpent);}

                //early exit if mate detected.
                if (bestScore > MATE_BOUND && !analysisMode) {break;}
                //early exit if insufficient time for next iteration.
                if (iterationTime * 2. > timeLeft && !analysisMode) {break;}
            }

            isSearchFinished = true;
        }

        void prepareSearch(int _maxDepth, double _searchTime, bool _analysisMode)
        {
            maxDepth = _maxDepth;
            analysisMode = _analysisMode;

            nodeCount = 0;
            isSearchAborted = false;
            isSearchFinished = false;

            searchTime = _searchTime;
            startTime = std::chrono::high_resolution_clock::now();

            bestMoveIndex = 0;
            bestMove = 0;
            bestScore = -MATE_SCORE;

            rootMoves.clear();

            //age history at root.
            b.history.age(8);
        }

        void prepareNewGame()
        {
            b.killer.clear();
            b.history.clear();
        }

        void outputInfo(int depth, double totalTimeSpent)
        {
            U64 nps = (U64)((double)(globalNodeCount) * 1000. / totalTimeSpent);
            std::cout << "info"
                << " depth " << depth
                << " score " << (abs(bestScore) > MATE_BOUND ? "mate " : "cp ") << formatScore(bestScore)
                << " time " << (U64)totalTimeSpent
                << " nodes " << globalNodeCount
                << " nps " << nps
                << " pv";

            pvMoves = {bestMove};
            b.makeMove(bestMove);
            collectPV(pvMoves, b, depth-1);
            b.unmakeMove();

            for (const U32 move: pvMoves)
            {
                std::cout << " " << moveToString(move);
            } std::cout << std::endl;
        }
};

#endif // THREAD_H_INCLUDED
