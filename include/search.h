#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <iostream>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "format.h"
#include "thread.h"

inline int formatScore(int score)
{
    return
        score > MATE_BOUND ? (MATE_SCORE - score + 1) / 2 :
        score < -MATE_BOUND ? (-MATE_SCORE - score) / 2 :
        score;
}

void collectPV(std::vector<U32> &pvMoves, Board &b, int depth)
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

class Search
{
    public:
        std::vector<Thread*> threads = {new Thread(&_cv)};
        Board pvBoard;

        std::mutex _m;
        std::condition_variable _cv;

        Search() {}

        ~Search()
        {
            while (threads.size())
            {
                delete threads.back();
                threads.pop_back();
            }
        }

        void outputInfo(int depth)
        {
            //collect info from all threads.
            U32 totalNodes = 0;
            double averageTime = 0.;

            //output pv and score of first thread (arbitrary).
            for (Thread* thread: threads)
            {
                totalNodes += thread->searchResults[depth-1].nodes;
                averageTime += thread->searchResults[depth-1].time / threads.size();
            }
            U32 nps = (U32)((double)totalNodes * 1000. / averageTime);

            int bestScore = threads[0]->searchResults[depth-1].bestScore;
            U32 bestMove = threads[0]->searchResults[depth-1].bestMove;

            std::cout << "info"
                << " depth " << depth
                << " score " << (abs(bestScore) > MATE_BOUND ? "mate " : "cp ") << formatScore(bestScore)
                << " time " << (U32)averageTime
                << " nodes " << totalNodes
                << " nps " << nps
                << " pv";

            std::vector<U32> pvMoves = {threads[0]->bestMove};
            pvBoard.makeMove(bestMove);
            collectPV(pvMoves, pvBoard, depth-1);
            pvBoard.unmakeMove();

            for (const U32 move: pvMoves)
            {
                std::cout << " " << moveToString(move);
            } std::cout << std::endl;
        }

        void setThreads(int numThreads)
        {
            if (!areThreadsTerminated()) {return;}
            if (numThreads < 1) {return;}

            while (numThreads < (int)threads.size())
            {
                delete threads.back();
                threads.pop_back();
            }

            while (numThreads > (int)threads.size())
            {
                threads.push_back(new Thread(&_cv));
                threads.back()->b.copyBoard(threads[0]->b);
            }
        }

        bool areThreadsTerminated()
        {
            for (Thread* thread: threads)
            {
                if (!thread->isSearchFinished) {return false;}
            }
            return true;
        }

        void terminateThreads()
        {
            for (Thread* thread: threads)
            {
                thread->isSearchAborted = true;
                while (!thread->isSearchFinished) {}
            }
        }

        void prepareForNewGame()
        {
            for (Thread* thread: threads)
            {
                thread->prepareNewGame();
            }
        }

        void setPositionFen(const std::string &fen)
        {
            pvBoard.setPositionFen(fen);
            for (Thread* thread: threads)
            {
                thread->b.setPositionFen(fen);
            }
        }

        void makeMove(const U32 move)
        {
            pvBoard.makeMove(move);
            for (Thread* thread: threads)
            {
                thread->b.makeMove(move);
            }
        }

        U32 go(int depth, double searchTime, bool analysisMode, bool verbose)
        {
            ++rootCounter;

            U32 nextDepth = 1;
            std::string input;
            bool isSearching = true;

            //set the threads to start searching.
            std::unique_lock<std::mutex> lock(_m);
            for (Thread* thread: threads)
            {
                thread->prepareSearch(depth, searchTime, analysisMode);
                std::thread t(&Thread::rootSearch, thread);
                t.detach();
            }

            while (isSearching)
            {
                _cv.wait(lock);
                //check if threads have finished.
                if (areThreadsTerminated()) {isSearching = false;}

                //check if threads have completed next depth.
                while (verbose)
                {
                    bool isNextDepthFinished = true;
                    for (Thread* thread: threads)
                    {
                        if (thread->depthCounter < nextDepth) {isNextDepthFinished = false; break;}
                    }

                    if (isNextDepthFinished)
                    {
                        outputInfo(nextDepth);
                        ++nextDepth;
                    }
                    else {break;}
                }
            }

            return threads[0]->bestMove;
        }
};

#endif // SEARCH_H_INCLUDED
