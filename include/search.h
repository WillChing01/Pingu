#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "format.h"
#include "thread.h"
#include "time-network.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>

struct searchParams {
    U32 time;
    U32 opponentTime;
    U32 inc;
    U32 opponentInc;
    U32 movesToGo;
    int depth = INT_MAX;
    U64 nodes = ULLONG_MAX;
    double moveTime = std::numeric_limits<double>::infinity();
};

class Search {
  private:
    TimeNetwork timeNetwork = TimeNetwork(&mainThread);

  public:
    Thread mainThread;
    std::vector<Thread*> threads = {};

    Search() {}

    ~Search() {
        while (threads.size()) {
            delete threads.back();
            threads.pop_back();
        }
    }

    void setThreads(int numThreads) {
        if (!areThreadsTerminated()) {
            return;
        }
        if (numThreads < 2) {
            return;
        }

        --numThreads; // main thread always there.
        while (numThreads < (int)threads.size()) {
            delete threads.back();
            threads.pop_back();
        }
        while (numThreads > (int)threads.size()) {
            threads.push_back(new Thread());
            threads.back()->b.copyBoard(mainThread.b);
        }
    }

    bool areThreadsTerminated() {
        for (Thread* thread : threads) {
            if (!thread->isSearchFinished) {
                return false;
            }
        }
        return true;
    }

    void terminateThreads() {
        for (Thread* thread : threads) {
            thread->isSearchAborted = true;
            while (!thread->isSearchFinished) {
            }
        }
    }

    void terminateSearch() {
        terminateThreads();
        mainThread.isSearchAborted = true;
        while (!mainThread.isSearchFinished) {
        }
    }

    void prepareForNewGame() {
        mainThread.prepareNewGame();
        for (Thread* thread : threads) {
            thread->prepareNewGame();
        }
    }

    void setPositionFen(const std::string& fen) {
        mainThread.b.setPositionFen(fen);
        for (Thread* thread : threads) {
            thread->b.setPositionFen(fen);
        }
    }

    void makeMove(const U32 move) {
        mainThread.b.makeMove(move);
        for (Thread* thread : threads) {
            thread->b.makeMove(move);
        }
    }

    U32 go(searchParams& params, bool analysisMode, bool verbose) {
        ++rootCounter;
        globalNodeCount = 0;
        globalNodeLimit = params.nodes;

        // TODO - update time management.
        if (params.time) {
            double t = params.time;
            double dt = params.inc;
            params.moveTime =
                std::min(params.movesToGo ? t / params.movesToGo + 0.5 * dt
                                          : t * timeNetwork.forward(params.time, params.inc, params.opponentTime),
                         0.8 * t);
        }

        // set helper threads to start searching.
        for (Thread* thread : threads) {
            thread->prepareSearch(params.depth, params.moveTime, analysisMode);
            std::thread(&Thread::rootSearch, thread, false).detach();
        }

        // search in main thread.
        mainThread.prepareSearch(params.depth, params.moveTime, analysisMode);
        mainThread.rootSearch(verbose);

        // wait for helper threads to finish.
        terminateThreads();

        return mainThread.bestMove;
    }
};

#endif // SEARCH_H_INCLUDED
