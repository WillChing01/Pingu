#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include "format.h"
#include "thread.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>

class Search {
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

    U32 go(int depth, double searchTime, U64 nodes, bool analysisMode, bool verbose) {
        ++rootCounter;
        globalNodeCount = 0;
        globalNodeLimit = nodes;

        // set helper threads to start searching.
        for (Thread* thread : threads) {
            thread->prepareSearch(depth, searchTime, analysisMode);
            std::thread(&Thread::rootSearch, thread, false).detach();
        }

        // search in main thread.
        mainThread.prepareSearch(depth, searchTime, analysisMode);
        mainThread.rootSearch(verbose);

        // wait for helper threads to finish.
        terminateThreads();

        return mainThread.bestMove;
    }
};

#endif // SEARCH_H_INCLUDED
