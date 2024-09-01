#ifndef BENCH_H_INCLUDED
#define BENCH_H_INCLUDED

#include <iostream>
#include <string>

#include "uci.h"

const std::array<std::string, 10> benchPositions = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "2kr4/Qnpq4/1p3pp1/2p1r2p/7P/R3p1P1/P3BPK1/1R6 b - - 0 30",
    "2r5/4k1p1/p2pp3/2n1q1b1/2P1P1P1/3Q3R/P3B1K1/4R3 w - - 3 36",
    "r4rk1/5p1p/p3p1p1/q1b1P3/Pp2Q1PP/3B4/P1P5/2KR3R w - - 0 22",
    "r4r2/4Npk1/4p2p/2Np1bpQ/p2Pn3/4P3/q4PPP/2R2RK1 b - - 3 22",
    "4k1r1/p1prqpPp/2Q1b3/PP2R1B1/2pb4/2p5/2K2PPP/RN6 b - - 3 20",
    "4r1k1/p2b1ppp/2pb4/3p2q1/7r/1P2PN1P/PB3PP1/2RQ1R1K b - - 5 21",
    "3rrbk1/5pp1/p2p2np/1pq2N2/4b3/PB4BP/1QP2PP1/3RR1K1 w - - 0 24"
};

const int benchDepth = 13;

void benchCommand()
{
    Search search;
    U32 nodes = 0;
    double time = 0;

    for (const std::string &fen: benchPositions)
    {
        clearTT();
        rootCounter = 0;
        search.setPositionFen(fen);
        search.go(benchDepth, INT_MAX, true, false);
    
        U32 iterNodes = search.threads[0]->searchResults.back().nodes;
        double iterTime = search.threads[0]->searchResults.back().time;

        U32 iterNps = (U32)((double)iterNodes * 1000. / iterTime);

        nodes += iterNodes;
        time += iterTime;

        std::cout << "info bench position " << fen << " nodes " << iterNodes << " nps " << iterNps << std::endl;
    }

    double nps = 1000. * (double)(nodes) / (time);
    std::cout << nodes << " nodes " << (U32)(nps) << " nps" << std::endl;
}

#endif // BENCH_H_INCLUDED
