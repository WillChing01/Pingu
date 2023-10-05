#ifndef PERFT_H_INCLUDED
#define PERFT_H_INCLUDED

#include <chrono>
#include <algorithm>

#include "constants.h"
#include "format.h"
#include "board.h"

U64 perft(Board &b, int depth, bool verbose = true)
{
    if (depth == 0)
    {
        return 1;
    }
    else if (depth == 1)
    {
        b.generatePseudoMoves(b.moveHistory.size() & 1);
        if (verbose)
        {
            for (const auto &move: b.moveBuffer)
            {
                std::cout << toCoord((move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET)
                          << toCoord((move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET)
                          << " : " << 1
                          << std::endl;
            }
        }
        return b.moveBuffer.size();
    }
    else
    {
        U64 total = 0;
        b.generatePseudoMoves(b.moveHistory.size() & 1);
        std::vector<U32> moveCache = b.moveBuffer;

        for (const auto &move: moveCache)
        {
            b.makeMove(move);
            U64 nodes = perft(b,depth-1, false);
            total += nodes;
            b.unmakeMove();
            if (verbose)
            {
                std::cout << toCoord((move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET)
                          << toCoord((move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET)
                          << " : " << nodes
                          << std::endl;
            }
        }

        return total;
    }
}

bool testMoveValidation(Board &b, int depth, U32 (&cache)[10][128])
{
    if (depth == 0) {return true;}
    else
    {
        bool res = true;
        bool inCheck = b.generatePseudoMoves(b.moveHistory.size() & 1);
        std::vector<U32> moveCache = b.moveBuffer;

        //validate moves at the same ply.
        //no reductions, so depth = max_depth - ply
        for (const auto &move: cache[depth])
        {
            if (move == 0) {continue;}
            bool isValid = b.isValidMove(move, inCheck);
            //check if move in the list.
            bool isInList = std::find(moveCache.begin(), moveCache.end(), move) != moveCache.end();
            if (isValid != isInList)
            {
                //error!
                res = false;
                b.display();
                b.unpackMove(move);
                std::cout << move << std::endl;
                std::cout << toCoord(b.currentMove.startSquare)
                          << toCoord(b.currentMove.finishSquare)
                          << " should be " << (isInList ? "valid" : "invalid") << std::endl;
                std::cout << "pieceType: " << b.currentMove.pieceType << std::endl;
                std::cout << "finishPieceType: " << b.currentMove.finishPieceType << std::endl;
                std::cout << "capturedPieceType: " << b.currentMove.capturedPieceType << std::endl;
                std::cout << "enPassant: " << b.currentMove.enPassant << std::endl;
            }
        }

        for (const auto &move: moveCache)
        {
            //check if move is valid.
            bool isValid = b.isValidMove(move, inCheck);
            if (!isValid)
            {
                //error!
                res = false;
                b.display();
                std::cout << toCoord((move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET)
                          << toCoord((move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET)
                          << " should be valid" << std::endl;
            }

            //add move to cache.
            U64 zHash = b.zHashPieces & b.zHashState;
            cache[depth][zHash & 127] = move;

            //play the move.
            b.makeMove(move);
            res = res && testMoveValidation(b, depth-1, cache);
            b.unmakeMove();
        }
        return res;
    }
}

bool testIncrementalUpdate(Board &b, int depth, auto Board::* param, void (Board::* hardUpdate)())
{
    //verify incremental updates of 'param' for given position.
    auto oldParam = b.*param;
    (b.*hardUpdate)();
    if ((depth == 0) || (oldParam != b.*param)) {return (oldParam == b.*param);}

    //make moves recursively.
    b.generatePseudoMoves(b.moveHistory.size() & 1);
    std::vector<U32> moveCache = b.moveBuffer;
    for (const auto &move: moveCache)
    {
        b.makeMove(move);
        if (!testIncrementalUpdate(b, depth-1, param, hardUpdate)) {return false;}
        b.unmakeMove();
    }
    return true;
}

bool incrementalTest(Board &b, int depth)
{
    //test all incrementally updated parameters.

    bool res = true;

    //material.
    res &= testIncrementalUpdate(b, depth, &Board::materialStart, &Board::evalHardUpdate);
    res &= testIncrementalUpdate(b, depth, &Board::materialEnd, &Board::evalHardUpdate);

    //pst.
    res &= testIncrementalUpdate(b, depth, &Board::pstStart, &Board::evalHardUpdate);
    res &= testIncrementalUpdate(b, depth, &Board::pstEnd, &Board::evalHardUpdate);

    //phase.
    res &= testIncrementalUpdate(b, depth, &Board::phase, &Board::phaseHardUpdate);
    res &= testIncrementalUpdate(b, depth, &Board::shiftedPhase, &Board::phaseHardUpdate);

    //zobrist.
    res &= testIncrementalUpdate(b, depth, &Board::zHashPieces, &Board::zHashHardUpdate);
    res &= testIncrementalUpdate(b, depth, &Board::zHashState, &Board::zHashHardUpdate);
    res &= testIncrementalUpdate(b, depth, &Board::zHashPawns, &Board::zHashHardUpdate);

    return res;
}

#endif // PERFT_H_INCLUDED
