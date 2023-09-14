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

bool testZobristHashing(Board &b, int depth)
{
    //verify the zobrist hashing for position.
    if (depth == 0)
    {
        //check if current position is equal to zHash.
        U64 incrementalHash = b.zHashState ^ b.zHashPieces;
        b.zHashHardUpdate();
        return (b.zHashState ^ b.zHashPieces) == incrementalHash;
    }

    //make moves recursively.
    bool res = true;
    b.generatePseudoMoves(b.moveHistory.size() & 1);
    std::vector<U32> moveCache = b.moveBuffer;

    //verify hash at current position.
    U64 incrementalHash = b.zHashState ^ b.zHashPieces;
    b.zHashHardUpdate();
    if ((b.zHashState ^ b.zHashPieces) != incrementalHash)
    {
        //error!
        b.display();
        for (const auto &history: b.moveHistory)
        {
            std::cout << moveToString(history) << " ";
        }
        std::cout << "\nIncorrect zobrist hash" << std::endl;

        return false;
    }

    for (const auto &move: moveCache)
    {
        b.makeMove(move);
        res = res && testZobristHashing(b, depth-1);
        b.unmakeMove();
    }
    return res;
}

#endif // PERFT_H_INCLUDED
