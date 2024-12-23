#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <unordered_set>
#include <vector>

#include "constants.h"

class History
{
public:
    // history table, scores[pieceType][to_square]
    int scores[12][64] = {};
    static const size_t historySize = 12 * 64;

    short extendedScores[12][64][6][64] = {};
    static const size_t extendedHistorySize = 12 * 64 * 6 * 64;

    History() {}

    void clear()
    {
        for (size_t i = 0; i < historySize; ++i)
        {
            (&scores[0][0])[i] = 0;
        }
        for (size_t i = 0; i < extendedHistorySize; ++i)
        {
            (&extendedScores[0][0][0][0])[i] = 0;
        }
    }

    void age(const int factor = 16)
    {
        for (size_t i = 0; i < historySize; ++i)
        {
            (&scores[0][0])[i] /= factor;
        }
    }

    void increment(U32 move, int bonus)
    {
        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

        int delta = 32 * bonus - ((int)scores[pieceType][finishSquare] * std::abs(bonus)) / 512;
        scores[pieceType][finishSquare] += delta;
    }

    void increment(U32 prevMove, U32 move, int bonus)
    {
        U32 prevPieceType = (prevMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        U32 prevFinishSquare = (prevMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

        int delta;

        delta = 32 * bonus - ((int)scores[pieceType][finishSquare] * std::abs(bonus)) / 512;
        scores[pieceType][finishSquare] += delta;

        delta = 32 * bonus - ((int)extendedScores[prevPieceType][prevFinishSquare][pieceType >> 1][finishSquare] * std::abs(bonus)) / 512;
        extendedScores[prevPieceType][prevFinishSquare][pieceType >> 1][finishSquare] += delta;
    }

    void update(const std::unordered_set<U32> &singles, U32 cutMove, int depth)
    {
        int bonus = std::min(depth * depth, 400);

        for (const auto &move : singles)
        {
            increment(move, move == cutMove ? bonus : -bonus);
        }
    }

    void update(U32 prevMove, const std::unordered_set<U32> &singles, U32 cutMove, int depth)
    {
        int bonus = std::min(depth * depth, 400);

        for (const auto &move : singles)
        {
            increment(prevMove, move, move == cutMove ? bonus : -bonus);
        }
    }

    void update(const std::unordered_set<U32> &singles, const std::vector<std::pair<U32, int>> &quiets, U32 index, U32 cutMove, int depth)
    {
        int bonus = std::min(depth * depth, 400);

        for (const auto &move : singles)
        {
            increment(move, -bonus);
        }

        for (size_t i = 0; i < index; ++i)
        {
            if (singles.contains(quiets[i].first))
            {
                continue;
            }
            increment(quiets[i].first, -bonus);
        }

        increment(cutMove, bonus);
    }

    void update(U32 prevMove, const std::unordered_set<U32> &singles, const std::vector<std::pair<U32, int>> &quiets, U32 index, U32 cutMove, int depth)
    {
        int bonus = std::min(depth * depth, 400);

        for (const auto &move : singles)
        {
            increment(prevMove, move, -bonus);
        }

        for (size_t i = 0; i < index; ++i)
        {
            if (singles.contains(quiets[i].first))
            {
                continue;
            }
            increment(prevMove, quiets[i].first, -bonus);
        }

        increment(prevMove, cutMove, bonus);
    }
};

#endif // HISTORY_H_INCLUDED
