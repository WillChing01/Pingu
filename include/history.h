#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <unordered_set>
#include <vector>

#include "constants.h"

class History
{
public:
    static const size_t historySize = 12 * 64;
    short scores[12][64] = {};

    static const size_t butterflyHistorySize = 64 * 64;
    short butterflyScores[2][64][64] = {};

    static const size_t continuationHistorySize = 2 * 6 * 64 * 12 * 64;
    short continuationScores[2][6][64][12][64] = {};

    History() {}

    void clear()
    {
        for (size_t i = 0; i < historySize; ++i)
        {
            (&scores[0][0])[i] = 0;
        }
        for (size_t i = 0; i < butterflyHistorySize; ++i)
        {
            (&butterflyScores[0][0][0])[i] = 0;
        }
        for (size_t i = 0; i < continuationHistorySize; ++i)
        {
            (&continuationScores[0][0][0][0][0])[i] = 0;
        }
    }

    void age(const int factor = 16)
    {
        for (size_t i = 0; i < historySize; ++i)
        {
            (&scores[0][0])[i] /= factor;
        }
        for (size_t i = 0; i < butterflyHistorySize; ++i)
        {
            (&butterflyScores[0][0][0])[i] /= factor;
        }
    }

    void increment_(short * entry, int bonus)
    {
        int delta = 32 * bonus - ((int)(*entry) * std::abs(bonus)) / 512;
        *entry += delta;
    }

    void increment(U32 move, int bonus, short (*currentContinuation[2])[12][64])
    {
        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        U32 startSquare = (move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
        U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

        increment_(&scores[pieceType][finishSquare], bonus);
        increment_(&butterflyScores[pieceType % 2][startSquare][finishSquare], bonus);
        for (size_t i = 0; i < 2; ++i)
        {
            if (*currentContinuation[i])
            {
                increment_(&(*currentContinuation[i])[pieceType][finishSquare], bonus);
            }
        }
    }

    void update(int depth, U32 cutMove, int quietsPlayed, const std::unordered_set<U32> &singles, const std::vector<std::pair<U32, int> > &quiets, const std::vector<U32> &moveHistory)
    {
        int bonus = std::min(depth * depth, 400);

        short (*currentContinuation[2])[12][64] = {nullptr, nullptr};
        for (size_t i = 0; i < 2; ++i)
        {
            if (moveHistory.size() > i)
            {
                if (U32 move = moveHistory[moveHistory.size() - 1 - i])
                {
                    U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                    U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                    currentContinuation[i] = &continuationScores[i][pieceType / 2][finishSquare];
                }
            }
        }

        if (!quietsPlayed)
        {
            for (const auto &move : singles)
            {
                increment(move, move == cutMove ? bonus : -bonus, currentContinuation);
            }
            return;
        }
        else
        {
            for (const auto &move : singles)
            {
                increment(move, -bonus, currentContinuation);
            }

            for (size_t i = 0; i < (U32)quietsPlayed - 1; ++i)
            {
                if (singles.contains(quiets[i].first))
                {
                    continue;
                }
                increment(quiets[i].first, -bonus, currentContinuation);
            }

            increment(quiets[quietsPlayed - 1].first, bonus, currentContinuation);
        }
    }
};

#endif // HISTORY_H_INCLUDED
