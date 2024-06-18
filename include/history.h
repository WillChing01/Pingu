#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <unordered_set>
#include <vector>

#include "constants.h"

const int HISTORY_MAX = 1048576;

class History
{
    public:
        //history table, scores[pieceType][to_square]
        int scores[12][64] = {};

        History() {}

        void clear()
        {
            for (int i=0;i<12;i++)
            {
                for (int j=0;j<64;j++) {scores[i][j] = 0;}
            }
        }

        void age(const int factor = 16)
        {
            for (int i=0;i<12;i++)
            {
                for (int j=0;j<64;j++)
                {
                    scores[i][j] /= factor;
                }
            }
        }

        void update(const std::unordered_set<U32> &singles, U32 cutMove, int depth)
        {
            bool shouldAge = false;
            int delta = depth * depth;

            //decrement history for single moves.
            for (const auto &move: singles)
            {
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                if (move == cutMove) {scores[pieceType][finishSquare] += delta;}
                else {scores[pieceType][finishSquare] -= delta;}
                if (scores[pieceType][finishSquare] < -HISTORY_MAX || scores[pieceType][finishSquare] > HISTORY_MAX) {shouldAge = true;}
            }

            //age history if necessary.
            if (shouldAge) {age();}
        }

        void update(const std::unordered_set<U32> &singles, const std::vector<std::pair<U32, int> > &quiets, int index, U32 cutMove, int depth)
        {
            bool shouldAge = false;
            int delta = depth * depth;

            //decrement history for single moves.
            for (const auto &move: singles)
            {
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                scores[pieceType][finishSquare] -= delta;
                if (scores[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
            }

            //decrement history for quiets.
            for (int i=0;i<index;i++)
            {
                if (singles.contains(quiets[i].first)) {continue;}
                U32 pieceType = (quiets[i].first & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 finishSquare = (quiets[i].first & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                scores[pieceType][finishSquare] -= delta;
                if (scores[pieceType][finishSquare] < -HISTORY_MAX) {shouldAge = true;}
            }

            //increment history for cut move.
            U32 pieceType = (cutMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 finishSquare = (cutMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            scores[pieceType][finishSquare] += delta;
            if (scores[pieceType][finishSquare] > HISTORY_MAX) {shouldAge = true;}

            //age history if necessary.
            if (shouldAge) {age();}
        }
};

#endif // HISTORY_H_INCLUDED
