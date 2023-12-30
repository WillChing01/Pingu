#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <vector>

#include "constants.h"

const int HISTORY_MAX = 1048576;
const int COUNTERMOVE_HISTORY_MAX = 1048576;
const int HISTORY_AGE_FACTOR = 16;

int HISTORY[12][64] = {};

int COUNTERMOVE_HISTORY[12][64][12][64] = {};

inline void clearHistory()
{
    for (int i=0;i<12;i++) {for (int j=0;j<64;j++) {HISTORY[i][j] = 0;}}
    for (int i=0;i<12;i++) {for (int j=0;j<64;j++) {for (int k=0;k<12;k++) {for (int l=0;l<64;l++) {COUNTERMOVE_HISTORY[i][j][k][l] = 0;}}}}
}

inline void ageHistory(int prevPieceType, int prevToSquare)
{
    for (int i=0;i<12;i++)
    {
        for (int j=0;j<64;j++)
        {
            HISTORY[i][j] /= HISTORY_AGE_FACTOR;
            COUNTERMOVE_HISTORY[prevPieceType][prevToSquare][i][j] /= HISTORY_AGE_FACTOR;
        }
    }
}

inline void updateHistory(const std::vector<U32> &quiets, U32 cutMove, U32 prevMove, int depth)
{
    bool shouldAge = false;

    int prevPieceType = (prevMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
    int prevToSquare = (prevMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

    //decrease history of previous quiets.
    for (const auto &move: quiets)
    {
        int pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        int toSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
        
        HISTORY[pieceType][toSquare] -= depth * depth;
        COUNTERMOVE_HISTORY[prevPieceType][prevToSquare][pieceType][toSquare] -= depth*depth;

        if (HISTORY[pieceType][toSquare] < -HISTORY_MAX ||
            COUNTERMOVE_HISTORY[prevPieceType][prevToSquare][pieceType][toSquare] < -COUNTERMOVE_HISTORY_MAX)
        {
            shouldAge = true;
        }
    }

    //increase history of cut move.
    int pieceType = (cutMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
    int toSquare = (cutMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

    HISTORY[pieceType][toSquare] += depth * depth;
    COUNTERMOVE_HISTORY[prevPieceType][prevToSquare][pieceType][toSquare] -= depth*depth;

    if (HISTORY[pieceType][toSquare] > HISTORY_MAX ||
        COUNTERMOVE_HISTORY[prevPieceType][prevToSquare][pieceType][toSquare] > COUNTERMOVE_HISTORY_MAX)
    {
        shouldAge = true;
    }

    if (shouldAge) {ageHistory(prevPieceType, prevToSquare);}
}

#endif // HISTORY_H_INCLUDED
