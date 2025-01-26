#ifndef PAWN_H_INCLUDED
#define PAWN_H_INCLUDED

#include "constants.h"

inline U64 pawnAttacks(U64 b, bool side) {
    if (!side) {
        // white.
        return ((b & NOT_A_FILE) << 7) | ((b & NOT_H_FILE) << 9);
    }
    return ((b & NOT_A_FILE) >> 9) | ((b & NOT_H_FILE) >> 7);
}

#endif // PAWN_H_INCLUDED
