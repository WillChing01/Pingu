#ifndef SLIDER_H_INCLUDED
#define SLIDER_H_INCLUDED

#include "constants.h"

inline U64 nortFillOccluded(U64 g, U64 p) {
    U64 old = g;
    g |= p & (g << 8);
    p &= p << 8;
    g |= p & (g << 16);
    p &= p << 16;
    g |= p & (g << 32);
    g |= g << 8;
    return g & ~old;
}

inline U64 soutFillOccluded(U64 g, U64 p) {
    U64 old = g;
    g |= p & (g >> 8);
    p &= p >> 8;
    g |= p & (g >> 16);
    p &= p >> 16;
    g |= p & (g >> 32);
    g |= g >> 8;
    return g & ~old;
}

inline U64 eastFillOccluded(U64 g, U64 p) {
    U64 old = g;
    p &= NOT_A_FILE;
    g |= p & (g << 1);
    p &= p << 1;
    g |= p & (g << 2);
    p &= p << 2;
    g |= p & (g << 4);
    g |= (g & NOT_H_FILE) << 1;
    return g & ~old;
}

inline U64 noEaFillOccluded(U64 g, U64 p) {
    U64 old = g;
    p &= NOT_A_FILE;
    g |= p & (g << 9);
    p &= p << 9;
    g |= p & (g << 18);
    p &= p << 18;
    g |= p & (g << 36);
    g |= (g & NOT_H_FILE) << 9;
    return g & ~old;
}

inline U64 soEaFillOccluded(U64 g, U64 p) {
    U64 old = g;
    p &= NOT_A_FILE;
    g |= p & (g >> 7);
    p &= p >> 7;
    g |= p & (g >> 14);
    p &= p >> 14;
    g |= p & (g >> 28);
    g |= (g & NOT_H_FILE) >> 7;
    return g & ~old;
}

inline U64 westFillOccluded(U64 g, U64 p) {
    U64 old = g;
    p &= NOT_H_FILE;
    g |= p & (g >> 1);
    p &= p >> 1;
    g |= p & (g >> 2);
    p &= p >> 2;
    g |= p & (g >> 4);
    g |= (g & NOT_A_FILE) >> 1;
    return g & ~old;
}

inline U64 soWeFillOccluded(U64 g, U64 p) {
    U64 old = g;
    p &= NOT_H_FILE;
    g |= p & (g >> 9);
    p &= p >> 9;
    g |= p & (g >> 18);
    p &= p >> 18;
    g |= p & (g >> 36);
    g |= (g & NOT_A_FILE) >> 9;
    return g & ~old;
}

inline U64 noWeFillOccluded(U64 g, U64 p) {
    U64 old = g;
    p &= NOT_H_FILE;
    g |= p & (g << 7);
    p &= p << 7;
    g |= p & (g << 14);
    p &= p << 14;
    g |= p & (g << 28);
    g |= (g & NOT_A_FILE) << 7;
    return g & ~old;
}

inline U64 rookAttacks(U64 b, U64 p) {
    // p is permitted squares.
    U64 nortAttacks = nortFillOccluded(b, p);
    U64 soutAttacks = soutFillOccluded(b, p);
    U64 westAttacks = westFillOccluded(b, p);
    U64 eastAttacks = eastFillOccluded(b, p);

    return nortAttacks | soutAttacks | westAttacks | eastAttacks;
}

inline U64 bishopAttacks(U64 b, U64 p) {
    // p is permitted squares.
    U64 noEaAttacks = noEaFillOccluded(b, p);
    U64 soEaAttacks = soEaFillOccluded(b, p);
    U64 soWeAttacks = soWeFillOccluded(b, p);
    U64 noWeAttacks = noWeFillOccluded(b, p);

    return noEaAttacks | soEaAttacks | soWeAttacks | noWeAttacks;
}

inline U64 queenAttacks(U64 b, U64 p) {
    // p is permitted squares.
    return rookAttacks(b, p) | bishopAttacks(b, p);
}

#endif // SLIDER_H_INCLUDED
