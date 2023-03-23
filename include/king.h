#ifndef KING_H_INCLUDED
#define KING_H_INCLUDED

#include "constants.h"

U64 kingAttacks(U64 b)
{
    U64 res = b | ((b & NOT_H_FILE) << 1) | ((b & NOT_A_FILE) >> 1);

    res = res | (res << 8) | (res >> 8);

    res = res ^ b;

    return res;
}

#endif // KING_H_INCLUDED
