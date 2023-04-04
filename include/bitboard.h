#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include <string>

#include "constants.h"

using namespace std;

void displayBitboard(U64 bitboard)
{
    string temp=bitset<64>(bitboard).to_string();

    cout << endl;
    for (int i=7;i>=0;i--)
    {
        for (int j=0;j<8;j++)
        {
            cout << " " << temp[63-(8*i+j)];
        } cout << endl;
    } cout << endl;
}

static int _i=0;

inline int popLSB(U64 &b)
{
    _i=__builtin_ctzll(b);
    b &= b-1;
    return _i;
}

string toCoord(int square)
{
    string cols = "abcdefgh";

    return cols[square%8]+to_string(square/8+1);
}

const U64 h1 = (0x5555555555555555);
const U64 h2 = (0x3333333333333333);
const U64 h4 = (0x0F0F0F0F0F0F0F0F);
const U64 v1 = (0x00FF00FF00FF00FF);
const U64 v2 = (0x0000FFFF0000FFFF);

inline U64 rotate180 (U64 x) {
    x = ((x >>  1) & h1) | ((x & h1) <<  1);
    x = ((x >>  2) & h2) | ((x & h2) <<  2);
    x = ((x >>  4) & h4) | ((x & h4) <<  4);
    x = ((x >>  8) & v1) | ((x & v1) <<  8);
    x = ((x >> 16) & v2) | ((x & v2) << 16);
    x = ( x >> 32)       | ( x       << 32);
    return x;
}

#endif // BITBOARD_H_INCLUDED
