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

#endif // BITBOARD_H_INCLUDED
