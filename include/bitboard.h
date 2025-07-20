#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include "constants.h"

#include <bitset>
#include <iostream>
#include <string>

inline void displayBitboard(U64 bitboard) {
    std::string temp = std::bitset<64>(bitboard).to_string();

    std::cout << std::endl;
    for (int i = 7; i >= 0; i--) {
        for (int j = 0; j < 8; j++) {
            std::cout << " " << temp[63 - (8 * i + j)];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

inline int popLSB(U64& b) {
    int _i = __builtin_ctzll(b);
    b &= b - 1ull;
    return _i;
}

inline std::string toCoord(int square) {
    std::string cols = "abcdefgh";
    return cols[square % 8] + std::to_string(square / 8 + 1);
}

inline int toSquare(std::string coord) {
    std::string cols = "abcdefgh";
    int square = 0;
    for (int i = 0; i < (int)cols.length(); i++) {
        if (cols[i] == coord[0]) {
            square += i;
            break;
        }
    }
    square += 8 * (coord[1] - '0') - 8;
    return square;
}

#endif // BITBOARD_H_INCLUDED
