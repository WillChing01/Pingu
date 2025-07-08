#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <filesystem>
#include <unordered_map>
#include <vector>

typedef unsigned long long U64;

const std::unordered_map<unsigned char, unsigned char> pieceTypeMap = {
    {'K', 0},
    {'k', 1},
    {'Q', 2},
    {'q', 3},
    {'R', 4},
    {'r', 5},
    {'B', 6},
    {'b', 7},
    {'N', 8},
    {'n', 9},
    {'P', 10},
    {'p', 11},
};

const U64 masks[16] = {
    0xFull,
    0xF0ull,
    0xF00ull,
    0xF000ull,
    0xF0000ull,
    0xF00000ull,
    0xF000000ull,
    0xF0000000ull,
    0xF00000000ull,
    0xF000000000ull,
    0xF0000000000ull,
    0xF00000000000ull,
    0xF000000000000ull,
    0xF0000000000000ull,
    0xF00000000000000ull,
    0xF000000000000000ull,
};

inline void parseFen(const std::string& fen, U64* const res) {
    unsigned char square = 56;
    for (const unsigned char x : fen) {
        switch (x) {
        case '/':
            square -= 16;
            break;
        case '1':
            ++square;
            break;
        case '2':
            square += 2;
            break;
        case '3':
            square += 3;
            break;
        case '4':
            square += 4;
            break;
        case '5':
            square += 5;
            break;
        case '6':
            square += 6;
            break;
        case '7':
            square += 7;
            break;
        case '8':
            square += 8;
            break;
        default:
            res[square >> 4] -= (15ull - (U64)pieceTypeMap.at(x)) << (U64)((square & 15) << 2);
            ++square;
            break;
        }
    }
}

template <typename F>
inline void parsePos(const U64* const pos, const F& callback) {
    for (size_t i = 0; i < 4; ++i) {
        U64 x = ~pos[i];
        while (x) {
            U64 j = __builtin_ctzll(x) >> 2ull;
            int pieceType = 15 - (U64)((x & masks[j]) >> (j << 2ull));
            int square = 16 * i + j;
            callback(pieceType, square);
            x &= ~masks[j];
        }
    }
}

std::vector<std::filesystem::path> getFiles(const std::filesystem::path& path, const std::filesystem::path& ext) {
    std::vector<std::filesystem::path> res = {};

    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        const std::filesystem::path entryPath = entry.path();
        if (entryPath.extension() == ext) {
            res.push_back(entryPath);
        }
    }

    return res;
}

#endif // UTILS_H_INCLUDED
