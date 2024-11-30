#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <filesystem>
#include <unordered_map>
#include <vector>

typedef unsigned long long U64;

// eval and result given relative to side.
struct datum
{
    U64 pos[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    unsigned char kingPos[2];
    short eval;
    bool isDraw;
    bool result;
    bool side;
};

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

std::vector<std::filesystem::path> getFiles(const std::filesystem::path& path, const std::filesystem::path& ext)
{
    std::vector<std::filesystem::path> res = {};

    for (const auto& entry: std::filesystem::recursive_directory_iterator(path))
    {
        const std::filesystem::path entryPath= entry.path();
        if (entryPath.extension() == ext)
        {
            res.push_back(entryPath);
        }
    }

    return res;
}

#endif // UTILS_H_INCLUDED
