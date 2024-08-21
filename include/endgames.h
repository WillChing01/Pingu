#ifndef ENDGAMES_H_INCLUDED
#define ENDGAMES_H_INCLUDED

#include <vector>
#include <string>
#include <unordered_map>

#include "constants.h"
#include "material-counter.h"

//{Q, q, R, r, B, b, N, n, P, p}

struct Endgame
{
    U64 signature;
    int factor;
};

const int endgameHashMaxPawns = 1;
std::unordered_map<U64, int> endgameHash;

U64 getSignature(const U64* values)
{
    U64 signature = 0;
    for (int i=0;i<10;i++) {signature += pieceKeys[i] * values[i];}
    return signature;
}

std::pair<Endgame, Endgame> parseEndgame(const std::string &config, const int factor)
{
    U64 values[2][10] = {};

    bool side = 1;
    for (const char &c: config)
    {
        switch (c)
        {
            case 'K':
                side = !side;
                break;
            case 'Q':
                ++values[0][0+side];
                ++values[1][0+!side];
                break;
            case 'R':
                ++values[0][2+side];
                ++values[1][2+!side];
                break;
            case 'B':
                ++values[0][4+side];
                ++values[1][4+!side];
                break;
            case 'N':
                ++values[0][6+side];
                ++values[1][6+!side];
                break;
            case 'P':
                ++values[0][8+side];
                ++values[1][8+!side];
                break;
        }
    }

    Endgame white = {getSignature(values[0]), factor};
    Endgame black = {getSignature(values[1]), factor};

    return std::pair<Endgame, Endgame>(white, black);
}

void populateEndgames()
{
    const int deadDrawFactor = 1 << 16;

    //KK dead draw.
    endgameHash.insert({0, deadDrawFactor});

    const std::vector<std::pair<std::string, int> > endgames = {
        {"KNK", deadDrawFactor},
        {"KBK", deadDrawFactor},

        {"KNNK", 64},
        {"KRKB", 16},
        {"KRKN", 16},
        {"KNKP", 16},
        {"KBKP", 16},
        {"KRBKR", 16},
        {"KRNKR", 16},

        {"KNPKN", 8},
        {"KNPKB", 8},
        {"KRKNP", 8},
        {"KRKBP", 8},

        {"KQKNN", 16},
        {"KQKBB", 16},
        {"KQKRN", 16},
        {"KQKRB", 16},
        {"KQKRNN", 16},
        {"KQKRBB", 16},
        {"KQKRBN", 16},
        {"KQNKRR", 16},
        {"KQKNNN", 16},
        {"KQKBNN", 16},
        {"KQKBBN", 16},
        {"KQKBBB", 16},

        {"KBNNKR", 16},
        {"KRNKNN", 16},
        {"KRNKBN", 16},
        {"KRBKBB", 16},
        {"KRNKBB", 16},

        {"KBBKB", 16},
        {"KNNKB", 16},
        {"KNNKN", 16},
        {"KBNKB", 16},
    };

    for (const auto &info: endgames)
    {
        std::pair<Endgame, Endgame> endgamePair = parseEndgame(info.first, info.second);
        endgameHash.insert({endgamePair.first.signature, endgamePair.first.factor});
        endgameHash.insert({endgamePair.second.signature, endgamePair.second.factor});
    }
}

#endif // ENDGAMES_H_INCLUDED
