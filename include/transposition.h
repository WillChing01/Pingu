#ifndef TRANSPOSITION_H_INCLUDED
#define TRANSPOSITION_H_INCLUDED

#include <random>
#include <chrono>

struct hashEntry
{
    U64 zHash;
    int depth;
    U32 bestMove;
    int evaluation;
    bool isExact;
    bool isBeta;
};

const U64 hashTableSize = 65537;
hashEntry hashTableAlways[hashTableSize] = {};
hashEntry hashTableDeep[hashTableSize] = {};
hashEntry tableEntry;

U64 randomNums[781] = {};

const int ZHASH_PIECES[12] = {0,64,128,192,256,320,384,448,512,576,640,704};
const int ZHASH_TURN = 768;
const int ZHASH_CASTLES[4] = {769,770,771,772};
const int ZHASH_ENPASSANT[8] = {773,774,775,776,777,778,779,780};

inline void ttSave(U64 zHash, int depth, U32 bestMove, int evaluation, bool isExact, bool isBeta)
{
    tableEntry.zHash = zHash;
    tableEntry.depth = depth;
    tableEntry.bestMove = bestMove;
    tableEntry.evaluation = evaluation;
    tableEntry.isExact = isExact;
    tableEntry.isBeta = isBeta;

    if (depth >= hashTableDeep[zHash % hashTableSize].depth)
    {
        //fits in deep table.
        hashTableDeep[zHash % hashTableSize] = tableEntry;
    }
    else
    {
        //fits in always table.
        hashTableAlways[zHash % hashTableSize] = tableEntry;
    }
}

inline bool ttProbe(U64 zHash, int depth, hashEntry &entry)
{
    if (hashTableDeep[zHash % hashTableSize].zHash == zHash &&
        hashTableDeep[zHash % hashTableSize].depth >= depth)
    {
        entry = hashTableDeep[zHash % hashTableSize];
        return true;
    }
    if (hashTableAlways[zHash % hashTableSize].zHash == zHash &&
        hashTableAlways[zHash % hashTableSize].depth >= depth)
    {
        entry = hashTableAlways[zHash % hashTableSize];
        return true;
    }

    return false;
}

void populateRandomNums()
{
    std::random_device _rd;
    std::size_t seed;

    if (_rd.entropy()) {seed = _rd();}
    else {seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();}

    std::mt19937 _mt(seed);
    std::uniform_int_distribution<unsigned long long> _dist(0ull,ULLONG_MAX);

    for (int i=0;i<781;i++) {randomNums[i] = _dist(_mt);}
}

#endif // TRANSPOSITION_H_INCLUDED
