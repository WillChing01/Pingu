#ifndef TRANSPOSITION_H_INCLUDED
#define TRANSPOSITION_H_INCLUDED

#include <random>
#include <chrono>

int rootCounter = 0;
const int ageLimit = 4;

struct hashEntry
{
    U64 zHash;
    int depth;
    U32 bestMove;
    int evaluation;
    bool isExact;
    bool isBeta;
};

U64 hashTableSize = 2796202; //default 64 MB for each table.
vector<hashEntry> hashTableAlways(hashTableSize);
vector<hashEntry> hashTableDeep(hashTableSize);
vector<int> hashTableAge(hashTableSize,0);

hashEntry tableEntry;
const hashEntry emptyEntry =
{
    .zHash = 0,
    .depth = 0,
    .bestMove = 0,
    .evaluation = 0,
    .isExact = 0,
    .isBeta = 0,
};

U64 randomNums[781] = {};

const int ZHASH_PIECES[12] = {0,64,128,192,256,320,384,448,512,576,640,704};
const int ZHASH_TURN = 768;
const int ZHASH_CASTLES[4] = {769,770,771,772};
const int ZHASH_ENPASSANT[8] = {773,774,775,776,777,778,779,780};

void clearTT()
{
    for (int i=0;i<(int)hashTableSize;i++)
    {
        hashTableAlways[i] = emptyEntry;
        hashTableDeep[i] = emptyEntry;
        hashTableAge[i] = 0;
    }
}

void resizeTT(U64 memory)
{
    //memory in MB.
    U64 length = (memory * 1048576ull)/((U64)sizeof(hashEntry));
    length = length >> 1; //two tables to fill.
    hashTableSize = length;
    hashTableAlways.resize(length, emptyEntry);
    hashTableDeep.resize(length, emptyEntry);
    hashTableAge.resize(length, 0);
}

inline void ttSave(U64 zHash, int depth, U32 bestMove, int evaluation, bool isExact, bool isBeta)
{
    tableEntry.zHash = zHash;
    tableEntry.depth = depth;
    tableEntry.bestMove = bestMove;
    tableEntry.evaluation = evaluation;
    tableEntry.isExact = isExact;
    tableEntry.isBeta = isBeta;

    if (depth >= hashTableDeep[zHash % hashTableSize].depth ||
        rootCounter > hashTableAge[zHash % hashTableSize] + ageLimit)
    {
        //fits in deep table.
        hashTableDeep[zHash % hashTableSize] = tableEntry;
        hashTableAge[zHash % hashTableSize] = rootCounter;
    }
    else
    {
        //fits in always table.
        hashTableAlways[zHash % hashTableSize] = tableEntry;
    }
}

inline bool ttProbe(U64 zHash, hashEntry &entry)
{
    if (hashTableDeep[zHash % hashTableSize].zHash == zHash)
    {
        entry = hashTableDeep[zHash % hashTableSize];
        return true;
    }
    if (hashTableAlways[zHash % hashTableSize].zHash == zHash)
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
