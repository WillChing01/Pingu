#ifndef TRANSPOSITION_H_INCLUDED
#define TRANSPOSITION_H_INCLUDED

#include <random>
#include <chrono>

#include "constants.h"

//define hash info compression.
const U64 BESTMOVEMASK = 4294967295ull;
const U64 EXACTFLAGMASK = 4294967296ull;
const U64 EXACTFLAGSHIFT = 32;
const U64 BETAFLAGMASK = 8589934592ull;
const U64 BETAFLAGSHIFT = 33;
const U64 DEPTHMASK = 1082331758592ull;
const U64 DEPTHSHIFT = 34;
const U64 EVALMASK = 36027697507336192ull;
const U64 EVALSHIFT = 40;
const U64 EVALSIGNMASK = 36028797018963968ull;
const U64 EVALSIGNSHIFT = 55;
const U64 AGEMASK = 18374686479671623680ull;
const U64 AGESHIFT = 56;

int rootCounter = 0;
const int ageLimit = 4;

struct hashEntry
{
    int depth;
    U32 bestMove;
    int evaluation;
    bool isExact;
    bool isBeta;
};

static hashEntry tableEntry;

struct hashStore
{
    U64 zHash;
    U64 info;
};

static hashStore tableStore;
const hashStore emptyStore =
{
    .zHash = 0,
    .info = 0,
};

U64 hashTableMask = 63; //start with a small hash.
std::vector<hashStore> hashTableAlways(hashTableMask + 1);
std::vector<hashStore> hashTableDeep(hashTableMask + 1);

U64 randomNums[781] = {};

const int ZHASH_PIECES[12] = {0,64,128,192,256,320,384,448,512,576,640,704};
const int ZHASH_TURN = 768;
const int ZHASH_CASTLES[4] = {769,770,771,772};
const int ZHASH_ENPASSANT[8] = {773,774,775,776,777,778,779,780};

void clearTT()
{
    for (int i=0;i<(int)(hashTableMask + 1);i++)
    {
        hashTableAlways[i] = emptyStore;
        hashTableDeep[i] = emptyStore;
    }
}

void resizeTT(U64 memory)
{
    //memory in MB.
    U64 length = (memory * 1048576ull)/((U64)sizeof(hashStore));
    length = length >> 1; //two tables to fill.
    hashTableMask = 1;
    while (hashTableMask <= length) {hashTableMask *= 2;}
    hashTableMask /= 2;
    hashTableMask -= 1;
    hashTableAlways.resize(hashTableMask + 1, emptyStore);
    hashTableDeep.resize(hashTableMask + 1, emptyStore);
}

inline void unpackInfo(U64 info)
{
    tableEntry.bestMove = (info & BESTMOVEMASK);
    tableEntry.isExact = (info & EXACTFLAGMASK);
    tableEntry.isBeta = (info & BETAFLAGMASK);
    tableEntry.depth = (info & DEPTHMASK) >> DEPTHSHIFT;
    tableEntry.evaluation = (info & EVALMASK) >> EVALSHIFT;
    if (!(info & EVALSIGNMASK)) {tableEntry.evaluation *= -1;}
}

inline void ttSave(U64 zHash, int depth, U32 bestMove, int evaluation, bool isExact, bool isBeta)
{
    tableStore.zHash = zHash;
    tableStore.info = bestMove;
    tableStore.info += ((U64)(depth) << DEPTHSHIFT) & DEPTHMASK;
    tableStore.info += ((U64)(abs(evaluation)) << EVALSHIFT) & EVALMASK;
    tableStore.info += ((U64)(evaluation > 0) << EVALSIGNSHIFT);
    tableStore.info += ((U64)(isExact) << EXACTFLAGSHIFT);
    tableStore.info += ((U64)(isBeta) << BETAFLAGSHIFT);
    tableStore.info += ((U64)(rootCounter) << AGESHIFT) & AGEMASK;

    if (depth >= (int)((hashTableDeep[zHash & hashTableMask].info & DEPTHMASK) >> DEPTHSHIFT) ||
        rootCounter > (int)((hashTableDeep[zHash & hashTableMask].info & AGEMASK) >> AGESHIFT) + ageLimit)
    {
        //fits in deep table.
        hashTableDeep[zHash & hashTableMask] = tableStore;
    }
    else
    {
        //fits in always table.
        hashTableAlways[zHash & hashTableMask] = tableStore;
    }
}

inline bool ttProbe(U64 zHash)
{
    if (hashTableDeep[zHash & hashTableMask].zHash == zHash)
    {
        unpackInfo(hashTableDeep[zHash & hashTableMask].info);
        return true;
    }
    if (hashTableAlways[zHash & hashTableMask].zHash == zHash)
    {
        unpackInfo(hashTableAlways[zHash & hashTableMask].info);
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
