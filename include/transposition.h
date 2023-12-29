#ifndef TRANSPOSITION_H_INCLUDED
#define TRANSPOSITION_H_INCLUDED

#include <random>
#include <chrono>

#include "constants.h"

//search parameters.
const int MATE_SCORE = 32767;
const int MAXDEPTH = 63;

const int MATE_BOUND = MATE_SCORE - 100;

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
const int ageLimit = 2;

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
std::vector<std::pair<hashStore, hashStore> > hashTable(hashTableMask + 1);

U64 randomNums[781] = {};

const int ZHASH_PIECES[12] = {0,64,128,192,256,320,384,448,512,576,640,704};
const int ZHASH_TURN = 768;
const int ZHASH_CASTLES[4] = {769,770,771,772};
const int ZHASH_ENPASSANT[8] = {773,774,775,776,777,778,779,780};

void clearTT()
{
    rootCounter = 0;
    for (int i=0;i<(int)(hashTableMask + 1);i++)
    {
        hashTable[i] = std::pair<hashStore, hashStore>(emptyStore, emptyStore);
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
    hashTable.resize(hashTableMask + 1, std::pair<hashStore, hashStore>(emptyStore, emptyStore));
}

inline int ttProbeScore(int score, int ply)
{
    return 
        score > MATE_BOUND ? score - ply :
        score < -MATE_BOUND ? score + ply :
        score;
}

inline int ttSaveScore(int score, int ply)
{
    return
        score > MATE_BOUND ? score + ply :
        score < -MATE_BOUND ? score - ply :
        score;
}

inline void unpackInfo(U64 info, int ply)
{
    tableEntry.bestMove = (info & BESTMOVEMASK);
    tableEntry.isExact = (info & EXACTFLAGMASK);
    tableEntry.isBeta = (info & BETAFLAGMASK);
    tableEntry.depth = (info & DEPTHMASK) >> DEPTHSHIFT;
    tableEntry.evaluation = (info & EVALMASK) >> EVALSHIFT;
    if (!(info & EVALSIGNMASK)) {tableEntry.evaluation *= -1;}
    tableEntry.evaluation = ttProbeScore(tableEntry.evaluation, ply);
}

inline void ttSave(U64 zHash, int ply, int depth, U32 bestMove, int evaluation, bool isExact, bool isBeta)
{
    evaluation = ttSaveScore(evaluation, ply);
    tableStore.zHash = zHash;
    tableStore.info = bestMove;
    tableStore.info += ((U64)(depth) << DEPTHSHIFT) & DEPTHMASK;
    tableStore.info += ((U64)(abs(evaluation)) << EVALSHIFT) & EVALMASK;
    tableStore.info += ((U64)(evaluation > 0) << EVALSIGNSHIFT);
    tableStore.info += ((U64)(isExact) << EXACTFLAGSHIFT);
    tableStore.info += ((U64)(isBeta) << BETAFLAGSHIFT);
    tableStore.info += ((U64)(rootCounter) << AGESHIFT) & AGEMASK;

    if (depth >= (int)((hashTable[zHash & hashTableMask].first.info & DEPTHMASK) >> DEPTHSHIFT) ||
        rootCounter > (int)((hashTable[zHash & hashTableMask].first.info & AGEMASK) >> AGESHIFT) + ageLimit)
    {
        //fits in deep table.
        hashTable[zHash & hashTableMask].first = tableStore;
    }
    else
    {
        //fits in always table.
        hashTable[zHash & hashTableMask].second = tableStore;
    }
}

inline bool ttProbe(U64 zHash, int ply)
{
    if (hashTable[zHash & hashTableMask].first.zHash == zHash)
    {
        unpackInfo(hashTable[zHash & hashTableMask].first.info, ply);
        return true;
    }
    if (hashTable[zHash & hashTableMask].second.zHash == zHash)
    {
        unpackInfo(hashTable[zHash & hashTableMask].second.info, ply);
        return true;
    }

    return false;
}

void populateRandomNums()
{
    std::mt19937_64 _mt(974u);
    std::uniform_int_distribution<unsigned long long> _dist(0ull,ULLONG_MAX);

    for (int i=0;i<781;i++) {randomNums[i] = _dist(_mt);}
}

#endif // TRANSPOSITION_H_INCLUDED
