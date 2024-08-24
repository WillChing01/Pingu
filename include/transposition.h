#ifndef TRANSPOSITION_H_INCLUDED
#define TRANSPOSITION_H_INCLUDED

#include <atomic>
#include <random>

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

struct hashStore
{
    std::atomic<U64> zHash = 0;
    std::atomic<U64> info = 0;
};

U64 hashTableMask = 63; //start with a small hash.
std::pair<hashStore, hashStore> * hashTable = new std::pair<hashStore, hashStore>[hashTableMask + 1]();

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
        hashTable[i].first.zHash = 0;
        hashTable[i].first.info = 0;
        hashTable[i].second.zHash = 0;
        hashTable[i].second.info = 0;
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
    delete[] hashTable;
    hashTable = new std::pair<hashStore, hashStore>[hashTableMask + 1]();
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

inline U32 getHashMove(const U64 info)
{
    return info & BESTMOVEMASK;
}

inline bool getHashExactFlag(const U64 info)
{
    return info & EXACTFLAGMASK;
}

inline bool getHashBetaFlag(const U64 info)
{
    return info & BETAFLAGMASK;
}

inline int getHashDepth(const U64 info)
{
    return (info & DEPTHMASK) >> DEPTHSHIFT;
}

inline int getHashEval(const U64 info, const int ply)
{
    int eval = (info & EVALMASK) >> EVALSHIFT;
    if (!(info & EVALSIGNMASK)) {eval = -eval;}
    eval = ttProbeScore(eval, ply);

    return eval;
}

inline void ttSave(U64 zHash, int ply, int depth, U32 bestMove, int evaluation, bool isExact, bool isBeta)
{
    U64 index = zHash & hashTableMask;

    int deepDepth = (hashTable[index].first.info & DEPTHMASK) >> DEPTHSHIFT;
    int deepAge = (hashTable[index].first.info & AGEMASK) >> AGESHIFT;

    evaluation = ttSaveScore(evaluation, ply);

    if (depth >= deepDepth || (rootCounter > deepAge + ageLimit))
    {
        //fits in deep table.
        hashTable[index].first.zHash = zHash;
        hashTable[index].first.info =
            bestMove +
            (((U64)(depth) << DEPTHSHIFT) & DEPTHMASK) +
            (((U64)(abs(evaluation)) << EVALSHIFT) & EVALMASK) +
            ((U64)(evaluation > 0) << EVALSIGNSHIFT) +
            ((U64)(isExact) << EXACTFLAGSHIFT) +
            ((U64)(isBeta) << BETAFLAGSHIFT) +
            (((U64)(rootCounter) << AGESHIFT) & AGEMASK);
    }
    else
    {
        //fits in always table.
        hashTable[index].second.zHash = zHash;
        hashTable[index].second.info =
            bestMove +
            (((U64)(depth) << DEPTHSHIFT) & DEPTHMASK) +
            (((U64)(abs(evaluation)) << EVALSHIFT) & EVALMASK) +
            ((U64)(evaluation > 0) << EVALSIGNSHIFT) +
            ((U64)(isExact) << EXACTFLAGSHIFT) +
            ((U64)(isBeta) << BETAFLAGSHIFT) +
            (((U64)(rootCounter) << AGESHIFT) & AGEMASK);
    }
}

inline U64 ttProbe(const U64 zHash)
{
    U64 index = zHash & hashTableMask;

    //check deep table then always table.
    if (hashTable[index].first.zHash == zHash) {return hashTable[index].first.info;}
    if (hashTable[index].second.zHash == zHash) {return hashTable[index].second.info;}

    return 0;
}

void populateRandomNums()
{
    std::seed_seq ss{866052240, 524600418, 294500052, 845566473};
    std::mt19937_64 _mt(ss);
    std::uniform_int_distribution<unsigned long long> _dist(1ull,ULLONG_MAX);

    for (int i=0;i<781;i++) {randomNums[i] = _dist(_mt);}
}

#endif // TRANSPOSITION_H_INCLUDED
