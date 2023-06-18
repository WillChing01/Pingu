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
static hashEntry hashTable[hashTableSize] = {};

U64 randomNums[781] = {};

const int ZHASH_PIECES[12] = {0,64,128,192,256,320,384,448,512,576,640,704};
const int ZHASH_TURN = 768;
const int ZHASH_CASTLES[4] = {769,770,771,772};
const int ZHASH_ENPASSANT[8] = {773,774,775,776,777,778,779,780};

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
