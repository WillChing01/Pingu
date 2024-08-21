#ifndef MATERIAL_COUNTER_H_INCLUDED
#define MATERIAL_COUNTER_H_INCLUDED

#include <bit>

#include "constants.h"

const U64 pieceBase = 12;
const U64 pieceKeys[10] = {1, 12, 144, 1728, 20736, 248832, 2985984, 35831808, 429981696, 5159780352};

class MaterialCounter
{
    public:
        U64 signature = 0;

        MaterialCounter() {}

        MaterialCounter(const U64* pieces) {setValues(pieces);}

        void setValues(const U64* pieces)
        {
            signature = 0;
            for (int i=2;i<12;i++) {signature += pieceKeys[i-2] * std::popcount(pieces[i]);}
        }

        void addPiece(const U32 i) {signature += pieceKeys[i-2];}

        void removePiece(const U32 i) {signature -= pieceKeys[i-2];}

        U32 getCount(const U32 i) {return (signature / pieceKeys[i-2]) % pieceBase;}
};

#endif // MATERIAL_COUNTER_H_INCLUDED
