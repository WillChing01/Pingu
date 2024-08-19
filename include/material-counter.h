#ifndef MATERIAL_COUNTER_H_INCLUDED
#define MATERIAL_COUNTER_H_INCLUDED

#include "constants.h"

const int pieceWeights[12] = {0, 0, 9, -9, 5, -5, 3, -3, 3, -3, 1, -1};

class MaterialCounter
{
    public:
        int values[12] = {};
        int materialDiff = 0;

        MaterialCounter() {}

        MaterialCounter(const int* _values) {setValues(_values);}

        void setValues(const int* _values)
        {
            materialDiff = 0;
            for (int i=0;i<12;i++)
            {
                values[i] = _values[i];
                materialDiff += pieceWeights[i] * _values[i];
            }
        }

        bool is_equal(const int* _values)
        {
            for (int i=2;i<12;i++) {if (values[i] != _values[i]) {return false;}}
            return true;
        }

        void addPiece(const int i)
        {
            ++values[i];
            materialDiff += pieceWeights[i]; 
        }

        void removePiece(const int i)
        {
            --values[i];
            materialDiff -= pieceWeights[i];
        }
};

#endif // MATERIAL_COUNTER_H_INCLUDED
