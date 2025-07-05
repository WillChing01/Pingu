#ifndef DATUM_H_INCLUDED
#define DATUM_H_INCLUDED

#include "../pipeline/utils.h"

struct Datum {
    U64 pos[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    bool side;
    bool isDraw;
    bool isWin; // from pov of player to move.
    int ply;
    int totalPly;
    int qSearch;
    int inCheck;
    int increment;
    int timeLeft;
    int timeSpent;
    int totalTimeSpent;
    int startTime;
    int opponentTime;
};



#endif // DATUM_H_INCLUDED
