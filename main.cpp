#include "uci.h"

int main()
{
    populateMagicTables();
    populateRandomNums();
    cacheKnightAttacks();
    clearTT();

    uciLoop();

    return 0;
}
