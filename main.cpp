#ifdef TUNING

#include <chrono>

#include "magic.h"
#include "tuning.h"

int main()
{
    populateMagicTables();
    populateRandomNums();

    readWeights();
    readPositions();

    optimiseFeatures(10000, 0.00001);

    return 0;
}

#else

#include "uci.h"

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    uciLoop();

    return 0;
}

#endif
