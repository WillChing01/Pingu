#include "uci.h"

#ifdef TUNING

#include <chrono>

#include "tuning.h"

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    readPositions();
    populateEval();

    // optimiseMaterial(8);
    // optimiseMaterial(4);
    // optimiseMaterial(2);
    // optimiseMaterial(1);

    populateDataset();

    optimiseFeatures();

    return 0;
}

#else

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    populateMagicTables();
    populateRandomNums();
    clearTT();

    uciLoop();

    return 0;
}

#endif
