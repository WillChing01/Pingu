#include "uci.h"

using namespace std;

#ifdef PERFT

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    bool good1 = testInitialPosition();
    bool good2 = testKiwipetePosition();

    return !(good1 && good2);
}

#elifdef TUNING

#include <chrono>

#include "tuning.h"

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    readPositions();

    optimiseMaterial(8);
    optimiseMaterial(4);
    optimiseMaterial(2);
    optimiseMaterial(1);

    return 0;
}

#else

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    uciLoop();

    return 0;
}

#endif
