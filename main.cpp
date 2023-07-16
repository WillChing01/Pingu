#include <iostream>

#include "board.h"
#include "perft.h"
#include "game.h"
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
