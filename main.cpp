#include "uci.h"

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    uciLoop();

    return 0;
}
