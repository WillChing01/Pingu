#include <iostream>

#include "board.h"
#include "perft.h"
#include "game.h"
#include "uci.h"

using namespace std;

int main()
{
    populateMagicTables();
    populateRandomNums();
    clearTT();
//    testInitialPosition();
//    testKiwipetePosition();

    uciLoop();

    return 0;
}
