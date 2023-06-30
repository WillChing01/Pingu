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

//    searchSpeedTest(6);

//    playCPU(5000);

    uciLoop();

//    system("pause");
    return 0;
}
