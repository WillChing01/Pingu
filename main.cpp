#include <iostream>

#include "board.h"
#include "perft.h"
#include "game.h"

using namespace std;

int main()
{
    populateMagicTables();
//    testInitialPosition();
//    testKiwipetePosition();

//    searchSpeedTest(6);

    playCPU(6);

    system("pause");
    return 0;
}
