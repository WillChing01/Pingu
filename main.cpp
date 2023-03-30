#include <iostream>

#include "board.h"
#include "test-suite.h"

using namespace std;

int main()
{
    populateMagicTables();
    testInitialPosition();

    system("pause");
    return 0;
}
