#include <cstring>
#include "uci.h"

int main(int argc, const char* argv[])
{
    populateMagicTables();
    populateRandomNums();
    clearTT();

    switch(argc)
    {
        case 1:
            uciLoop();
            break;
        case 2:
            if (std::strcmp(argv[1], "bench") == 0) {Board b; benchCommand(b);}
            break;
    }

    return 0;
}
