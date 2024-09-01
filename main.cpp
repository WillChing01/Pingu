#include <cstring>
#include "uci.h"
#include "bench.h"

int main(int argc, const char* argv[])
{
    populateMagicTables();
    populateRandomNums();

    switch(argc)
    {
        case 1:
            uciLoop();
            break;
        case 2:
            if (std::strcmp(argv[1], "bench") == 0) {benchCommand();}
            break;
    }

    delete[] hashTable;

    return 0;
}
