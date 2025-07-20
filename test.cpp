#include "thread.h"
#include "time-network.h"

int main(int argc, const char** argv) {
    populateMagicTables();
    populateRandomNums();
    populateLmrTable();

    Thread thread;
    TimeNetwork network = TimeNetwork(&thread);

    if (argc == 2) {
        thread.b.setPositionFen(argv[1]);
        thread.b.display();
        std::cout << network.forward(1, 0, 1) << std::endl;
    }

    delete[] hashTable;

    return 0;
}
