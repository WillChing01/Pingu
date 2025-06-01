#include "uci.h"
#include "bench.h"
#include "gensfen.h"
#include "engine-commands.h"
#include "process-time-pgn.h"

#include <cstring>
#include <unordered_map>

int main(int argc, const char** argv) {
    populateMagicTables();
    populateRandomNums();
    populateLmrTable();

    if (argc == 1) {
        uciLoop();
    } else {
        const std::unordered_map<const char*, void (*)(int, const char**)> commands = {
            {"bench", benchCommand},
            {"gensfen", gensfenCommand},
            {"process-time-pgn", processTimePgnCommand},
            {"-h", displayHelpCLI},
            {"--help", displayHelpCLI},
        };

        for (const auto& [command, callback] : commands) {
            if (!std::strcmp(argv[1], command)) {
                callback(argc, argv);
                break;
            }
        }
    }

    delete[] hashTable;

    return 0;
}
