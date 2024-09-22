#include <cstring>
#include <unordered_map>
#include "uci.h"
#include "bench.h"
#include "gensfen.h"
#include "engine-commands.h"

int main(int argc, const char** argv)
{
    populateMagicTables();
    populateRandomNums();

    if (argc == 1) {uciLoop();}
    else
    {
        const std::unordered_map<const char*, void(*)(int, const char**)> commands = {
            {"bench", benchCommand},
            {"gensfen", gensfenCommand},
            {"-h", displayHelpCLI},
            {"--help", displayHelpCLI},
        };

        for (const auto &[command, callback]: commands)
        {
            if (!std::strcmp(argv[1], command))
            {
                callback(argc, argv);
                break;
            }
        }
    }

    delete[] hashTable;

    return 0;
}
