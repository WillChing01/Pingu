#ifndef ENGINE_COMMANDS_H_INCLUDED
#define ENGINE_COMMANDS_H_INCLUDED

#include <vector>
#include <string>

struct engineCommand
{
    std::string name;
    std::string desc;
    std::vector<std::string> usage;
    std::vector<std::pair<std::string,std::string> > options;

    engineCommand(
        std::string commandName,
        std::string commandDesc,
        std::vector<std::string> commandUsage,
        std::vector<std::pair<std::string,std::string> > commandOptions
    )
    {
        name = commandName;
        desc = commandDesc;
        usage = commandUsage;
        options = commandOptions;
    }
};

const std::vector<engineCommand> COMMANDS = {
    engineCommand(
        "uci",
        "tell engine to use uci, id itself, and list options",
        {},
        {}
    ),
    engineCommand(
        "isready",
        "ping the engine to check that it is ready",
        {},
        {}
    ),
    engineCommand(
        "setoption",
        "change the internal parameters of the engine",
        {
            "setoption name <id> [value <x>]",
            "e.g. setoption name Hash value 256",
            "e.g. setoption name Clear Hash"
        },
        {
            {"name <id>", "the name of the option"},
            {"value <x>", "the value of the option"}
        }
    ),
    engineCommand(
        "ucinewgame",
        "tell the engine to prepare for a new game",
        {},
        {}
    ),
    engineCommand(
        "position",
        "set up the position on the internal board",
        {
            "position (fen <fenstring> | startpos) [moves <move1> ... <movei>]",
            "e.g. position startpos moves e2e4 e7e5",
            "e.g. position fen 8/5k2/8/5N2/5Q2/2K5/8/8 w - - 0 1"
        },
        {
            {"fen <fenstring>", "the position as described by FEN"},
            {"startpos", "the starting position"},
            {"moves <move1> ... <movei>", "moves to be played from the given position, in long algebraic notation"}
        }
    ),
    engineCommand(
        "go",
        "start calculating on the current position",
        {
            "go (depth <n> | movetime <ms> | infinite)",
            "go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <n>]",
            "e.g. go depth 5",
            "e.g. go wtime 1000 btime 1000 winc 100 binc 100 movestogo 10"
        },
        {
            {"depth <n>", "positive integer depth in units of ply"},
            {"movetime <ms>", "time in ms to search for"},
            {"infinite", "search forever until manually terminated"},
            {"wtime <ms>", "white time in ms"},
            {"btime <ms>", "black time in ms"},
            {"winc <ms>", "white increment in ms"},
            {"binc <ms>", "black increment in ms"},
            {"movestogo <n>", "moves until time control, otherwise sudden death"}
        }
    ),
    engineCommand(
        "stop",
        "stop calculating as soon as possible",
        {},
        {}
    ),
    engineCommand(
        "eval",
        "display static evaluation of current position",
        {},
        {}
    ),
    engineCommand(
        "see",
        "perform static exchange evaluation in current position",
        {
            "see move <x>",
            "e.g. see move e4d5"
        },
        {
            {"move <x>", "move in long algebraic notation"}
        }
    ),
    engineCommand(
        "perft",
        "count moves to given depth in current position",
        {
            "perft depth <n>",
            "e.g. perft depth 5"
        },
        {
            {"depth <n>", "positive integer depth in units of ply"}
        }
    ),
    engineCommand(
        "test",
        "test engine features via perft-like search",
        {
            "test (validation | incremental) depth <n>",
            "e.g. test validation depth 5"
        },
        {
            {"validation", "test if move legality check works"},
            {"incremental", "test incremental updates of game state"},
            {"depth <n>", "positive integer depth in units of ply"}
        }
    ),
    engineCommand(
        "gensfen",
        "generate quiet positions via fixed-depth self-play",
        {
            "gensfen depth <n> positions <n> randomply <n> maxply <n> evalbound <n>",
            "e.g. gensfen depth 12 positions 10000 randomply 4 maxply 200 evalbound 2500"
        },
        {
            {"depth <n>", "positive integer depth in units of ply"},
            {"positions <n>", "number of positions to generate"},
            {"randomply <n>", "number of random moves at start of each game"},
            {"maxply <n>", "maximum ply per game"},
            {"evalbound <n>", "maximum eval score in cp to save in file"}
        }
    ),
    engineCommand(
        "display",
        "display the position of the internal board",
        {},
        {}
    ),
    engineCommand(
        "quit",
        "quit the program as soon as possible",
        {},
        {}
    )
};

void displayHelp(const std::vector<std::string> &words)
{
    //display help.
    const std::string t = "    ";

    //overview.
    if (words.size() == 1)
    {
        int maxLength = 0;
        for (const auto &command: COMMANDS)
        {
            maxLength = std::max(maxLength,(int)command.name.length());
        }
        std::cout << "\nCommands:\n";
        for (const auto &command: COMMANDS)
        {
            std::cout << std::left << t << std::setw(maxLength+4)
                << command.name << command.desc
            << "\n";
        }
        std::cout << "\nType 'help <command>' for more information\n";
        std::cout << std::endl;
        return;
    }

    //verbose.
    for (const auto &command: COMMANDS)
    {
        if (words[1] != command.name) {continue;}
        std::cout << "\n" << command.desc << "\n";
        std::cout << "\nUsage: ";
        if (command.usage.size())
        {
            std::cout << "\n";
            for (const auto &use: command.usage)
            {
                std::cout << t << use << "\n";
            }
        }
        else {std::cout << command.name << "\n";}
        if (command.options.size())
        {
            int maxLength = 0;
            for (const auto &[option,desc]: command.options)
            {
                maxLength = std::max(maxLength,(int)option.length());
            }
            std::cout << "\nOptions:\n";
            for (const auto &[option,desc]: command.options)
            {
                std::cout << std::left << t << std::setw(maxLength+4)
                    << option << desc
                << "\n";
            }
        }
        std::cout << std::endl;
        return;
    }
}

#endif // ENGINE_COMMANDS_H_INCLUDED
