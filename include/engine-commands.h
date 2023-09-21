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
            {"moves <move1> ... <movei>", "moves to be played from the given position"}
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
        "quit",
        "quit the program as soon as possible",
        {},
        {}
    ),
    engineCommand(
        "perft",
        "count moves to given depth in current position",
        {
            "perft <depth>",
            "e.g. perft 5"
        },
        {
            {"<depth>", "positive integer depth in ply"}
        }
    ),
    engineCommand(
        "test",
        "test engine features via perft-like search",
        {
            "test (validation | zobrist) <depth>",
            "e.g. test validation 5"
        },
        {
            {"validation", "test if move legality check works"},
            {"zobrist", "test if zobrist hashing works"},
            {"<depth>", "positive integer depth in units of ply"}
        }
    ),
    engineCommand(
        "display",
        "display the position of the internal board",
        {},
        {}
    )
};

#endif // ENGINE_COMMANDS_H_INCLUDED
