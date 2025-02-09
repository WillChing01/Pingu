#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include "constants.h"
#include "engine-commands.h"
#include "testing.h"
#include "format.h"
#include "search.h"
#include "board.h"

#include <chrono>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <thread>

const std::string ENGINE_NAME = "Pingu 5.0.0";
const std::string ENGINE_AUTHOR = "William Ching";
const std::string ENGINE_NAME_NO_SPACE = "Pingu_5.0.0";

std::atomic<bool> isSearching(false);

// define options.
const std::vector<std::string> optionsDescription = {
    "option name Hash type spin default 1 min 1 max 8192",
    "option name Clear Hash type button",
    "option name Threads type spin default 1 min 1 max 64",
};

void uciCommand() {
    // id engine.
    std::cout << "id name " << ENGINE_NAME << std::endl;
    std::cout << "id author " << ENGINE_AUTHOR << std::endl;

    // tell GUI which options can be changed.
    for (const std::string& option : optionsDescription) {
        std::cout << option << std::endl;
    }

    // confirmation command.
    std::cout << "uciok" << std::endl;
}

void setOptionCommand(Search& s, const std::vector<std::string>& words) {
    if (words[2] == "Hash") {
        resizeTT(std::stoi(words[4]));
    } else if (words[2] == "Clear" && words[3] == "Hash") {
        clearTT();
    } else if (words[2] == "Threads" && words[3] == "value" && isNumber(words[4])) {
        s.setThreads(std::stoi(words[4]));
    }
}

void positionCommand(Search& s, const std::vector<std::string>& words) {
    int ind = 3;
    if (words[1] == "startpos") {
        // start position.
        s.setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    } else if (words[1] == "fen") {
        s.setPositionFen(words[2] + " " + words[3] + " " + words[4] + " " + words[5] + " " + words[6] + " " + words[7]);
        ind = 9;
    }

    // play the specified moves.
    for (int i = ind; i < (int)words.size(); i++) {
        U32 chessMove = stringToMove(s.mainThread.b, words[i]);
        if (chessMove == 0) {
            break;
        }
        s.makeMove(chessMove);
    }
}

void perftCommand(Board& b, const std::vector<std::string>& words) {
    if (words.size() != 3) {
        return;
    }
    if (words[1] != "depth") {
        return;
    }
    if (!isNumber(words[2])) {
        return;
    }
    int depth = std::stoi(words[2]);
    U64 nodes = perft(b, depth);
    std::cout << "info nodes " << nodes << std::endl;
}

void testCommand(Board& b, const std::vector<std::string>& words) {
    if (words.size() != 4) {
        return;
    }
    if (words[2] != "depth") {
        return;
    }
    if (!isNumber(words[3])) {
        return;
    }
    int depth = std::stoi(words[3]);
    if (words[1] == "validation") {
        U32 cache[10][128] = {};
        bool res = testMoveValidation(b, depth, cache);
        std::cout << "info success " << res << std::endl;
    } else if (words[1] == "incremental") {
        bool res = incrementalTest(b, depth);
        std::cout << "info success " << res << std::endl;
    }
}

void evalCommand(Board& b, const std::vector<std::string>& words) {
    if (words.size() != 1) {
        return;
    }
    std::cout << "info score " << b.nnue.forward() << std::endl;
}

void seeCommand(Board& b, const std::vector<std::string>& words) {
    if (words.size() != 3) {
        return;
    }
    if (words[1] != "move") {
        return;
    }
    U32 chessMove = stringToMove(b, words[2]);
    if (!chessMove) {
        return;
    }
    std::cout << "info score " << b.see.evaluate(chessMove) << std::endl;
}

void goCommand(Search& s, const std::vector<std::string>& words) {
    isSearching = true;

    searchParams params;

    U32 whiteTime, whiteInc, blackTime, blackInc;
    whiteTime = whiteInc = blackTime = blackInc = 0;

    for (int i = 1; i < (int)words.size(); i++) {
        if (words[i] == "wtime") {
            whiteTime = std::stoi(words[i + 1]);
        } else if (words[i] == "btime") {
            blackTime = std::stoi(words[i + 1]);
        } else if (words[i] == "winc") {
            whiteInc = std::stoi(words[i + 1]);
        } else if (words[i] == "binc") {
            blackInc = std::stoi(words[i + 1]);
        } else if (words[i] == "movestogo") {
            params.movesToGo = std::stoi(words[i + 1]);
        } else if (words[i] == "depth") {
            params.depth = std::stoi(words[i + 1]);
        } else if (words[i] == "nodes") {
            params.nodes = std::stoi(words[i + 1]);
        } else if (words[i] == "mate") {
            // TODO
        } else if (words[i] == "movetime") {
            params.moveTime = std::stoi(words[i + 1]);
        }
    }

    if (s.mainThread.b.side) {
        params.time = blackTime;
        params.inc = blackInc;
        params.opponentTime = whiteTime;
        params.opponentInc = whiteInc;
    } else {
        params.time = whiteTime;
        params.inc = whiteInc;
        params.opponentTime = blackTime;
        params.opponentInc = blackInc;
    }

    U32 bestMove = s.go(params, false, true);
    std::cout << "bestmove " << moveToString(bestMove) << std::endl;
    isSearching = false;
}

void prepareForNewGame(Search& s) {
    // reset internal boards.
    s.prepareForNewGame();

    // reset hash table.
    clearTT();
    rootCounter = 0;
}

void uciLoop() {
    Search search;
    std::string input;
    std::vector<std::string> commands;

    std::cout << "id name " << ENGINE_NAME << std::endl;
    std::cout << "id author " << ENGINE_AUTHOR << std::endl;
    std::cout << "Type 'help' for a list of available commands" << std::endl;

    while (true) {
        std::getline(std::cin, input);
        commands = separateByWhiteSpace(input);

        if (isSearching) {
            if (input == "isready") {
                std::cout << "readyok" << std::endl;
            } else if (input == "stop") {
                search.terminateSearch();
                while (isSearching)
                    ;
            } else if (input == "quit") {
                search.terminateSearch();
                while (isSearching)
                    ;
                break;
            }
        } else {
            if (commands[0] == "uci") {
                uciCommand();
            } else if (commands[0] == "isready") {
                std::cout << "readyok" << std::endl;
            } else if (commands[0] == "setoption") {
                setOptionCommand(search, commands);
            } else if (commands[0] == "ucinewgame") {
                prepareForNewGame(search);
            } else if (commands[0] == "position") {
                positionCommand(search, commands);
            } else if (commands[0] == "go") {
                std::thread(goCommand, std::ref(search), std::ref(commands)).detach();
            } else if (commands[0] == "eval") {
                evalCommand(search.mainThread.b, commands);
            } else if (commands[0] == "see") {
                seeCommand(search.mainThread.b, commands);
            } else if (commands[0] == "perft") {
                perftCommand(search.mainThread.b, commands);
            } else if (commands[0] == "display") {
                search.mainThread.b.display();
            } else if (commands[0] == "test") {
                testCommand(search.mainThread.b, commands);
            } else if (commands[0] == "help") {
                displayHelpUCI(commands);
            } else if (commands[0] == "quit") {
                break;
            }
        }
    }
}

#endif // UCI_H_INCLUDED
