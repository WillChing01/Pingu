#ifndef PROCESS_TIME_PGN_H_INCLUDED
#define PROCESS_TIME_PGN_H_INCLUDED

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>

#include "format.h"
#include "thread.h"

struct Datum {
    std::string fen;
    bool isDraw;
    bool isWin; // from pov of player to move.
    int ply;
    int totalPly;
    int qSearch;
    int inCheck;
    int increment;
    int timeLeft;
    int timeSpent;
    int totalTimeSpent;
    int startTime;
    int opponentTime;
};

const std::regex timeControlRegex(R"_(\[TimeControl "(\d+)\+(\d+)"\])_");
const std::regex moveRegex(R"([a-h][1-8][a-h][1-8][QRBNqrbn]?[+#]?)");
const std::regex checkRegex(R"([+#])");
const std::regex clockRegex(R"(\[%clk (\d+):(\d+):(\d+)\])");
const std::regex resultRegex(R"(1-0|1\/2-1\/2|0-1)");

#endif // PROCESS_TIME_PGN_H_INCLUDED
