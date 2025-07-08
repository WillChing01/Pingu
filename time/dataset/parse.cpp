#include <filesystem>
#include <sstream>
#include "../../pipeline/dataparser.h"
#include "../../pipeline/utils.h"
#include "../include/datum.h"

inline Datum parseLine(const std::string& line) {
    Datum res;
    std::istringstream lineStream(line);
    std::string s;
    size_t i = 0;
    while (std::getline(lineStream, s, ',')) {
        switch (i) {
        case 0: {
            std::istringstream fenStream(s);
            std::string fenPart;
            size_t j = 0;
            while (j < 2 && std::getline(fenStream, fenPart, ' ')) {
                switch (j) {
                case 0:
                    parseFen(fenPart, res.pos);
                    break;
                case 1:
                    res.side = fenPart != "w";
                    break;
                }
                ++j;
            }
            break;
        }
        case 1:
            res.isDraw = s == "1";
            break;
        case 2:
            res.isWin = s == "1";
            break;
        case 3:
            res.ply = std::stoi(s);
            break;
        case 4:
            res.totalPly = std::stoi(s);
            break;
        case 5:
            res.qSearch = std::stoi(s);
            break;
        case 6:
            res.inCheck = s == "1";
            break;
        case 7:
            res.increment = std::stoi(s);
            break;
        case 8:
            res.timeLeft = std::stoi(s);
            break;
        case 9:
            res.timeSpent = std::stoi(s);
            break;
        case 10:
            res.totalTimeSpent = std::stoi(s);
            break;
        case 11:
            res.startTime = std::stoi(s);
            break;
        case 12:
            res.opponentTime = std::stoi(s);
            break;
        }
        ++i;
    }
    return res;
}

int main() {
    std::filesystem::path cwd = std::filesystem::current_path();
    DataParser<6, 25000000, 0.95, Datum, parseLine> dataParser(
        cwd / "_processed", ".csv", cwd / "training", cwd / "validation");
    dataParser.parseFiles();

    return 0;
}
