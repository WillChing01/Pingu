#ifndef PROCESS_TIME_PGN_H_INCLUDED
#define PROCESS_TIME_PGN_H_INCLUDED

#include <filesystem>
#include <vector>

#include "thread.h"
#include "format.h"

namespace processTime {
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

    const std::string csvHeader = "fen,isDraw,isWin,ply,totalPly,qSearch,inCheck,increment,timeLeft,timeSpent,"
                                  "totalTimeSpent,startTime,opponentTime";

    const std::regex timeControlRegex(R"_(\[TimeControl "(\d+)\+(\d+)"\])_");
    const std::regex moveRegex(R"([a-h][1-8][a-h][1-8][QRBNqrbn]?[+#]?)");
    const std::regex checkRegex(R"([+#])");
    const std::regex clockRegex(R"(\[%clk (\d+):(\d+):(\d+)\])");
    const std::regex resultRegex(R"(1-0|1\/2-1\/2|0-1)");

    std::vector<std::smatch> extractMatches(const std::string& s, const std::regex& r) {
        std::vector<std::smatch> res;
        auto begin = std::sregex_iterator(s.begin(), s.end(), r);
        auto end = std::sregex_iterator();
        for (auto i = begin; i != end; ++i) {
            res.push_back(*i);
        }
        return res;
    }

    bool isValidInput(int argc, const char** argv) {
        if (argc < 6) {
            std::cout << "Error - expected to find at least 6 arguments" << std::endl;
            return false;
        }

        if (std::strcmp(argv[2], "out") || std::strcmp(argv[4], "in")) {
            std::cout << "Error - invalid input format" << std::endl;
            return false;
        }

        if (!std::filesystem::exists(argv[3])) {
            std::cout << "Error - output directory does not exist" << std::endl;
            return false;
        }

        for (int i = 5; i < argc; ++i) {
            if (!std::filesystem::exists(argv[i])) {
                std::cout << "Error - the file '" << argv[i] << "' does not exist" << std::endl;
                return false;
            }
        }

        return true;
    }

    void addCsvHeaderToFile(const std::filesystem::path& outputFilePath) {
        std::ofstream outputFile(outputFilePath);
        outputFile << csvHeader << std::endl;
        outputFile.close();
    }

    void writeDataToFile(const std::vector<Datum>& data, const std::filesystem::path& outputFilePath) {
        std::ofstream outputFile(outputFilePath, std::ios::app);
        for (const Datum& datum : data) {
            outputFile << datum.fen << "," << datum.isDraw << "," << datum.isWin << "," << datum.ply << ","
                       << datum.totalPly << "," << datum.qSearch << "," << datum.inCheck << "," << datum.increment
                       << "," << datum.timeLeft << "," << datum.timeSpent << "," << datum.totalTimeSpent << ","
                       << datum.startTime << "," << datum.opponentTime << std::endl;
        }
        outputFile.close();
    }

    inline std::string cleanMove(const std::string& move) {
        std::string s = std::regex_replace(move, checkRegex, "");
        s.back() = std::tolower(s.back()); // promotion.
        return s;
    }

    void processFile(const std::filesystem::path& outputDir, const std::filesystem::path& inputPath) {
        const std::filesystem::path outputFilePath =
            std::filesystem::path(outputDir / inputPath.filename()).replace_extension(std::filesystem::path("csv"));

        addCsvHeaderToFile(outputFilePath);

        std::ifstream file(inputPath);
        std::string line;
        std::smatch timeControlMatch;
        int startingTime = 0;
        int increment = 0;
        int nGame = 0;
        int totalGames = 0;
        const int barWidth = 50;
        int percent = -1;
        int totalPositions = 0;

        while (std::getline(file, line)) {
            if (line == "") continue;
            if (std::regex_match(line, timeControlMatch, timeControlRegex)) {
                startingTime = std::stoi(timeControlMatch[1].str());
                increment = std::stoi(timeControlMatch[2].str());
                continue;
            }
            if (line[0] == '[') continue;
            ++totalGames;
        };

        file.clear();
        file.seekg(0, std::ios::beg);

        Thread t;
        globalNodeCount = 0;
        t.isSearchAborted = false;
        t.prepareSearch(MAXDEPTH, std::numeric_limits<double>::infinity(), false);

        while (std::getline(file, line)) {
            if (line == "") continue;
            if (std::regex_match(line, timeControlMatch, timeControlRegex)) {
                startingTime = std::stoi(timeControlMatch[1].str());
                increment = std::stoi(timeControlMatch[2].str());
                continue;
            }
            if (line[0] == '[') continue;

            std::vector<std::smatch> moveMatches = extractMatches(line, moveRegex);
            std::vector<std::smatch> clockMatches = extractMatches(line, clockRegex);
            std::vector<std::smatch> resultMatches = extractMatches(line, resultRegex);

            if (moveMatches.size() != clockMatches.size() || !resultMatches.size()) continue;

            std::vector<Datum> res;

            t.b.setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

            bool isDraw = resultMatches[0].str() == "1/2-1/2";
            bool isWin[2] = {!isDraw && resultMatches[0].str() == "1-0", !isDraw && resultMatches[0].str() == "0-1"};
            int timeLeft[2] = {startingTime, startingTime};
            int totalTimeSpent[2] = {0, 0};
            bool isError = false;

            for (size_t i = 0; i < moveMatches.size(); ++i) {
                int hours = std::stoi(clockMatches[i][1]);
                int minutes = std::stoi(clockMatches[i][2]);
                int seconds = std::stoi(clockMatches[i][3]);
                int clock = 3600 * hours + 60 * minutes + seconds;
                res.push_back({.fen = positionToFen(t.b.pieces, t.b.current, t.b.side),
                               .isDraw = isDraw,
                               .isWin = isWin[i % 2],
                               .ply = (int)(i + 1),
                               .totalPly = (int)moveMatches.size(),
                               .qSearch = t.qSearch(1, -MATE_SCORE, MATE_SCORE),
                               .inCheck = util::isInCheck(t.b.side, t.b.pieces, t.b.occupied),
                               .increment = increment,
                               .timeLeft = timeLeft[i % 2],
                               .timeSpent = timeLeft[i % 2] - (clock - increment),
                               .startTime = startingTime,
                               .opponentTime = timeLeft[(i + 1) % 2]});
                totalTimeSpent[i % 2] += timeLeft[i % 2] - (clock - increment);
                timeLeft[i % 2] = clock;

                std::string move = cleanMove(moveMatches[i][0].str());
                U32 encodedMove = stringToMove(t.b, move);
                if (!encodedMove) {
                    t.b.display();
                    std::cout << "Error! " << move << std::endl;
                    std::cout << moveMatches[i][0].str() << std::endl;
                    isError = true;
                    break;
                }
                t.b.makeMove(encodedMove);
            }

            if (!isError) {
                for (size_t i = 0; i < res.size(); ++i) {
                    res[i].totalTimeSpent = totalTimeSpent[i % 2];
                }
                writeDataToFile(res, outputFilePath);
                totalPositions += res.size();
            }

            double prog = double(++nGame) / double(totalGames);
            int p = 100 * prog;
            if (p > percent) {
                percent = p;
                std::cout << inputPath << " [";
                int bars = barWidth * prog;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < bars) {
                        std::cout << "=";
                    } else if (i == bars) {
                        std::cout << ">";
                    } else {
                        std::cout << " ";
                    }
                }
                std::cout << "] " << int(100 * prog) << "% - " << totalPositions << " positions" << "\r";
                std::cout.flush();
            }
        }
        std::cout << std::endl;
    }

    void processTimePgnCommand(int argc, const char** argv) {
        if (!isValidInput(argc, argv)) return;

        for (int i = 5; i < argc; ++i) {
            processFile(argv[3], argv[i]);
        }
    }

} // namespace processTime

#endif // PROCESS_TIME_PGN_H_INCLUDED
