#ifndef PROCESS_TIME_PGN_H_INCLUDED
#define PROCESS_TIME_PGN_H_INCLUDED

#include <filesystem>
#include <vector>

#include "board.h"
#include "format.h"

namespace processTime {
    struct Datum {
        std::string fen;
        bool isDraw;
        bool isWin; // from pov of player to move.
        int ply;
        int totalPly;
        int increment;
        int timeLeft;
        int timeSpent;
        int opponentTime;
    };

    const std::string csvHeader = "fen,isDraw,isWin,ply,totalPly,increment,timeLeft,timeSpent,opponentTime";

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
        std::ofstream outputFile(outputFilePath);
        for (const Datum& datum : data) {
            outputFile << datum.fen << "," << datum.isDraw << "," << datum.isWin << "," << datum.ply << ","
                       << datum.totalPly << "," << datum.increment << "," << datum.timeLeft << "," << datum.timeSpent
                       << "," << datum.opponentTime << std::endl;
        }
        outputFile.close();
    }

    inline std::string cleanMove(const std::string& move) {
        std::string s = std::regex_replace(move, checkRegex, "");
        s.back() = std::tolower(s.back()); // promotion.
        return s;
    }

    void processFile(const std::filesystem::path& outputDir, const std::filesystem::path& inputPath) {
        const std::filesystem::path outputFilePath = outputDir / inputPath.filename();

        addCsvHeaderToFile(outputFilePath);

        std::ifstream file(inputPath);
        std::string line;
        std::smatch timeControlMatch;
        int startingTime = 0;
        int increment = 0;

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

            Board b;

            bool isDraw = resultMatches[0].str() == "1/2-1/2";
            bool isWin[2] = {!isDraw && resultMatches[0].str() == "1-0", !isDraw && resultMatches[0].str() == "0-1"};
            int timeLeft[2] = {startingTime, startingTime};

            for (size_t i = 0; i < moveMatches.size(); ++i) {
                int hours = std::stoi(clockMatches[i][1]);
                int minutes = std::stoi(clockMatches[i][2]);
                int seconds = std::stoi(clockMatches[i][3]);
                int clock = 3600 * hours + 60 * minutes + seconds;
                res.push_back({.fen = positionToFen(b.pieces, b.current, b.side),
                               .isDraw = isDraw,
                               .isWin = isWin[i % 2],
                               .ply = (int)(i + 1),
                               .totalPly = (int)moveMatches.size(),
                               .increment = increment,
                               .timeLeft = timeLeft[i % 2],
                               .timeSpent = clock - increment - timeLeft[i % 2],
                               .opponentTime = timeLeft[(i + 1) % 2]});
                timeLeft[i % 2] = clock;

                std::string move = cleanMove(moveMatches[i][0].str());
                U32 encodedMove = stringToMove(b, move);
                if (!encodedMove) {
                    b.display();
                    std::cout << "Error! " << move << std::endl;
                    std::cout << moveMatches[i][0].str() << std::endl;
                }
                b.makeMove(encodedMove);
            }

            writeDataToFile(res, outputFilePath);
        }
    }

    void processTimePgnCommand(int argc, const char** argv) {
        if (!isValidInput(argc, argv)) return;

        for (int i = 5; i < argc; ++i) {
            processFile(argv[2], argv[i]);
        }
    }

} // namespace processTime

#endif // PROCESS_TIME_PGN_H_INCLUDED
