#ifndef PROCESS_TIME_PGN_H_INCLUDED
#define PROCESS_TIME_PGN_H_INCLUDED

#include <filesystem>
#include <vector>

#include "board.h"

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

    const std::regex moveRegex(R"([a-h][1-8][a-h][1-8][qrbn]?[+#]?)");
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

    void processFile(const std::filesystem::path& outputDir, const std::filesystem::path& inputPath) {
        const std::filesystem::path outputFile = outputDir / inputPath.filename();

        std::ifstream file(inputPath);
        std::string line;

        while (std::getline(file, line)) {
            if (line == "") continue;

            std::vector<std::smatch> moveMatches = extractMatches(line, moveRegex);
            std::vector<std::smatch> clockMatches = extractMatches(line, clockRegex);
            std::vector<std::smatch> resultMatches = extractMatches(line, resultRegex);

            if (moveMatches.size() != clockMatches.size() || !resultMatches.size()) continue;

            std::vector<Datum> res;

            for (size_t i = 0; i < moveMatches.size(); ++i) {
                std::string move = std::regex_replace(moveMatches[i][0].str(), checkRegex, "");
                int hours = std::stoi(clockMatches[i][1]);
                int minutes = std::stoi(clockMatches[i][2]);
                int seconds = std::stoi(clockMatches[i][3]);
                int clock = 3600 * hours + 60 * minutes + seconds;
                std::cout << move << " " << clock << std::endl;
            }
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
