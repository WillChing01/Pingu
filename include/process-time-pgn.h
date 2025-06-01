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
    const std::regex clockRegex(R"(\[%clk (\d+):(\d+):(\d+)\])");
    const std::regex resultRegex(R"(1-0|1\/2-1\/2|0-1)");

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
