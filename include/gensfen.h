#ifndef GENSFEN_H_INCLUDED
#define GENSFEN_H_INCLUDED

#include <cstring>
#include <filesystem>
#include <unordered_map>

#include "uci.h"

struct gensfenData
{
    std::string fen;
    int eval; //from white pov.
    int result; //0 if black win, 1 if draw, 2 if white win.
};

bool isValidInput(int argc, const char** argv)
{
    if (argc != 16)
    {
        std::cout << "Error - expected to find exactly 7 arguments" << std::endl;
        return false;
    }

    const std::array<std::pair<const char*, const char*>, 7> requiredArgs = {{
        {"mindepth", "int"},
        {"maxdepth", "int"},
        {"positions", "int"},
        {"randomply", "int"},
        {"maxply", "int"},
        {"evalbound", "int"},
        {"book", "file"},
    }};

    for (int i=0;i<(int)requiredArgs.size();++i)
    {
        //check field name.
        if (std::strcmp(requiredArgs[i].first, argv[2+2*i]))
        {
            std::cout << "Error - expected to find argument '" << requiredArgs[i].first << "'" << std::endl;
            return false;
        }

        //check field type.
        if (!std::strcmp(requiredArgs[i].second, "int"))
        {
            if (!isNumber(argv[3+2*i]))
            {
                std::cout << "Error - expected value of '" << requiredArgs[i].first << "' to be an integer" << std::endl;
                return false;
            }
        }
        if (!std::strcmp(requiredArgs[i].second, "file"))
        {
            //check if file exists.
            if (std::strcmp(argv[3+2*i], "None") && !std::filesystem::exists(argv[3+2*i]))
            {
                std::cout << "Error - the file '" << argv[3+2*i] << "' does not exist the current directory" << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool playOpening(Search &s, int randomPly, const std::string &bookFile)
{
    std::random_device _rd;
    std::size_t seed;

    if (_rd.entropy()) {seed = _rd();}
    else {seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();}
    std::mt19937_64 _mt(seed);

    //get random position from book.
    //todo.

    //random playout.
    for (int i=0;i<randomPly;++i)
    {
        s.mainThread.b.generatePseudoMoves();

        //check for game over.
        if (!s.mainThread.b.moveBuffer.size()) {return false;}
        if (s.mainThread.isDrawByMaterial()) {return false;}
        if (s.mainThread.isDrawByRepetition()) {return false;}
        if (s.mainThread.isDrawByFifty()) {return false;}

        std::uniform_int_distribution<int> _dist(0, s.mainThread.b.moveBuffer.size() - 1);
        U32 move = s.mainThread.b.moveBuffer[_dist(_mt)];
        s.makeMove(move);
    }
    return true;
}

void gensfenCommand(int argc, const char** argv)
{
    if (!isValidInput(argc, argv)) {return;}

    std::string dateTime = std::format("{:%F_%H-%M-%S_%Z}",
        std::chrono::zoned_time{
            std::chrono::current_zone(),
            std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now())
        }
    );

    int mindepth = std::stoi(argv[3]);
    int maxdepth = std::stoi(argv[5]);
    int positions = std::stoi(argv[7]);
    int randomply = std::stoi(argv[9]);
    int maxply = std::stoi(argv[11]);
    int evalbound = std::stoi(argv[13]);
    std::string bookFile = argv[15];

    std::vector<gensfenData> output = {};
    std::vector<gensfenData> outputBuffer = {};
    int numGames = 1;
    Search s;

    //open the book and store its contents.

    std::cout << "Generating " << positions << " positions..." << std::endl;

    while ((int)output.size() < positions)
    {
        //start a new game.
        int result = 1;
        prepareForNewGame(s);
        s.setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        outputBuffer.clear();

        //random playout.
        if (!playOpening(s, randomply, bookFile)) {continue;}

        //fixed-depth search.
        while (true)
        {
            //check for game over.
            bool inCheck = s.mainThread.b.generatePseudoMoves();
            if (!s.mainThread.b.moveBuffer.size()) {result = inCheck ? (s.mainThread.b.side ? 2 : 0) : 1; break;}
            if (s.mainThread.isDrawByMaterial()) {result = 1; break;}
            if (s.mainThread.isDrawByRepetition()) {result = 1; break;}
            if (s.mainThread.isDrawByFifty()) {result = 1; break;}

            //scale depth based on game phase.
            int searchDepth = mindepth + ((24 - s.mainThread.b.phase) * (maxdepth - mindepth)) / 24;
            U32 bestMove = s.go(searchDepth, INT_MAX, true, false);
            int score = s.mainThread.bestScore;
            if (s.mainThread.b.side) {score *= -1;}

            //record move if quiet, below maxply, and within eval bound.
            U32 pieceType = (bestMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 capturedPieceType = (bestMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
            U32 finishPieceType = (bestMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
            bool isQuiet = !inCheck && capturedPieceType == 15 && pieceType == finishPieceType;
            bool isBelowMaxPly = (int)s.mainThread.b.moveHistory.size() < maxply;
            bool isWithinEvalBound = std::abs(score) < evalbound;

            if (isQuiet && isBelowMaxPly && isWithinEvalBound)
            {
                std::string fen = positionToFen(s.mainThread.b.pieces, s.mainThread.b.current, s.mainThread.b.side);
                outputBuffer.push_back(gensfenData(fen, score, 1));
            }

            s.mainThread.b.makeMove(bestMove);
        }

        //update contents of buffer with game result.
        for (auto &x: outputBuffer) {x.result = result;}

        //add contents of buffer to output.
        for (const auto &x: outputBuffer) {output.push_back(x);}

        std::cout << "Finished game " << numGames++ << "; " << output.size() << " positions overall" << std::endl;
    }

    //write output to file.
    std::string fileName = "gensfen_"+ ENGINE_NAME_NO_SPACE +
                           "_n" + std::to_string(output.size()) +
                           "_d" + std::to_string(mindepth) +
                           "_r" + std::to_string(randomply) +
                           "_m" + std::to_string(maxply) +
                           "_b" + std::to_string(evalbound) +
                           "_" + dateTime +
                           ".txt";

    std::ofstream file;
    file.open(fileName);

    const std::unordered_map<int, std::string> resultMapping = {
        {0, "0"},
        {1, "0.5"},
        {2, "1"},
    };

    for (const gensfenData &x: output)
    {
        file << x.fen << "; " << x.eval << "; " << resultMapping.at(x.result) << std::endl;
    }

    file.close();

    std::cout << "Finished generating positions." << std::endl;

    return;
}

#endif // GENSFEN_H_INCLUDED
