#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <iostream>
#include <iomanip>
#include <chrono>
#include <format>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <thread>

#include "constants.h"
#include "engine-commands.h"
#include "testing.h"
#include "format.h"
#include "search.h"
#include "board.h"
#include "bench.h"
#include "gensfen.h"

const std::string ENGINE_NAME = "Pingu 3.0.0";
const std::string ENGINE_AUTHOR = "William Ching";
const std::string ENGINE_NAME_NO_SPACE = "Pingu_3.0.0";

std::atomic_bool isSearching(false);

void searchThread(Board &b, int depth, double moveTime)
{
    isSearchAborted = false; timeLeft = moveTime;
    alphaBetaRoot(b, depth);

    //output best move after search is complete.
    std::cout << "bestmove " << moveToString(storedBestMove) << std::endl;

    isSearching = false;
}

void uciCommand()
{
    //id engine.
    std::cout << "id name " << ENGINE_NAME << std::endl;
    std::cout << "id author " << ENGINE_AUTHOR << std::endl;

    //tell GUI which options can be changed.
    std::cout << "option name Hash type spin default 1 min 1 max 8192" << std::endl;

    //confirmation command.
    std::cout << "uciok" << std::endl;
}

void setOptionCommand(const std::vector<std::string> &words)
{
    if (words[2] == "Hash") {resizeTT(std::stoi(words[4]));}
    else if (words[2] == "Clear" && words[3] == "Hash") {clearTT();}
}

void positionCommand(Board &b, const std::vector<std::string> &words)
{
    int ind = 3;
    if (words[1] == "startpos")
    {
        //start position.
        b.setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    else if (words[1] == "fen")
    {
        b.setPositionFen(words[2]+" "+words[3]+" "+words[4]+" "+words[5]+" "+words[6]+" "+words[7]);
        ind = 9;
    }

    //play the specified moves.
    for (int i=ind;i<(int)words.size();i++)
    {
        U32 chessMove = stringToMove(b,words[i]);
        if (chessMove == 0) {break;}
        b.makeMove(chessMove);
    }
}

void perftCommand(Board &b, const std::vector<std::string> &words)
{
    if (words.size() != 3) {return;}
    if (words[1] != "depth") {return;}
    if (!isNumber(words[2])) {return;}
    int depth = std::stoi(words[2]);
    U64 nodes = perft(b, depth);
    std::cout << "info nodes " << nodes << std::endl;
}

void testCommand(Board &b, const std::vector<std::string> &words)
{
    if (words.size() != 4) {return;}
    if (words[2] != "depth") {return;}
    if (!isNumber(words[3])) {return;}
    int depth = std::stoi(words[3]);
    if (words[1] == "validation")
    {
        U32 cache[10][128] = {};
        bool res = testMoveValidation(b, depth, cache);
        std::cout << "info success " << res << std::endl;
    }
    else if (words[1] == "incremental")
    {
        bool res = incrementalTest(b, depth);
        std::cout << "info success " << res << std::endl;
    }
}

void evalCommand(Board &b, const std::vector<std::string> &words)
{
    if (words.size() != 1) {return;}
    std::cout << "info score " << b.nnue.forward() << std::endl;
}

void seeCommand(Board &b, const std::vector<std::string> &words)
{
    if (words.size() != 3) {return;}
    if (words[1] != "move") {return;}
    U32 chessMove = stringToMove(b, words[2]);
    if (!chessMove) {return;}
    std::cout << "info score " << b.see.evaluate(chessMove) << std::endl;
}

void helpCommand(const std::vector<std::string> &words)
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

void goCommand(Board &b, const std::vector<std::string> &words)
{
    bool isInfinite = false;
    double whiteTime = 0;
    double blackTime = 0;
    double whiteInc = 0;
    double blackInc = 0;
    double moveTime = 0;
    int movesToGo = 0;
    int depth = 0;

    for (int i=1;i<(int)words.size();i++)
    {
        if (words[i] == "wtime") {whiteTime = std::stoi(words[i+1]);}
        else if (words[i] == "btime") {blackTime = std::stoi(words[i+1]);}
        else if (words[i] == "winc") {whiteInc = std::stoi(words[i+1]);}
        else if (words[i] == "binc") {blackInc = std::stoi(words[i+1]);}
        else if (words[i] == "movestogo") {movesToGo = std::stoi(words[i+1]);}
        else if (words[i] == "depth") {depth = std::stoi(words[i+1]);}
        else if (words[i] == "nodes") {}
        else if (words[i] == "mate") {}
        else if (words[i] == "movetime") {moveTime = std::stoi(words[i+1]);}
        else if (words[i] == "infinite") {isInfinite = true;}
    }

    if (isInfinite)
    {
        //infinite search.
        depth = 100;
        moveTime = std::numeric_limits<double>::infinity();
    }
    else if (depth > 0)
    {
        //search to specified depth.
        moveTime = std::numeric_limits<double>::infinity();
    }
    else if (moveTime > 0)
    {
        //search for a specified time.
        depth = 100;
    }
    else
    {
        //allocate the time to search for.

        //use 5% of remaining time plus 50% of increment.
        //if this exceeds time left, then use 80% of time left.

        depth = 100;

        double timeLeft = b.side ? blackTime : whiteTime;
        double increment = b.side ? blackInc : whiteInc;

        moveTime = timeLeft / std::max(movesToGo, 20) + 0.5 * increment;
        if (moveTime > timeLeft) {moveTime = 0.8 * timeLeft;}
    }

    isSearching = true;
    auto calculation = std::thread(searchThread, std::ref(b), depth, moveTime);
    calculation.detach();
}

void prepareForNewGame(Board &b)
{
    b.stateHistory.clear();
    b.moveHistory.clear();
    b.hashHistory.clear();
    b.irrevMoveInd.clear();

    //reset hash table.
    clearTT();
    rootCounter = 0;

    //clear history.
    b.history.clear();
}

void gensfenCommand(Board &b, const std::vector<std::string> &words)
{
    //generate self-play data.
    if (words.size() != 11) {return;}
    if (words[1] != "depth" || words[3] != "positions" ||
        words[5] != "randomply" || words[7] != "maxply" ||
        words[9] != "evalbound")
    {
        return;
    }
    if (!isNumber(words[2]) || !isNumber(words[4]) ||
        !isNumber(words[6]) || !isNumber(words[8]) ||
        !isNumber(words[10]))
    {
        return;
    }

    std::string dateTime = std::format("{:%F_%H-%M-%S_%Z}",
        std::chrono::zoned_time{
            std::chrono::current_zone(),
            std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now())
        }
    );

    int depth = std::stoi(words[2]);
    int positions = std::stoi(words[4]);
    int randomply = std::stoi(words[6]);
    int maxply = std::stoi(words[8]);
    int evalbound = std::stoi(words[10]);

    std::vector<gensfenData> output = {};
    std::vector<gensfenData> outputBuffer = {};

    //set up random device.
    std::random_device _rd;
    std::size_t seed;

    if (_rd.entropy()) {seed = _rd();}
    else {seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();}

    std::mt19937 _mt(seed);

    //set up search params.
    timeLeft = INT_MAX;
    isSearchAborted = false;

    int numGames = 1;

    std::cout << "Generating " << positions << " positions..." << std::endl;

    while ((int)output.size() < positions)
    {
        //start a new game.
        bool gameover = false;
        double result = 0.5;
        prepareForNewGame(b);
        b.setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        outputBuffer.clear();

        //track zHash for draw by repetition.
        std::set<U64> zHash;

        //play random moves up to randomply.
        for (int i=0;i<randomply;i++)
        {
            b.moveBuffer.clear();
            b.generatePseudoMoves();

            //if no moves left we abort the game.
            if (b.moveBuffer.size() == 0) {gameover = true; break;}

            std::uniform_int_distribution<int> _dist(0, (int)b.moveBuffer.size() - 1);
            U32 move = b.moveBuffer[_dist(_mt)];
            b.makeMove(move);

            //store zHash.
            zHash.insert(b.zHashPieces ^ b.zHashState);
        }

        if (gameover) {continue;}

        //fixed-depth search.
        while (true)
        {
            int score = alphaBetaRoot(b, depth, true);

            //check that score within bounds.
            if (b.side) {score *= -1;}
            if (abs(score) > evalbound || isGameOver)
            {
                result = score == 0 ? 0.5 :
                         score > evalbound ? 1. : 0.;
                break;
            }

            //exit if maxply exceeded.
            if ((int)b.moveHistory.size() > maxply)
            {
                result = score > 400 ? 1. :
                         score < -400 ? 0. : 0.5;
                break;
            }

            //update output buffer.
            U32 pieceType = (storedBestMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 capturedPieceType = (storedBestMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
            U32 finishPieceType = (storedBestMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
            bool isQuiet = !util::isInCheck(b.side, b.pieces, b.occupied) && capturedPieceType == 15 && pieceType == finishPieceType;
            if (isQuiet) {outputBuffer.push_back(gensfenData(positionToFen(b.pieces, b.current, b.side), score, 0.5));}

            b.makeMove(storedBestMove);

            //check if position is repeated and remove last output if so.
            if (zHash.find(b.zHashPieces ^ b.zHashState) != zHash.end())
            {
                if (isQuiet) {outputBuffer.pop_back();}
                result = 0.5;
                break;
            }
            zHash.insert(b.zHashPieces ^ b.zHashState);
        }

        //update contents of buffer with game result.
        for (auto &x: outputBuffer) {x.result = result;}

        //add contents of buffer to output.
        for (const auto &x: outputBuffer) {output.push_back(x);}

        std::cout << "Finished game " << numGames << "; " << output.size() << " positions overall" << std::endl;
        numGames++;
    }

    //trim size of output.
    while ((int)output.size() > positions) {output.pop_back();}

    //write output to file.
    std::string fileName = "gensfen_"+ ENGINE_NAME_NO_SPACE +
                           "_n" + std::to_string(positions) +
                           "_d" + std::to_string(depth) +
                           "_r" + std::to_string(randomply) +
                           "_m" + std::to_string(maxply) +
                           "_b" + std::to_string(evalbound) +
                           "_" + dateTime +
                           ".txt";

    std::ofstream file;
    file.open(fileName);

    for (const auto &x: output)
    {
        file << x.fen << "; " << x.eval << "; " << x.result << std::endl;
    }

    file.close();

    std::cout << "Finished generating positions." << std::endl;

    return;
}

void uciLoop()
{
    Board b;
    std::string input;
    std::vector<std::string> commands;

    std::cout << "id name " << ENGINE_NAME << std::endl;
    std::cout << "id author " << ENGINE_AUTHOR << std::endl;

    while (true)
    {
        std::getline(std::cin,input);
        commands = separateByWhiteSpace(input);

        if (commands[0] == "stop") {isSearchAborted = true;}
        else if (commands[0] == "quit") {isSearchAborted = true; while (isSearching); break;}
        
        if (isSearching) {continue;}

        if (commands[0] == "uci") {uciCommand();}
        else if (commands[0] == "isready") {std::cout << "readyok" << std::endl;}
        else if (commands[0] == "setoption") {setOptionCommand(commands);}
        else if (commands[0] == "ucinewgame") {prepareForNewGame(b);}
        else if (commands[0] == "position") {positionCommand(b, commands);}
        else if (commands[0] == "go") {goCommand(b, commands);}
        else if (commands[0] == "eval") {evalCommand(b, commands);}
        else if (commands[0] == "see") {seeCommand(b, commands);}
        else if (commands[0] == "perft") {perftCommand(b, commands);}
        else if (commands[0] == "display") {b.display(); std::cout << positionToFen(b.pieces, b.current, b.side) << std::endl;}
        else if (commands[0] == "gensfen") {gensfenCommand(b, commands);}
        else if (commands[0] == "test") {testCommand(b, commands);}
        else if (commands[0] == "help") {helpCommand(commands);}
    }
}

#endif // UCI_H_INCLUDED
