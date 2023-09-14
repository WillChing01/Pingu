#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <iostream>
#include <cassert>
#include <limits>
#include <thread>
#include <future>

#include "constants.h"
#include "testing.h"
#include "format.h"
#include "search.h"
#include "board.h"

std::atomic_bool isSearching(false);

void searchThread(Board &b, int depth, double moveTime)
{
    isSearchAborted = false; totalNodes = 0; timeLeft = moveTime;
    int res = alphaBetaRoot(b, -MATE_SCORE, MATE_SCORE, depth);

    //output best move after search is complete.
    std::cout << "info nodes " << totalNodes << " score cp " << res << std::endl;
    std::cout << "bestmove " << moveToString(storedBestMove) << std::endl;

    isSearching = false;
}

void uciCommand()
{
    //id engine.
    std::cout << "id name Pingu 1.0.0" << std::endl;
    std::cout << "id author William Ching" << std::endl;

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
    try
    {
        int depth = std::stoi(words[1]);
        assert(depth >= 0);
        U64 nodes = perft(b, depth);
        std::cout << "info nodes " << nodes << std::endl;
    }
    catch (...) {}
}

void testCommand(Board &b, const std::vector<std::string> &words)
{
    try
    {
        assert(words.size() == 3);
        int depth = std::stoi(words[2]);
        assert(depth >= 0);
        assert(depth < 10);
        if (words[1] == "validation")
        {
            U32 cache[10][128] = {};
            bool res = testMoveValidation(b, depth, cache);
            std::cout << "info success " << res << std::endl;
        }
        else if (words[1] == "zobrist")
        {
            bool res = testZobristHashing(b, depth);
            std::cout << "info success " << res << std::endl;
        }
    }
    catch (...) {}
}

void goCommand(Board &b, const std::vector<std::string> &words)
{
    bool isInfinite = false;
    double whiteTime = 0;
    double blackTime = 0;
    double whiteInc = 0;
    double blackInc = 0;
    double moveTime = 0;
    // int movesToGo = 0;
    int depth = 0;

    for (int i=1;i<(int)words.size();i++)
    {
        if (words[i] == "wtime") {whiteTime = std::stoi(words[i+1]);}
        else if (words[i] == "btime") {blackTime = std::stoi(words[i+1]);}
        else if (words[i] == "winc") {whiteInc = std::stoi(words[i+1]);}
        else if (words[i] == "binc") {blackInc = std::stoi(words[i+1]);}
        else if (words[i] == "movestogo") {}
        else if (words[i] == "depth") {depth = std::stoi(words[i+1]);}
        else if (words[i] == "nodes") {}
        else if (words[i] == "mate") {}
        else if (words[i] == "movetime") {moveTime = std::stoi(words[i+1]);}
        else if (words[i] == "infinite") {isInfinite = true;}
    }

    if (isInfinite)
    {
        //infinite search.
        isSearching = true;
        auto calculation = std::thread(searchThread, std::ref(b), 100, std::numeric_limits<double>::infinity());
        calculation.detach();
    }
    else if (depth != 0)
    {
        //search to specified depth.
        auto calculation = std::thread(searchThread, std::ref(b), depth, std::numeric_limits<double>::infinity());
        calculation.detach();
    }
    else if (moveTime != 0)
    {
        //search for a specified time.
        auto calculation = std::thread(searchThread, std::ref(b), 100, moveTime);
        calculation.detach();
    }
    else
    {
        //allocate the time to search for.

        //current time management is quite basic.
        //use 3% of remaining time plus 75% of any increment.

        double totalTime = 0;
        if (b.moveHistory.size() & 1) {totalTime = 0.03 * blackTime + 0.75 * blackInc;}
        else {totalTime = 0.03 * whiteTime + 0.75 * whiteInc;}

        auto calculation = std::thread(searchThread, std::ref(b), 100, totalTime);
        calculation.detach();
    }
}

void prepareForNewGame(Board &b)
{
    b.stateHistory.clear();
    b.moveHistory.clear();
    b.hashHistory.clear();

    //reset hash table.
    clearTT();
    rootCounter = 0;

    //clear pawn hash table.
    for (int i=0;i<(int)(b.pawnHashMask + 1);i++) {b.pawnHash[i] = std::pair<U64,std::pair<int,int> >(0,std::pair<int,int>(0,0));}
}

void uciLoop()
{
    Board b;
    std::string input;
    std::vector<std::string> commands;

    while (true)
    {
        std::getline(std::cin,input);
        commands = separateByWhiteSpace(input);

        if (commands[0] == "uci") {uciCommand();}
        else if (commands[0] == "isready") {std::cout << "readyok" << std::endl;}
        else if (commands[0] == "setoption") {setOptionCommand(commands);}
        else if (commands[0] == "ucinewgame") {prepareForNewGame(b);}
        else if (commands[0] == "position") {positionCommand(b, commands);}
        else if (commands[0] == "go" && !isSearching) {goCommand(b, commands);}
        else if (commands[0] == "perft") {perftCommand(b, commands);}
        else if (commands[0] == "display") {b.display();}
        else if (commands[0] == "test") {testCommand(b, commands);}
        else if (commands[0] == "stop") {isSearchAborted = true;}
        else if (commands[0] == "quit") {isSearchAborted = true; break;}
    }
}

#endif // UCI_H_INCLUDED
