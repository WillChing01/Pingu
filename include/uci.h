#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <iostream>
#include <limits>
#include <thread>
#include <future>

#include "format.h"
#include "search.h"

std::atomic_bool isSearching(false);

//template<typename T>
//bool isReady(const std::future<T> &f)
//{
//    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
//}

void searchThread(Board &b, int depth, double moveTime)
{
    isSearchAborted = false; totalNodes = 0; timeLeft = moveTime;
    int res = alphaBetaRoot(b, -MATE_SCORE, MATE_SCORE, depth);

    //output best move after search is complete.
    cout << "info nodes " << totalNodes << " score cp " << res << endl;
    cout << "bestmove " << moveToString(storedBestMove) << endl;

    isSearching = false;
}

void uciCommand()
{
    //id engine.
    cout << "id name William's Engine" << endl;
    cout << "id author William Ching" << endl;

    //tell GUI which options can be changed.
    cout << "option name Hash type spin default 1 min 1 max 8192" << endl;

    //confirmation command.
    cout << "uciok" << endl;
}

void setOptionCommand(vector<string> words)
{
    if (words[2] == "Hash") {resizeTT(std::stoi(words[4]));}
    else if (words[2] == "Clear" && words[3] == "Hash") {clearTT();}
}

void positionCommand(Board &b, vector<string> words)
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

void goCommand(Board &b, vector<string> words)
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
        if (words[i] == "wtime") {whiteTime = stoi(words[i+1]);}
        else if (words[i] == "btime") {blackTime = stoi(words[i+1]);}
        else if (words[i] == "winc") {whiteInc = stoi(words[i+1]);}
        else if (words[i] == "binc") {blackInc = stoi(words[i+1]);}
        else if (words[i] == "movestogo") {movesToGo = stoi(words[i+1]);}
        else if (words[i] == "depth") {depth = stoi(words[i+1]);}
        else if (words[i] == "nodes") {}
        else if (words[i] == "mate") {}
        else if (words[i] == "movetime") {moveTime = stoi(words[i+1]);}
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

    //reset hash table.
    clearTT();
    rootCounter = 0;
}

void uciLoop()
{
    Board b;
    string input;
    vector<string> commands;

    while (true)
    {
        getline(cin,input);
        commands = separateByWhiteSpace(input);

        if (commands[0] == "uci") {uciCommand();}
        else if (commands[0] == "isready") {cout << "readyok" << endl;}
        else if (commands[0] == "setoption") {setOptionCommand(commands);}
        else if (commands[0] == "ucinewgame") {prepareForNewGame(b);}
        else if (commands[0] == "position") {positionCommand(b, commands);}
        else if (commands[0] == "go" && !isSearching) {goCommand(b, commands);}
        else if (commands[0] == "display") {b.display();}
        else if (commands[0] == "stop") {isSearchAborted = true;}
        else if (commands[0] == "quit") {isSearchAborted = true; break;}
    }
}

#endif // UCI_H_INCLUDED
