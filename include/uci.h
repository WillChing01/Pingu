#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <iostream>
#include <thread>

#include "format.h"
#include "search.h"

vector<string> separateByWhiteSpace(string input)
{
    //assume only a single whitespace separates each word.
    vector<string> words; words.push_back("");
    for (int i=0;i<(int)input.length();i++)
    {
        if (input[i] == ' ') {words.push_back("");}
        else {words.back() += input[i];}
    }
    while (words.back() == "") {words.pop_back();}
    return words;
}

void uciCommand()
{
    cout << "id name William's Engine" << endl;
    cout << "id author William Ching" << endl;
    cout << "uciok" << endl;
}

void setoptionCommand()
{
    return;
}

void registerCommand()
{
    return;
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
        b.makeMove(chessMove);
    }
}

void goCommand(Board &b, vector<string> words)
{
    //start calculating.
//    timeLeft = 5000;
    timeLeft = INT_MAX;
    isSearchAborted = false;
    auto calculation = std::thread(alphaBetaRoot, std::ref(b), -INT_MAX, INT_MAX, 64);
    calculation.detach();
//    cout << "bestmove " << moveToString(storedBestMove) << endl;
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
        else if (commands[0] == "position") {positionCommand(b, commands);}
        else if (commands[0] == "go") {goCommand(b, commands);}
        else if (commands[0] == "display") {b.display();}
        else if (commands[0] == "stop")
        {
            isSearchAborted = true;
            cout << moveToString(storedBestMove) << endl;
        }
        else if (commands[0] == "quit") {isSearchAborted = true; break;}
    }
}

#endif // UCI_H_INCLUDED
