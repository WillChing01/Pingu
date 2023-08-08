#ifndef TUNING_H_INCLUDED
#define TUNING_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <math.h>
#include <thread>
#include <future>

#include "evaluation.h"
#include "board.h"

std::array<int,6> piece_values_start =
{{
    20000,
    807,
    363,
    303,
    275,
    100,
}};

std::array<int,6> piece_values_end =
{{
    20000,
    1359,
    730,
    433,
    411,
    135,
}};

std::array<std::array<int,64>,6> piece_tables_start =
{{
    {{
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    }},
    {{
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    }},
    {{
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0,
    }},
    {{
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    }},
    {{
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    }},
    {{
       100,100,100,100,100,100,100,100,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    }},
}};

std::array<std::array<int,64>,6> piece_tables_end =
{{
    {{
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50,
    }},
    {{
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    }},
    {{
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0,
    }},
    {{
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    }},
    {{
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    }},
    {{
       100,100,100,100,100,100,100,100,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    }},
}};

int N = 0;
std::vector<std::pair<std::string,long double> > testPositions;

const long double K = 0.97;

std::string formatWeight(int weight)
{
    std::string res = std::to_string(weight);
    while (res.length() < 3) {res.insert(0," ");}
    return res;
}

void saveWeights()
{
    std::ofstream file;
    file.open("../weights.txt", ios::trunc);

    //piece values start.
    file << "Piece values start" << std::endl;
    for (int i=0;i<6;i++) {file << piece_values_start[i] << "," << std::endl;}
    file << std::endl;

    //piece values end.
    file << "Piece values end" << std::endl;
    for (int i=0;i<6;i++) {file << piece_values_end[i] << "," << std::endl;}
    file << std::endl;

    //piece tables start.
    file << "Piece tables start" << std::endl;
    for (int i=0;i<6;i++)
    {
        for (int j=0;j<8;j++)
        {
            for (int k=0;k<8;k++)
            {
                file << formatWeight(piece_tables_start[i][8*j+k]) << ",";
            } file << std::endl;
        } file << std::endl;
    }

    //piece tables end.
    file << "Piece tables end" << std::endl;
    for (int i=0;i<6;i++)
    {
        for (int j=0;j<8;j++)
        {
            for (int k=0;k<8;k++)
            {
                file << formatWeight(piece_tables_end[i][8*j+k]) << ",";
            } file << std::endl;
        } file << std::endl;
    }

    std::cout << "Weights saved." << std::endl;
}

void readPositions()
{
    std::ifstream file("../testing/E12.52-1M-D12-Resolved.book");
    std::string str;
    while (std::getline(file, str))
    {
        int x = str.find("[");
        testPositions.push_back(
            std::pair<std::string,long double>(
                str.substr(0,x-1),
                stold(str.substr(x+1,3))
            )
        );
    }
    N = testPositions.size();
    std::cout << "Finished reading " << N << " positions." << std::endl;
}

inline int customEval(Board &b)
{
    int startTotal = 0;
    int endTotal = 0;

    U64 x;

    //queens.
    U64 temp = b.pieces[b._nQueens];
    while (temp)
    {
        x = popLSB(temp);
        startTotal += piece_values_start[1] + piece_tables_start[1][x ^ 56];
        endTotal += piece_values_end[1] + piece_tables_end[1][x ^ 56];
    }

    temp = b.pieces[b._nQueens+1];
    while (temp)
    {
        x = popLSB(temp);
        startTotal -= piece_values_start[1] + piece_tables_start[1][x];
        endTotal -= piece_values_end[1] + piece_tables_end[1][x];
    }

    //rooks.
    temp = b.pieces[b._nRooks];
    while (temp)
    {
        x = popLSB(temp);
        startTotal += piece_values_start[2] + piece_tables_start[2][x ^ 56];
        endTotal += piece_values_end[2] + piece_tables_end[2][x ^ 56];
    }

    temp = b.pieces[b._nRooks+1];
    while (temp)
    {
        x = popLSB(temp);
        startTotal -= piece_values_start[2] + piece_tables_start[2][x];
        endTotal -= piece_values_end[2] + piece_tables_end[2][x];
    }

    //bishops.
    temp = b.pieces[b._nBishops];
    while (temp)
    {
        x = popLSB(temp);
        startTotal += piece_values_start[3] + piece_tables_start[3][x ^ 56];
        endTotal += piece_values_end[3] + piece_tables_end[3][x ^ 56];
    }

    temp = b.pieces[b._nBishops+1];
    while (temp)
    {
        x = popLSB(temp);
        startTotal -= piece_values_start[3] + piece_tables_start[3][x];
        endTotal -= piece_values_end[3] + piece_tables_end[3][x];
    }

    //knights.
    temp = b.pieces[b._nKnights];
    while (temp)
    {
        x = popLSB(temp);
        startTotal += piece_values_start[4] + piece_tables_start[4][x ^ 56];
        endTotal += piece_values_end[4] + piece_tables_end[4][x ^ 56];
    }

    temp = b.pieces[b._nKnights+1];
    while (temp)
    {
        x = popLSB(temp);
        startTotal -= piece_values_start[4] + piece_tables_start[4][x];
        endTotal -= piece_values_end[4] + piece_tables_end[4][x];
    }

    //pawns.
    temp = b.pieces[b._nPawns];
    while (temp)
    {
        x = popLSB(temp);
        startTotal += piece_values_start[5] + piece_tables_start[5][x ^ 56];
        endTotal += piece_values_end[5] + piece_tables_end[5][x ^ 56];
    }

    temp = b.pieces[b._nPawns+1];
    while (temp)
    {
        x = popLSB(temp);
        startTotal -= piece_values_start[5] + piece_tables_start[5][x];
        endTotal -= piece_values_end[5] + piece_tables_end[5][x];
    }

    //kings.
    int kingPos = __builtin_ctzll(b.pieces[b._nKing]);
    int kingPos2 = __builtin_ctzll(b.pieces[b._nKing+1]);

    startTotal += piece_tables_start[0][kingPos ^ 56] - piece_tables_start[0][kingPos2];
    endTotal += piece_tables_end[0][kingPos ^ 56] - piece_tables_end[0][kingPos2];

    return (((startTotal * b.shiftedPhase) + (endTotal * (256 - b.shiftedPhase))) / 256) * (1-2*(int)(b.moveHistory.size() & 1));
}

int qSearchTuning(Board &b, int alpha, int beta)
{
    bool inCheck = b.generatePseudoQMoves(b.moveHistory.size() & 1);

    if (b.moveBuffer.size() > 0)
    {
        int bestScore = -INT_MAX;

        if (!inCheck)
        {
            //do stand-pat check.
            bestScore=customEval(b);
            if (bestScore >= beta) {return bestScore;}
            alpha = max(alpha,bestScore);
        }

        int score;
        b.updateOccupied();
        vector<pair<U32,int> > moveCache = b.orderQMoves(-INT_MAX);
        for (int i=0;i<(int)(moveCache.size());i++)
        {
            b.makeMove(moveCache[i].first);
            score = -qSearchTuning(b, -beta, -alpha);
            b.unmakeMove();

            if (score > bestScore)
            {
                if (score >= beta) {return score;}
                bestScore = score;
                alpha = max(alpha,score);
            }
        }

        return bestScore;
    }
    else
    {
        //no captures left. evaluate normally.
        bool inCheck2 = b.generatePseudoMoves(b.moveHistory.size() & 1);
        
        if (b.moveBuffer.size() > 0) {return customEval(b);}
        else if (inCheck2) {return -MATE_SCORE;}
        else {return 0;}
    }
}

inline long double sigmoid(const long double x)
{
    return 1./(1. + powl(10.,-K*x/400.));
}

//calculate mean squared error for a subset of data (allows for threading).
inline long double errorChild(int startInd, int finishInd)
{
    Board b;
    int score;
    long double E=0.;
    for (int i=startInd;i<finishInd;i++)
    {
        b.setPositionFen(testPositions[i].first);
        score = qSearchTuning(b,-MATE_SCORE,MATE_SCORE);
        if (b.moveHistory.size() & 1) {score *= -1;}
        E += powl(testPositions[i].second - sigmoid(score),2.);
    }
    return E / N;
}

inline long double error()
{
    int numThreads = std::thread::hardware_concurrency() / 2;
    vector<std::future<long double> > threads;

    int x = N/numThreads;
    for (int i=0;i<numThreads;i++)
    {
        threads.push_back(
            std::async(
                errorChild,
                i*x,
                i==(numThreads-1) ? N : (i+1)*x
            )
        );
    }

    bool allDone = false;
    while (!allDone)
    {
        allDone = true;
        for (int i=0;i<numThreads;i++)
        {
            if (!threads[i].valid()) {allDone = false; break;}
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    long double E = 0.;
    for (int i=0;i<numThreads;i++)
    {
        E += threads[i].get();
    }

    return E;
}

// void optimiseK()
// {
//     long double oldError = error();
//     long double currentError;
//     bool direction = 0;
//     while (true)
//     {
//         std::cout << K << " " << oldError << std::endl;
//         if (!direction)
//         {
//             K += 0.01;
//             currentError = error();
//             if (currentError > oldError) {direction = 1; K -= 0.01;}
//             else {oldError = currentError;}
//         }
//         else if (direction)
//         {
//             K -= 0.01;
//             currentError = error();
//             if (currentError > oldError) {K += 0.01; break;}
//             else {oldError = currentError;}
//         }
//     }
//     std::cout << "Final K of " << K << std::endl;
// }

void optimiseMaterial(int delta = 1)
{
    //king material is fixed at 20000.
    //middlegame pawn is fixed at 100.
    std::cout << "Optimising material..." << std::endl;
    std::cout << "Resolution: " << delta << std::endl;

    long double oldError = error();
    long double currentError;

    int iter = 0;
    bool improved = true;
    while (improved)
    {
        std::cout << "Iteration " << iter << " error: " << oldError << std::endl;
        improved = false;
        for (int i=1;i<5;i++)
        {
            piece_values_start[i] += delta;
            currentError = error();
            if (currentError < oldError)
            {
                oldError = currentError; improved = true;
            }
            else
            {
                piece_values_start[i] -= 2 * delta;
                currentError = error();
                if (currentError < oldError)
                {
                    oldError = currentError; improved = true;
                }
                else {piece_values_start[i] += delta;}
            }
        }
        for (int i=1;i<6;i++)
        {
            piece_values_end[i] += delta;
            currentError = error();
            if (currentError < oldError)
            {
                oldError = currentError; improved = true;
            }
            else
            {
                piece_values_end[i] -= 2 * delta;
                currentError = error();
                if (currentError < oldError)
                {
                    oldError = currentError; improved = true;
                }
                else {piece_values_end[i] += delta;}
            }
        }
        iter++;
        saveWeights();
    }
    std::cout << "Finished optimising material!" << std::endl;
}

void optimisePST(int pieceType)
{
    std::array<int,64> deltaStart = {};
    std::array<int,64> deltaEnd = {};

    for (int i=0;i<64;i++)
    {
        deltaStart[i] = 8; deltaEnd[i] = 8;
    }

    std::cout << "Optimising PST" << std::endl;
    std::cout << "Piece type: " << pieceType << std::endl;

    long double oldError = error();
    long double currentError;

    int iter = 0;
    bool improved = true;
    while (improved)
    {
        std::cout << "Iteration " << iter << " error: " << oldError << std::endl;
        improved = false;
        for (int i=0;i<64;i++)
        {
            piece_tables_start[pieceType][i] += deltaStart[i];
            currentError = error();
            if (currentError < oldError)
            {
                oldError = currentError; improved = true;
            }
            else
            {
                piece_tables_start[pieceType][i] -= 2 * deltaStart[i];
                currentError = error();
                if (currentError < oldError)
                {
                    oldError = currentError; improved = true;
                }
                else
                {
                    piece_tables_start[pieceType][i] += deltaStart[i];
                    deltaStart[i] = max(deltaStart[i] / 2, 1);
                }
            }
        }
        for (int i=0;i<64;i++)
        {
            piece_tables_end[pieceType][i] += deltaEnd[i];
            currentError = error();
            if (currentError < oldError)
            {
                oldError = currentError; improved = true;
            }
            else
            {
                piece_tables_end[pieceType][i] -= 2 * deltaEnd[i];
                currentError = error();
                if (currentError < oldError)
                {
                    oldError = currentError; improved = true;
                }
                else
                {
                    piece_tables_end[pieceType][i] += deltaEnd[i];
                    deltaEnd[i] = max(deltaEnd[i] / 2, 1);
                }
            }
        }
        iter++;
        saveWeights();
    }
    std::cout << "Finished optimising PST!" << std::endl;
}

#endif // TUNING_H_INCLUDED
