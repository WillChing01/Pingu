#ifndef TUNING_H_INCLUDED
#define TUNING_H_INCLUDED

#include <iostream>
#include <iomanip>
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
    759,
    339,
    286,
    263,
    100,
}};

std::array<int,6> piece_values_end =
{{
    20000,
    1694,
    901,
    542,
    517,
    169,
}};

std::array<std::array<int,64>,6> piece_tables_start =
{{
    {{
  0,  0,  0,  0,  0,  0,  0,-28,
  0,  0,  0,  0,  0,  0,  0, -2,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0, -6,
  0,  0,  0,  0,  0,  0,  0,  0,
 16, 12,  0,  0,  0,  0, 18, 20,
  0, 30,  7,  0,  0,  0, 30, 13,
    }},
    {{
 -1,  0,  0,  0,  0,  0,  0,  0,
 -8,  0,  0,  0,  0,  0,  0,  0,
 -8,  0,  0,  5,  5,  5,  0,  0,
 -3,  0,  0,  5,  5,  5,  0,  0,
  0,  0,  0,  5,  5,  5,  0,  0,
 -7,  4,  5,  0,  0,  0,  0,  0,
-10,  0,  0,  0,  0,  0,  0,-10,
 -8,-10,-10, -3, -5,-10,-10,-20,
    }},
    {{
  0,  0,  0,  0,  0,  0,  0,  0,
  5, 10, 10, 10, 10, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
 -5,  0,  0,  0,  0,  0,  0,  0,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
  0,  0,  0,  5,  0,  0,  0,  0,
    }},
    {{
-18,  0, -8, -8, -8, -8,  0,-11,
 -8,  0,  0,  0,  0,  0,  0, -5,
 -8,  0,  5, 10, 10,  5,  0,  0,
 -8,  5,  5, 10, 10,  5,  5,  0,
 -8,  0, 10, 10, 10, 10,  0,  0,
 -4,  3,  6, 10,  7,  0,  0,  0,
 -3,  0,  0,  0,  0,  0,  0,  0,
 -4,  0,-10,-10,-10,-10,-10,-20,
    }},
    {{
-48,-38,-28, -8,  0,-20,  0,-48,
-38,-18,  0,  0,  0,  0, -4,  0,
-28,  0, 10, 15, 15, 10,  0,  0,
-13,  5, 15, 20, 20, 15,  5,  0,
-16,  0,  8, 10, 15, 15,  0,  0,
-27,  0,  0,  8,  6,  0,  0,-14,
-40,-20,  0,  0,  0,  0,-20,-30,
-50,-28,-30,-30,-30,-30,-36,-50,
    }},
    {{
100,100,100,100,100,100,100,100,
 50, 50, 50, 50, 50, 50, 50, 50,
 10, 10, 20, 30, 30, 20, 10, 10,
  5,  5,  3, 16, 22, 10,  5,  2,
  0,  0,  0,  0,  1,  0,  0,  0,
  0, -1, -4,  0,  0, -7,  0,  0,
  0,  0,  0,-10,-10, 10,  9,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
    }},
}};

std::array<std::array<int,64>,6> piece_tables_end =
{{
    {{
-48, -5,  0,  0, -3, -8,  0,-48,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0, 20, 30, 30, 20,  0,  0,
  0,  0, 30, 40, 40, 30,  0,  0,
-19,  0, 30, 37, 37, 30,  0, -9,
-29, -5,  6, 15, 15, 11,  0,-18,
-24,-22,  0,  0,  0,  0, -5,-23,
-50,-21,-20,-30,-30,-30,-18,-50,
    }},
    {{
  0,  0,  0,  0,  0,  0,  0,  0,
 -8,  0,  0,  0,  0,  0,  0,  0,
 -8,  0,  5,  5,  5,  5,  0,  0,
 -3,  0,  0,  5,  5,  5,  0,  0,
  0,  0,  4,  5,  5,  5,  0,  0,
-10,  0,  4,  0,  0,  0,  0,  0,
-10,  0,  0,  0,  0,  0,  0,-10,
-17,-10,-10, -5, -5,-10,-10,-20,
    }},
    {{
  0,  0,  0,  0,  0,  0,  0,  0,
  5, 10, 10, 10, 10, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
 -3,  0,  0,  0,  0,  0,  0, -3,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
  0,  0,  0,  0,  0,  0,  0,  0,
    }},
    {{
 -1,  0,  0,  0,  0, -9, -4,-10,
-10,  0,  0,  0,  0,  0,  0, -8,
 -1,  0,  5, 10, 10,  5,  0,  0,
 -2,  5,  5, 10, 10,  5,  5,  0,
-10,  0, 10, 10, 10, 10,  0, -3,
 -8,  3, 10, 10, 10,  0,  0, -2,
-10,  0,  0,  0,  0,  0,  0,-10,
-20,-10,-10,-10,-10,-10,-10,-20,
    }},
    {{
-48,-16, -7, -3,  0,  0,-20,-48,
-40,-16,  0,  0,  0,  0,-16,-35,
-28,  0, 10, 15, 15, 10,  0,-12,
-20,  2, 15, 20, 20, 15,  5,  0,
-23,  0, 15, 20, 20, 15,  0, -5,
-30,  0,  0,  3,  2,  0,  0,-29,
-40,-20,  0,  0,  0,  0,-20,-40,
-50,-40,-30,-30,-30,-30,-40,-50,
    }},
    {{
100,100,100,100,100,100,100,100,
 50, 50, 50, 50, 50, 50, 50, 50,
 10, 10, 20, 30, 30, 20, 10, 10,
  5,  5, 10,  0,  0,  3,  5,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0, -6,  0,  0,  0,  0,  0,
  0,  0,  0, -5,  0, 10, 10,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
    }},
}};

struct dataSample
{
    std::vector<long double> featuresWhite;
    std::vector<long double> featuresBlack;
    long double res;
    long double materialDiffStart;
    long double materialDiffEnd;
    long double phaseStart;
    long double phaseEnd;
    long double eval;
};

int N = 0;
std::vector<std::pair<std::string,long double> > testPositions;
std::vector<dataSample> dataset;
std::array<int, 384> evalStart;
std::array<int, 384> evalEnd;

const long double K = 0.00475;

std::string formatWeight(int weight)
{
    std::string res = std::to_string(weight);
    while (res.length() < 3) {res.insert(0," ");}
    return res;
}

void saveWeights()
{
    std::ofstream file;
    file.open("../weights.txt", std::ios::trunc);

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
                file << formatWeight(evalStart[64 * i + 8 * j + k]) << ",";
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
                file << formatWeight(evalEnd[64 * i + 8 * j + k]) << ",";
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

void populateEval()
{
    for (int i=0;i<6;i++)
    {
        for (int j=0;j<64;j++)
        {
            evalStart[64 * i + j] = piece_tables_start[i][j];
            evalEnd[64 * i + j] = piece_tables_end[i][j];
        }
    }
}

void fenToFeatures(const std::string &fen, long double res)
{
    //append to features vector.
    dataset.push_back(dataSample());
    dataset.back().res = res;

    Board b;
    b.setPositionFen(fen);

    dataset.back().phaseStart = b.shiftedPhase / 256.;
    dataset.back().phaseEnd = (256. - b.shiftedPhase) / 256.;

    U64 temp; U64 x;

    for (int i=0;i<12;i++)
    {
        temp = b.pieces[i];
        while (temp)
        {
            x = popLSB(temp);
            if (i & 1)
            {
                //black.
                dataset.back().materialDiffStart -= piece_values_start[i >> 1];
                dataset.back().materialDiffEnd -= piece_values_end[i >> 1];
                dataset.back().featuresBlack.push_back((i >> 1) * 64 + x);
            }
            else
            {
                //white.
                dataset.back().materialDiffStart += piece_values_start[i >> 1];
                dataset.back().materialDiffEnd += piece_values_end[i >> 1];
                dataset.back().featuresWhite.push_back((i >> 1) * 64 + (x ^ 56));
            }
        }
    }
}

void populateDataset()
{
    for (int i=0;i<N;i++)
    {
        fenToFeatures(testPositions[i].first, testPositions[i].second);
    }
}

inline long double evalFeatures(dataSample &data)
{
    long double resStart = data.materialDiffStart;
    long double resEnd = data.materialDiffEnd;

    for (int i=0;i<(int)data.featuresWhite.size();i++)
    {
        resStart += evalStart[data.featuresWhite[i]];
        resEnd += evalEnd[data.featuresWhite[i]];
    }

    for (int i=0;i<(int)data.featuresBlack.size();i++)
    {
        resStart -= evalStart[data.featuresBlack[i]];
        resEnd -= evalEnd[data.featuresBlack[i]];
    }

    return resStart * data.phaseStart + resEnd * data.phaseEnd;
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

    return (((startTotal * b.shiftedPhase) + (endTotal * (256 - b.shiftedPhase))) / 256);
}

inline long double sigmoid(const long double x)
{
    return 1./(1. + exp(-K*x));
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
        score = customEval(b);
        E += powl(testPositions[i].second - sigmoid(score),2.);
    }
    return E / N;
}

inline long double error()
{
    int numThreads = std::thread::hardware_concurrency() / 2;
    std::vector<std::future<long double> > threads;

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
//     long double delta = 0.01;
//     bool direction = 0;
//     while (true)
//     {
//         std::cout << K << " " << oldError << std::endl;
//         if (!direction)
//         {
//             K += delta;
//             currentError = error();
//             if (currentError > oldError) {direction = 1; K -= delta;}
//             else {oldError = currentError;}
//         }
//         else if (direction)
//         {
//             K -= delta;
//             currentError = error();
//             if (currentError > oldError) {K += delta; delta /= 2.; if (delta < 0.0000001) {break;}}
//             else {oldError = currentError;}
//         }
//     }
//     std::cout << "Final K of " << K << std::endl;
// }

void optimiseFeatures(int epochs = 100, long double alpha = 0.00001)
{
    std::array<long double, 384> gradStart = {};
    std::array<long double, 384> gradEnd = {};
    
    long double err;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {

        //reset the gradient.
        for (int i=0;i<384;i++)
        {
            gradStart[i] = 0.;
            gradEnd[i] = 0.;
        }

        //update the gradient.
        for (int i=0;i<N;i++)
        {
            long double factor = (dataset[i].res - sigmoid(dataset[i].eval)) * sigmoid(dataset[i].eval);
            for (int j=0;j<(int)dataset[i].featuresWhite.size();j++)
            {
                gradStart[dataset[i].featuresWhite[j]] += factor * dataset[i].phaseStart;
                gradEnd[dataset[i].featuresWhite[j]] += factor * dataset[i].phaseEnd;
            }
            for (int j=0;j<(int)dataset[i].featuresBlack.size();j++)
            {
                gradStart[dataset[i].featuresBlack[j]] -= factor * dataset[i].phaseStart;
                gradEnd[dataset[i].featuresBlack[j]] -= factor * dataset[i].phaseEnd;
            }
        }

        //increment pst by gradient.
        for (int i=0;i<384;i++)
        {
            evalStart[i] += gradStart[i] * alpha;
            evalEnd[i] += gradEnd[i] * alpha;
        }

        //evaluate positions and error.
        err = 0.;
        for (int i=0;i<N;i++)
        {
            dataset[i].eval = evalFeatures(dataset[i]);
            err += powl(dataset[i].res - sigmoid(dataset[i].eval), 2.); 
        }
        err /= N;

        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << std::setprecision(10) << "Error: " << err << std::endl;

        //save weights.
        saveWeights();
    }
}

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
                    deltaStart[i] = std::max(deltaStart[i] / 2, 1);
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
                    deltaEnd[i] = std::max(deltaEnd[i] / 2, 1);
                }
            }
        }
        iter++;
        saveWeights();
    }
    std::cout << "Finished optimising PST!" << std::endl;
}

#endif // TUNING_H_INCLUDED
