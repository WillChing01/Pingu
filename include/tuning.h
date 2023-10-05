#ifndef TUNING_H_INCLUDED
#define TUNING_H_INCLUDED

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <math.h>
#include <cmath>
#include <thread>
#include <future>

#include "board.h"

const long double K = 0.00475;

static const int NUM_FEATURES = 393;
const int PST_INDEX = 0;
const int MAT_INDEX = 384;
const int MOB_INDEX = 390;
const int PAS_INDEX = 392;

std::array<long double,NUM_FEATURES> weights_start = {};
std::array<long double,NUM_FEATURES> weights_end = {};

struct dataSample
{
    //pst contains indices of non-zero features (sparse pop).
    std::vector<int> pstWhite;
    std::vector<int> pstBlack;
    //material and mobility feature vectors.
    std::array<long double,6> material;
    std::array<long double,2> mobility;
    long double passers;
    //game phase in that position.
    long double phase_start;
    long double phase_end;
    //our eval of position.
    long double eval;
    //actual result of game.
    long double R;
    std::string fen;
};

int N = 0;
std::vector<dataSample> dataset;

void populateFeatures(dataSample &sample)
{
    Board b;
    b.setPositionFen(sample.fen);

    //phase.
    sample.phase_start = b.shiftedPhase / 256.;
    sample.phase_end = (256. - b.shiftedPhase) / 256.;

    //passed pawns.
    sample.passers = b.countPassedPawns();

    //features.
    U64 temp; U64 x;
    for (int i=0;i<12;i++)
    {
        temp = b.pieces[i];
        while (temp)
        {
            x = popLSB(temp);
            if (i & 1)
            {
                sample.material[i >> 1] -= 1;
                sample.pstBlack.push_back((i >> 1) * 64 + x);
            }
            else
            {
                sample.material[i >> 1] += 1;
                sample.pstWhite.push_back((i >> 1) * 64 + (x ^ 56));
            }

            //bishop mobility.
            if (i >> 1 == b._nBishops >> 1)
            {
                int mob = magicBishopMob(b.occupied[0] | b.occupied[1], x);
                if (i & 1) {sample.mobility[0] -= mob;}
                else {sample.mobility[0] += mob;}
            }
            //rook mobility.
            if (i >> 1 == b._nRooks >> 1)
            {
                int mob = magicRookMob(b.occupied[0] | b.occupied[1], x);
                if (i & 1) {sample.mobility[1] -= mob;}
                else {sample.mobility[1] += mob;}
            }
        }
    }
}

void readPositions()
{
    //read all positions from a given file.
    //generate features for gradient descent.
    N = 0;
    std::string path = "../tuning/";
    std::cout << "Reading files in '" << path << "'" << std::endl;
    for (const auto &entry: std::filesystem::directory_iterator(path))
    {
        if (entry.path() == "../tuning/weights.txt") {continue;}
        std::cout << "Started reading " << entry.path() << std::endl;
        std::ifstream file(entry.path());
        std::string str;
        while (std::getline(file, str))
        {
            int x = str.find("[");
            dataset.push_back(dataSample());
            dataset.back().fen = str.substr(0, x-1);
            dataset.back().R = std::stold(str.substr(x+1, 3));
            populateFeatures(dataset.back());
            ++N;
        }
        std::cout << "Finished reading " << entry.path() << std::endl;
    }
    std::cout << "Processed " << N << " positions" << std::endl;
    std::cout << "Done!" << std::endl;
}

std::string formatWeight(int weight)
{
    std::string res = std::to_string(weight);
    while (res.length() < 3) {res.insert(0," ");}
    return res;
}

void readWeights()
{
    std::ifstream file("../tuning/weights.txt");
    std::string str;
    std::vector<std::string> data;
    while (std::getline(file, str))
    {
        std::stringstream line(str);
        std::string x;
        while (std::getline(line, x, ',')) {data.push_back(x);}
    }

    int i=0; bool start = false;
    for (const auto &x: data)
    {
        if (x == "Piece values start") {i = MAT_INDEX; start = true;}
        else if (x == "Piece values end") {i = MAT_INDEX; start = false;}
        else if (x == "Mobility start") {i = MOB_INDEX; start = true;}
        else if (x == "Mobility end") {i = MOB_INDEX; start = false;}
        else if (x == "Passers start") {i = PAS_INDEX; start = true;}
        else if (x == "Passers end") {i = PAS_INDEX; start = false;}
        else if (x == "Piece tables start") {i = PST_INDEX; start = true;}
        else if (x == "Piece tables end") {i = PST_INDEX; start = false;}
        else
        {
            if (start) {weights_start[i] = std::stold(x);}
            else {weights_end[i] = std::stold(x);}
            ++i;
        }
    }

    std::cout << "Weights read." << std::endl;
}

void saveWeights()
{
    std::ofstream file;
    file.open("../tuning/weights.txt", std::ios::trunc);

    //piece values start.
    file << "Piece values start" << std::endl;
    for (int i=0;i<6;i++) {file << std::lround(weights_start[MAT_INDEX + i]) << "," << std::endl;}
    file << std::endl;

    //piece values end.
    file << "Piece values end" << std::endl;
    for (int i=0;i<6;i++) {file << std::lround(weights_end[MAT_INDEX + i]) << "," << std::endl;}
    file << std::endl;

    //mobility start.
    file << "Mobility start" << std::endl;
    file << std::lround(weights_start[MOB_INDEX]) << "," << std::endl;
    file << std::lround(weights_start[MOB_INDEX + 1]) << "," << std::endl;
    file << std::endl;

    //mobility end.
    file << "Mobility end" << std::endl;
    file << std::lround(weights_end[MOB_INDEX]) << "," << std::endl;
    file << std::lround(weights_end[MOB_INDEX + 1]) << "," << std::endl;
    file << std::endl;

    //passers start.
    file << "Passers start" << std::endl;
    file << std::lround(weights_start[PAS_INDEX]) << "," << std::endl;
    file << std::endl;

    //passers end.
    file << "Passers end" << std::endl;
    file << std::lround(weights_end[PAS_INDEX]) << "," << std::endl;
    file << std::endl;

    //pst start.
    file << "Piece tables start" << std::endl;
    for (int i=0;i<6;i++)
    {
        for (int j=0;j<8;j++)
        {
            for (int k=0;k<8;k++)
            {
                file << formatWeight(std::lround(weights_start[PST_INDEX + 64 * i + 8 * j + k])) << ",";
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
                file << formatWeight(std::lround(weights_end[PST_INDEX + 64 * i + 8 * j + k])) << ",";
            } file << std::endl;
        } file << std::endl;
    }

    std::cout << "Weights saved." << std::endl;
}

void updateEval(dataSample &sample)
{
    //update the evaluation for this sample.
    
    long double evalStart = 0.;
    long double evalEnd = 0.;

    //material.
    for (int i=0;i<6;i++)
    {
        evalStart += sample.material[i] * weights_start[MAT_INDEX + i];
        evalEnd += sample.material[i] * weights_end[MAT_INDEX + i];
    }
    
    //mobility.
    for (int i=0;i<2;i++)
    {
        evalStart += sample.mobility[i] * weights_start[MOB_INDEX + i];
        evalEnd += sample.mobility[i] * weights_end[MOB_INDEX + i];
    }

    //passers.
    evalStart += sample.passers * weights_start[PAS_INDEX];
    evalEnd += sample.passers * weights_end[PAS_INDEX];

    //pst.
    for (const int x: sample.pstWhite)
    {
        evalStart += weights_start[PST_INDEX + x];
        evalEnd += weights_end[PST_INDEX + x];
    }
    for (const int x: sample.pstBlack)
    {
        evalStart -= weights_start[PST_INDEX + x];
        evalEnd -= weights_end[PST_INDEX + x];
    }

    sample.eval = evalStart * sample.phase_start + evalEnd * sample.phase_end;
}

inline long double sigmoid(const long double x)
{
    return 1./(1. + exp(-K*x));
}

long double getError()
{
    long double E = 0.;
    for (const auto& sample: dataset)
    {
        E += powl(sample.R - sigmoid(sample.eval), 2.);
    }
    return E / N;
}

void optimiseFeatures(int epochs = 100, long double alpha = 0.00001)
{
    std::array<long double, NUM_FEATURES> gradStart = {};
    std::array<long double, NUM_FEATURES> gradEnd = {};

    long double err;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        std::cout << "Starting epoch " << epoch << std::endl;

        //update eval.
        for (auto& x: dataset) {updateEval(x);}

        //update error.
        err = getError();
        std::cout << std::setprecision(10) << "Error: " << err << std::endl;

        //reset the gradient.
        gradStart.fill(0);
        gradEnd.fill(0);

        //update gradient.
        for (const auto& s: dataset)
        {
            long double factor = (s.R - sigmoid(s.eval)) * sigmoid(s.eval) * (1. - sigmoid(s.eval));
            
            //material.
            for (int i=0;i<6;i++)
            {
                gradStart[MAT_INDEX + i] += factor * s.phase_start * s.material[i];
                gradEnd[MAT_INDEX + i] += factor * s.phase_end * s.material[i];
            }

            //mobility.
            for (int i=0;i<2;i++)
            {
                gradStart[MOB_INDEX + i] += factor * s.phase_start * s.mobility[i];
                gradEnd[MOB_INDEX + i] += factor * s.phase_end * s.mobility[i];
            }

            //passers.
            gradStart[PAS_INDEX] += factor * s.phase_start * s.passers;
            gradEnd[PAS_INDEX] += factor * s.phase_end * s.passers;

            //pst.
            for (const auto& x: s.pstWhite)
            {
                gradStart[PST_INDEX + x] += factor * s.phase_start;
                gradEnd[PST_INDEX + x] += factor * s.phase_end;
            }
            for (const auto& x: s.pstBlack)
            {
                gradStart[PST_INDEX + x] -= factor * s.phase_start;
                gradEnd[PST_INDEX + x] -= factor * s.phase_end;
            }
        }

        //update weights.
        for (int i=0;i<NUM_FEATURES;i++)
        {
            weights_start[i] += alpha * gradStart[i];
            weights_end[i] += alpha * gradEnd[i];
        }

        //save weights.
        saveWeights();

        std::cout << "Finished epoch " << epoch << std::endl;
    }
}

// //calculate mean squared error for a subset of data (allows for threading).
// inline long double errorChild(int startInd, int finishInd)
// {
//     Board b;
//     int score;
//     long double E=0.;
//     for (int i=startInd;i<finishInd;i++)
//     {
//         b.setPositionFen(testPositions[i].first);
//         score = customEval(b);
//         E += powl(testPositions[i].second - sigmoid(score),2.);
//     }
//     return E / N;
// }

// inline long double error()
// {
//     int numThreads = std::thread::hardware_concurrency() / 2;
//     std::vector<std::future<long double> > threads;

//     int x = N/numThreads;
//     for (int i=0;i<numThreads;i++)
//     {
//         threads.push_back(
//             std::async(
//                 errorChild,
//                 i*x,
//                 i==(numThreads-1) ? N : (i+1)*x
//             )
//         );
//     }

//     bool allDone = false;
//     while (!allDone)
//     {
//         allDone = true;
//         for (int i=0;i<numThreads;i++)
//         {
//             if (!threads[i].valid()) {allDone = false; break;}
//         }
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }

//     long double E = 0.;
//     for (int i=0;i<numThreads;i++)
//     {
//         E += threads[i].get();
//     }

//     return E;
// }

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

#endif // TUNING_H_INCLUDED
