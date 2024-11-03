#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "utils.h"

const double TRAINING_RATIO = 0.95;

inline datum parseLine(const std::string &line)
{
    datum res;

    std::istringstream f(line);
    std::string s;
    size_t i = 0;
    while (std::getline(f, s, ';'))
    {
        switch (i)
        {
            case 0:
            {
                std::istringstream g(s);
                std::string t;
                size_t j = 0;
                while (j < 2 && std::getline(g, t, ' '))
                {
                    switch (j)
                    {
                        case 0:
                        {
                            unsigned char square = 56;
                            for (const unsigned char x: t)
                            {
                                switch (x)
                                {
                                    case '/':
                                        square -= 16;
                                        break;
                                    case '1':
                                        ++square;
                                        break;
                                    case '2':
                                        square += 2;
                                        break;
                                    case '3':
                                        square += 3;
                                        break;
                                    case '4':
                                        square += 4;
                                        break;
                                    case '5':
                                        square += 5;
                                        break;
                                    case '6':
                                        square += 6;
                                        break;
                                    case '7':
                                        square += 7;
                                        break;
                                    case '8':
                                        square += 8;
                                        break;
                                    default:
                                        res.pos[square >> 4] &= (U64)pieceTypeMap.at(x) << (U64)(square & 15);
                                        if (x == 'K') {res.kingPos[0] = square;}
                                        else if (x == 'k') {res.kingPos[1] = square;}
                                        ++square;
                                        break;
                                }
                            }
                            break;
                        }
                        case 1:
                            res.side = t != "w";
                            break;
                    }
                    ++j;
                }
                break;
            }
            case 1:
                s.erase(s.begin());
                res.eval = std::stoi(s) * (1-2*res.side);
                break;
            case 2:
                s.erase(s.begin());
                if (s == "0") {res.isDraw = false; res.result = res.side;}
                else if (s == "1") {res.isDraw = false; res.result = !res.side;}
                else {res.isDraw = true;}
                break;
        }
        ++i;
    }
    return res;
}

inline void parseFile(const std::filesystem::path& filePath, std::mt19937_64& _mt, const double trainingRatio, const U64 trainingChunks, const U64 validationChunks, std::mutex* const trainingMutex, std::mutex* const validationMutex)
{
    std::uniform_int_distribution<U64> trainingDist(0, trainingChunks-1);
    std::uniform_int_distribution<U64> validationDist(0, validationChunks-1);
    std::uniform_real_distribution<double> splitDist(0., 1.);

    std::vector<datum>* trainingBuffer = new std::vector<datum>[trainingChunks];
    std::vector<datum>* validationBuffer = new std::vector<datum>[validationChunks];

    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line))
    {
        datum res = parseLine(line);

        if (splitDist(_mt) < TRAINING_RATIO) {trainingBuffer[trainingDist(_mt)].push_back(res);}
        else {validationBuffer[validationDist(_mt)].push_back(res);}
    }

    //write results to chunks.
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path trainingDir = cwd / "dataset" / "training";
    std::filesystem::path validationDir = cwd / "dataset" / "validation";

    for (size_t i=0;i<trainingChunks;++i)
    {
        if (size_t length = trainingBuffer[i].size())
        {
            std::unique_lock<std::mutex> lock(trainingMutex[i]);
            std::string chunkName = "chunk_" + std::to_string(i) + ".dat";
            std::ofstream chunk(trainingDir / chunkName, std::ios::binary | std::ios::app);
            chunk.write(reinterpret_cast<char*>(&trainingBuffer[i][0]), length * sizeof(datum));
            chunk.close();
        }
    }

    for (size_t i=0;i<validationChunks;++i)
    {
        if (size_t length = validationBuffer[i].size())
        {
            std::unique_lock<std::mutex> lock(validationMutex[i]);
            std::string chunkName = "chunk_" + std::to_string(i) + ".dat";
            std::ofstream chunk(validationDir / chunkName, std::ios::binary | std::ios::app);
            chunk.write(reinterpret_cast<char*>(&validationBuffer[i][0]), length * sizeof(datum));
            chunk.close();
        }
    }

    delete[] trainingBuffer;
    delete[] validationBuffer;
}

int main(int argc, const char** argv)
{
    if (argc != 3 || !!std::strcmp(argv[1], "-N")) {return 0;}
    auto isNumber = [&]() -> bool {
        bool good = true; size_t i = 0;
        while (good && argv[2][i] != '\0') {good &= (bool)std::isdigit(argv[2][i++]);}
        return good;
    };
    if (!isNumber()) {return 0;}

    size_t numCpu = std::stoi(argv[2]);
    std::thread threads[numCpu];

    std::random_device _rd;
    std::size_t seed;

    std::mt19937_64 _mt[numCpu];
    for (size_t i=0;i<numCpu;++i)
    {
        if (_rd.entropy()) {seed = _rd();}
        else {seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();}
        _mt[i] = std::mt19937_64(seed);
    }

    const std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<std::filesystem::path> rawFiles = getFiles(cwd / "_raw", ".txt");

    U64 total = 0;

    const std::regex lengthRegex(R"(_n([1-9][0-9]*)_)");
    for (const auto& file: rawFiles)
    {
        const std::string fileName = file.filename().string();
        std::smatch match;
        std::regex_search(fileName, match, lengthRegex);
        total += std::stoull(match[1]);
    }
    std::cout << "Found " << rawFiles.size() << " files containing " << total << " pieces of data" << std::endl;

    const U64 chunkSize = 25000000ull;
    const double trainingRatio = 0.95;

    const U64 expectedTraining = (U64)(trainingRatio * total);
    const U64 expectedValidation = total - expectedTraining;

    const U64 trainingChunks = std::max(1ull, expectedTraining / chunkSize);
    const U64 validationChunks = std::max(1ull, expectedValidation / chunkSize);

    std::mutex* trainingMutex = new std::mutex[trainingChunks];
    std::mutex* validationMutex = new std::mutex[validationChunks];

    for (size_t i=0;i<rawFiles.size();++i)
    {
        std::cout << "Iteration: " << i+1 << " / " << rawFiles.size() << std::endl;
        const size_t cpu = i % numCpu;
        threads[cpu] = std::thread(parseFile, std::ref(rawFiles[i]), std::ref(_mt[cpu]), trainingRatio, trainingChunks, validationChunks, trainingMutex, validationMutex);
        if (cpu == numCpu - 1) {for (size_t j=0;j<numCpu;++j) {threads[j].join();}}
    }
    for (size_t i=0;i<rawFiles.size()%numCpu;++i) {threads[i].join();}

    delete[] trainingMutex;
    delete[] validationMutex;

    return 0;
}
