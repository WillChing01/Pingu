#ifndef DATAPARSER_H_INCLUDED
#define DATAPARSER_H_INCLUDED

#include "utils.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

template <size_t NUM_CPU, size_t CHUNK_SIZE, double TRAINING_RATIO, typename Datum,
          Datum (*parseLine)(const std::string& line)>
class DataParser {
  private:
    std::filesystem::path RAW_PATH;
    std::string RAW_EXT;
    std::filesystem::path TRAINING_PATH;
    std::filesystem::path VALIDATION_PATH;

    std::thread _threads[NUM_CPU];
    std::mt19937_64 _mt[NUM_CPU];

    U64 _trainingChunks;
    U64 _validationChunks;

    std::vector<std::mutex> _trainingMutex;
    std::vector<std::mutex> _validationMutex;

    void parseFile(size_t cpuIndex, const std::filesystem::path& filePath) {
        std::uniform_int_distribution<U64> trainingDist(0, _trainingChunks - 1);
        std::uniform_int_distribution<U64> validationDist(0, _validationChunks - 1);
        std::uniform_real_distribution<double> splitDist(0., 1.);

        std::vector<std::vector<Datum>> trainingBuffer = std::vector<std::vector<Datum>>(_trainingChunks);
        std::vector<std::vector<Datum>> validationBuffer = std::vector<std::vector<Datum>>(_validationChunks);

        std::ifstream file(filePath);
        std::string line;
        while (std::getline(file, line)) {
            Datum res = parseLine(line);

            if (splitDist(_mt[cpuIndex]) < TRAINING_RATIO) {
                trainingBuffer[trainingDist(_mt[cpuIndex])].push_back(res);
            } else {
                validationBuffer[validationDist(_mt[cpuIndex])].push_back(res);
            }
        }
        file.close();
        std::filesystem::remove(filePath);

        for (size_t i = 0; i < _trainingChunks; ++i) {
            if (size_t length = trainingBuffer[i].size()) {
                std::unique_lock<std::mutex> lock(_trainingMutex[i]);
                std::string chunkName = "chunk_" + std::to_string(i) + ".dat";
                std::ofstream chunk(TRAINING_PATH / chunkName, std::ios::binary | std::ios::app);
                chunk.write(reinterpret_cast<char*>(&trainingBuffer[i][0]), length * sizeof(Datum));
                chunk.close();
            }
        }

        for (size_t i = 0; i < _validationChunks; ++i) {
            if (size_t length = validationBuffer[i].size()) {
                std::unique_lock<std::mutex> lock(_validationMutex[i]);
                std::string chunkName = "chunk_" + std::to_string(i) + ".dat";
                std::ofstream chunk(VALIDATION_PATH / chunkName, std::ios::binary | std::ios::app);
                chunk.write(reinterpret_cast<char*>(&validationBuffer[i][0]), length * sizeof(Datum));
                chunk.close();
            }
        }
    }

  public:
    DataParser(const std::filesystem::path& rawPath, const std::string& rawExt,
               const std::filesystem::path& trainingPath, const std::filesystem::path& validationPath)
        : RAW_PATH(rawPath), RAW_EXT(rawExt), TRAINING_PATH(trainingPath), VALIDATION_PATH(validationPath) {
        std::random_device _rd;
        std::size_t seed;

        for (size_t i = 0; i < NUM_CPU; ++i) {
            if (_rd.entropy()) {
                seed = _rd();
            } else {
                seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + i;
            }
            _mt[i] = std::mt19937_64(seed);
        }
    }

    void parseFiles() {
        std::vector<std::filesystem::path> rawFiles = getFiles(RAW_PATH, RAW_EXT);

        size_t total = 0;

        const std::regex lengthRegex(R"(_n([1-9][0-9]*)_)");
        for (const auto& file : rawFiles) {
            const std::string fileName = file.filename().string();
            std::smatch match;
            std::regex_search(fileName, match, lengthRegex);
            total += std::stoull(match[1]);
        }
        std::cout << "Found " << rawFiles.size() << " files containing " << total << " pieces of data" << std::endl;

        const double expectedTraining = TRAINING_RATIO * total;
        const double expectedValidation = total - expectedTraining;

        _trainingChunks = std::max(1ull, (U64)std::round(expectedTraining / CHUNK_SIZE));
        _validationChunks = std::max(1ull, (U64)std::llround(expectedValidation / CHUNK_SIZE));

        _trainingMutex = std::vector<std::mutex>(_trainingChunks);
        _validationMutex = std::vector<std::mutex>(_validationChunks);

        for (size_t i = 0; i < rawFiles.size(); ++i) {
            std::cout << i + 1 << " / " << rawFiles.size() << " : " << rawFiles[i].filename().string() << std::endl;
            const size_t cpuIndex = i % NUM_CPU;
            _threads[cpuIndex] = std::thread(&DataParser::parseFile, this, cpuIndex, std::ref(rawFiles[i]));
            if (cpuIndex == NUM_CPU - 1) {
                for (size_t j = 0; j < NUM_CPU; ++j) {
                    _threads[j].join();
                }
            }
        }
        for (size_t i = 0; i < rawFiles.size() % NUM_CPU; ++i) {
            _threads[i].join();
        }
    }
};

const std::unordered_map<unsigned char, unsigned char> pieceTypeMap = {
    {'K', 0},
    {'k', 1},
    {'Q', 2},
    {'q', 3},
    {'R', 4},
    {'r', 5},
    {'B', 6},
    {'b', 7},
    {'N', 8},
    {'n', 9},
    {'P', 10},
    {'p', 11},
};

const U64 masks[16] = {
    0xFull,
    0xF0ull,
    0xF00ull,
    0xF000ull,
    0xF0000ull,
    0xF00000ull,
    0xF000000ull,
    0xF0000000ull,
    0xF00000000ull,
    0xF000000000ull,
    0xF0000000000ull,
    0xF00000000000ull,
    0xF000000000000ull,
    0xF0000000000000ull,
    0xF00000000000000ull,
    0xF000000000000000ull,
};

inline void parseFen(const std::string& fen, U64* const res) {
    unsigned char square = 56;
    for (const unsigned char x : fen) {
        switch (x) {
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
            res[square >> 4] -= (15ull - (U64)pieceTypeMap.at(x)) << (U64)((square & 15) << 2);
            ++square;
            break;
        }
    }
}

#endif // DATAPARSER_H_INCLUDED
