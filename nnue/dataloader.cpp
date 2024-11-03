#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "utils.h"

struct halfKaSparseBatch
{
    U64* firstFeatures;
    U64* secondFeatures;
    double* result;
    short* eval;
    unsigned char* activeFeatures;

    halfKaSparseBatch(size_t batchSize, datum* data)
    {
        firstFeatures = new U64[batchSize * 31]();
        secondFeatures = new U64[batchSize * 31]();
        result = new double[batchSize]();
        eval = new short[batchSize]();
        activeFeatures = new unsigned char[batchSize]();

        for (size_t i=0;i<batchSize;++i) {datumToSparse(i, data[i]);}
    }

    void datumToSparse(size_t idx, const datum& datum)
    {
        result[idx] = datum.isDraw ? 0.5 : datum.result;
        eval[idx] = datum.eval;

        U64* whiteFeatures = datum.side ? secondFeatures + 31 * idx : firstFeatures + 31 * idx;
        U64* blackFeatures = datum.side ? firstFeatures + 31 * idx : secondFeatures + 31 * idx;

        activeFeatures[idx] = 0;

        for (size_t i=0;i<4;++i)
        {
            for (size_t j=0;j<16;++j)
            {
                size_t pieceType = (datum.pos[i] & masks[j]) >> (4 * j);

                if (pieceType != 15)
                {
                    size_t square = 16 * i + j;
                    switch (pieceType)
                    {
                        case 0:
                            *blackFeatures++ = (704 * datum.kingPos[1]) + (square ^ 56);
                            break;
                        case 1:
                            *whiteFeatures++ = (704 * datum.kingPos[0]) + square;
                            break;
                        default:
                            *whiteFeatures++ = (704 * datum.kingPos[0]) + 64 * (pieceType - 1) + square;
                            *blackFeatures++ = (704 * datum.kingPos[1]) + 64 * (pieceType - 2 * (pieceType & 1)) + (square ^ 56);
                            break;
                    }
                    ++activeFeatures[idx];
                }
            }
        }

        //order indices in ascending order.
        std::sort(whiteFeatures - activeFeatures[idx], whiteFeatures);
        std::sort(blackFeatures - activeFeatures[idx], blackFeatures);
    }

    ~halfKaSparseBatch()
    {
        delete[] firstFeatures;
        delete[] secondFeatures;
        delete[] result;
        delete[] eval;
        delete[] activeFeatures;
    }
};

struct dataLoader
{
    std::string path;
    size_t chunkIndex = 0;
    std::vector<std::filesystem::path> chunkFiles = {};

    datum* chunk = nullptr;
    size_t chunkSize;

    size_t batchSize;
    size_t batchIndex = 0;

    bool finished = false;

    std::mt19937_64 _mt{std::random_device{}()};

    dataLoader(const std::string& x, size_t y) : path(x), batchSize(y)
    {
        chunkFiles = getFiles(path, ".dat");
        std::shuffle(chunkFiles.begin(), chunkFiles.end(), _mt);
    }

    void prepareForNewIteration()
    {
        finished = false;

        chunkIndex = 0;
        std::shuffle(chunkFiles.begin(), chunkFiles.end(), _mt);

        batchIndex = 0;
    }

    void refreshChunk(size_t index)
    {
        batchIndex = 0;

        delete[] chunk;
        chunkSize = std::filesystem::file_size(chunkFiles[index]) / sizeof(datum);
        chunk = new datum[chunkSize];

        std::ifstream data(chunkFiles[index], std::ios::binary);
        data.read((char*)chunk, chunkSize * sizeof(datum));

        std::shuffle(chunk, chunk + chunkSize, _mt);
    }

    halfKaSparseBatch next(size_t startIndex)
    {
        return halfKaSparseBatch(std::min(chunkSize - startIndex, batchSize), chunk + startIndex);
    }

    ~dataLoader() {delete[] chunk;}
};
