#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "utils.h"

struct halfKaSparseBatch
{
    U64* indices;
    U64* firstFeatures;
    U64* secondFeatures;
    double* result;
    short* eval;
    int totalFeatures = 0;

    halfKaSparseBatch() {}

    halfKaSparseBatch(size_t batchSize, datum* data)
    {
        indices = new U64[batchSize * 31];
        firstFeatures = new U64[batchSize * 31];
        secondFeatures = new U64[batchSize * 31];
        result = new double[batchSize];
        eval = new short[batchSize];

        for (size_t i=0;i<batchSize;++i) {datumToSparse(i, data[i]);}
    }

    void datumToSparse(size_t idx, const datum& datum)
    {
        result[idx] = datum.isDraw ? 0.5 : datum.result;
        eval[idx] = datum.eval;

        U64* whiteFeatures = datum.side ? secondFeatures : firstFeatures;
        U64* blackFeatures = datum.side ? firstFeatures : secondFeatures;

        int startFeatures = totalFeatures;

        for (size_t i=0;i<4;++i)
        {
            U64 x = ~datum.pos[i];
            while (x)
            {
                U64 j = __builtin_ctzll(x) / 4ull;
                U64 square = 16ull * i + j;
                U64 pieceType = 15ull - ((x & masks[j]) >> (j << 2ull));
                x &= ~masks[j];
                switch(pieceType)
                {
                    case 0:
                        blackFeatures[totalFeatures] = (704 * datum.kingPos[1]) + (square ^ 56ull);
                        break;
                    case 1:
                        whiteFeatures[totalFeatures] = (704 * datum.kingPos[0]) + square;
                        break;
                    default:
                        whiteFeatures[totalFeatures] = (704 * datum.kingPos[0]) + 64ull * (pieceType - 1ull) + square;
                        blackFeatures[totalFeatures] = (704 * datum.kingPos[1]) + 64ull * (pieceType - 2ull * (pieceType & 1ull)) + (square ^ 56ull);
                        break;
                }
                indices[totalFeatures] = idx;
                ++totalFeatures;
            }
        }

        //order indices in ascending order.
        std::sort(firstFeatures + startFeatures, firstFeatures + totalFeatures);
        std::sort(secondFeatures + startFeatures, secondFeatures + totalFeatures);
    }

    ~halfKaSparseBatch()
    {
        delete[] indices;
        delete[] firstFeatures;
        delete[] secondFeatures;
        delete[] result;
        delete[] eval;
    }
};

struct dataLoader
{
    std::filesystem::path path;
    size_t batchSize;
    size_t numWorkers;

    size_t chunkIndex = 0;
    std::vector<std::filesystem::path> chunkFiles = {};

    datum* chunk = nullptr;
    size_t chunkSize;

    size_t qLength = 8;
    size_t batchCounter = 0;
    std::vector<int>* batchIndices;
    std::queue<halfKaSparseBatch*>* batchQueue;
    std::mutex* _m;

    std::mt19937_64 _mt;

    dataLoader(const std::filesystem::path& x, size_t y, size_t z) : path(x), batchSize(y), numWorkers(z)
    {
        chunkFiles = getFiles(path, ".dat");
        _mt = std::mt19937_64{std::random_device{}()};

        batchIndices = new std::vector<int>[numWorkers];
        batchQueue = new std::queue<halfKaSparseBatch*>[numWorkers];
        _m = new std::mutex[numWorkers];

        prepareForNewIteration();
        refreshChunk(0);
    }

    void prepareForNewIteration()
    {
        chunkIndex = 0;
        // std::shuffle(chunkFiles.begin(), chunkFiles.end(), _mt);
    }

    void refreshChunk(size_t index)
    {
        delete[] chunk;
        chunkSize = std::filesystem::file_size(chunkFiles[index]) / sizeof(datum);
        chunk = new datum[chunkSize];

        std::ifstream data(chunkFiles[index], std::ios::binary);
        data.read((char*)chunk, chunkSize * sizeof(datum));

        // std::shuffle(chunk, chunk + chunkSize, _mt);

        batchCounter = 0;
        for (size_t i=0;i<chunkSize;i+=batchSize)
        {
            batchIndices[(i/batchSize) % numWorkers].push_back(i);
        }
        for (size_t i=0;i<numWorkers;++i)
        {
            std::thread(&dataLoader::processBatches, this, std::ref(_m[i]), std::ref(batchIndices[i]), std::ref(batchQueue[i])).detach();
        }
    }

    void processBatches(std::mutex& m, std::vector<int>& indices, std::queue<halfKaSparseBatch*>& queue)
    {
        while (indices.size())
        {
            std::unique_lock<std::mutex> lock(m);
            if (queue.size() < qLength)
            {
                lock.unlock();
                int idx = indices.back();
                indices.pop_back();
                halfKaSparseBatch* batch = new halfKaSparseBatch(std::min(chunkSize - idx, batchSize), chunk + idx);
                lock.lock();
                queue.push(batch);
            }
        }
    }

    halfKaSparseBatch* next()
    {
        bool isChunkFinished = batchCounter == ((chunkSize / batchSize) + (bool)(chunkSize % batchSize));
        if (isChunkFinished)
        {
            bool isIterationFinished = chunkIndex == chunkFiles.size() - 1;
            if (isIterationFinished) {return nullptr;}
            refreshChunk(++chunkIndex);
            return next();
        }

        int worker = batchCounter++ % numWorkers;

        while (true)
        {
            std::unique_lock<std::mutex> lock(_m[worker]);
            if (batchQueue[worker].size())
            {
                halfKaSparseBatch* batch = batchQueue[worker].front();
                batchQueue[worker].pop();
                return batch;
            }
        }
    }

    ~dataLoader()
    {
        delete[] chunk;
        delete[] batchIndices;
        delete[] batchQueue;
        delete[] _m;
    }
};

int main()
{
    dataLoader dataloader(std::filesystem::current_path() / "dataset" / "training", 1024, 6);

    auto startTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    int batchCounter = 0;
    while (true)
    {
        halfKaSparseBatch* batch = dataloader.next();
        if (batch == nullptr) {break;}
        if (batchCounter % 1000 == 0) {std::cout << batchCounter << std::endl;}
        ++batchCounter;
        delete batch;
    }

    auto finishTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "done in " << finishTime - startTime << " seconds" << std::endl;

    return 0;
}
