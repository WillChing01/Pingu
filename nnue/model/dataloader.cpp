#include <algorithm>
#include <atomic>
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
    int size = 0;

    halfKaSparseBatch() {}

    halfKaSparseBatch(size_t batchSize, datum* data): size(batchSize)
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
    
        //king indices.
        whiteFeatures[totalFeatures] = (704 * datum.kingPos[0]) + datum.kingPos[1];
        blackFeatures[totalFeatures] = (704 * datum.kingPos[1]) + (datum.kingPos[0] ^ 56);
        ++totalFeatures;

        for (size_t i=0;i<4;++i)
        {
            U64 x = ~datum.pos[i];
            while (x)
            {
                U64 j = __builtin_ctzll(x) / 4ull;
                U64 square = 16ull * i + j;
                U64 pieceType = 15ull - ((x & masks[j]) >> (j << 2ull));
                x &= ~masks[j];

                if (pieceType >= 2)
                {
                    whiteFeatures[totalFeatures] = (704 * datum.kingPos[0]) + 64ull * (pieceType - 1ull) + square;
                    blackFeatures[totalFeatures] = (704 * datum.kingPos[1]) + 64ull * (pieceType - 2ull * (pieceType & 1ull)) + (square ^ 56ull);
                    ++totalFeatures;
                }
            }
        }

        std::fill(indices + startFeatures, indices + totalFeatures, idx);

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

struct chunkLoader
{
    std::filesystem::path path;
    
    size_t chunkIndex = 0;
    std::vector<std::filesystem::path> chunkFiles = {};

    // preload only one chunk to save RAM.
    datum* chunkBuffer[2] = {nullptr, nullptr};
    size_t chunkSizeBuffer[2] = {0, 0};
    bool chunkToggle = 0;

    std::atomic<bool> finishFlag = true;

    datum* chunk = nullptr;
    size_t chunkSize = 0;

    std::mt19937_64 _mt;

    chunkLoader(const std::filesystem::path& x): path(x)
    {
        _mt = std::mt19937_64{std::random_device{}()};

        chunkFiles = getFiles(path, ".dat");
        std::shuffle(chunkFiles.begin(), chunkFiles.end(), _mt);

        loadNext();
    }

    void loadNext()
    {
        delete[] chunkBuffer[chunkToggle];

        if (chunkIndex == chunkFiles.size())
        {
            chunkBuffer[chunkToggle] = nullptr;
            finishFlag = true;
            return;
        }

        chunkSizeBuffer[chunkToggle] = std::filesystem::file_size(chunkFiles[chunkIndex]) / sizeof(datum);
        std::ifstream data(chunkFiles[chunkIndex], std::ios::binary);

        chunkBuffer[chunkToggle] = new datum[chunkSizeBuffer[chunkToggle]];
        data.read((char*)chunkBuffer[chunkToggle], chunkSizeBuffer[chunkToggle] * sizeof(datum));

        std::shuffle(&chunkBuffer[chunkToggle][0], &chunkBuffer[chunkToggle][chunkSizeBuffer[chunkToggle]], _mt);

        finishFlag = true;
    }

    std::pair<datum*, size_t> next()
    {
        if (chunkIndex == chunkFiles.size()) {return {nullptr, 0};}

        // wait for pre-loaded chunk.
        while (!finishFlag) {}

        chunk = chunkBuffer[chunkToggle];
        chunkSize = chunkSizeBuffer[chunkToggle];

        //start pre-loading the next chunk.
        ++chunkIndex;
        chunkToggle = !chunkToggle;
        finishFlag = false;
        std::thread(&chunkLoader::loadNext, this).detach();

        return {chunk, chunkSize};
    }

    ~chunkLoader()
    {
        while (!finishFlag) {}
        delete[] chunkBuffer[0];
        delete[] chunkBuffer[1];
    }
};

struct dataLoader
{
    size_t batchSize;
    size_t numWorkers;

    chunkLoader* chunkloader = nullptr;
    datum* chunk = nullptr;
    size_t chunkSize;

    size_t qLength = 8;
    size_t batchCounter = 0;
    std::queue<halfKaSparseBatch*>* batchQueue;
    std::mutex* _m;

    std::atomic<bool>* stopFlags;
    std::atomic<bool>* finishFlags;

    dataLoader(const std::filesystem::path& path, size_t x, size_t y): batchSize(x), numWorkers(y)
    {
        batchQueue = new std::queue<halfKaSparseBatch*>[numWorkers];
        _m = new std::mutex[numWorkers];

        stopFlags = new std::atomic<bool>[numWorkers];
        finishFlags = new std::atomic<bool>[numWorkers];

        std::fill(stopFlags, stopFlags+numWorkers, true);
        std::fill(finishFlags, finishFlags+numWorkers, true);

        chunkloader = new chunkLoader(path);
        getNextChunk();
    }

    void getNextChunk()
    {
        stopThreads();

        std::pair<datum*, size_t> res = chunkloader->next();
        chunk = res.first;
        chunkSize = res.second;

        batchCounter = 0;
        startThreads();
    }

    void startThreads()
    {
        std::fill(stopFlags, stopFlags+numWorkers, false);
        std::fill(finishFlags, finishFlags+numWorkers, false);
        for (size_t i=0;i<numWorkers;++i)
        {
            std::thread(&dataLoader::processBatches, this, i).detach();
        }
    }

    void processBatches(int worker)
    {
        for (size_t i=batchSize*worker;i<chunkSize;i+=batchSize*numWorkers)
        {
            if (stopFlags[worker]) {break;}

            bool pushed = false;
            while (!pushed && !stopFlags[worker])
            {
                std::unique_lock<std::mutex> lock(_m[worker]);
                if (batchQueue[worker].size() < qLength)
                {
                    lock.unlock();
                    halfKaSparseBatch* batch = new halfKaSparseBatch(std::min(chunkSize - i, batchSize), chunk + i);
                    lock.lock();
                    batchQueue[worker].push(batch);
                    pushed = true;
                }
            }
        }
        finishFlags[worker] = true;
    }

    void stopThreads()
    {
        std::fill(stopFlags, stopFlags+numWorkers, true);
        for (size_t i=0;i<numWorkers;++i)
        {
            while (!finishFlags[i]) {}
            while (!batchQueue[i].empty())
            {
                halfKaSparseBatch* batch = batchQueue[i].front();
                batchQueue[i].pop();
                delete batch;
            }
        }
    }

    halfKaSparseBatch* next()
    {
        bool isChunkFinished = batchCounter == ((chunkSize / batchSize) + (bool)(chunkSize % batchSize));
        if (isChunkFinished)
        {
            getNextChunk();
            if (chunk == nullptr) {return nullptr;}
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
        stopThreads();
        delete[] chunk;
        delete[] batchQueue;
        delete[] _m;
        delete[] stopFlags;
        delete[] finishFlags;
    }
};

extern "C" {
    void* constructDataLoader(const char* path, const int batchSize, const int numWorkers)
    {
        return new dataLoader(std::filesystem::path(std::string(path)), batchSize, numWorkers);
    }

    void destructDataLoader(void* dataloader)
    {
        delete static_cast<dataLoader*>(dataloader);
    }

    U64 length(const char* path)
    {
        U64 res = 0;
        for (const auto& file: getFiles(std::filesystem::path(path), ".dat"))
        {
            res += std::filesystem::file_size(file) / sizeof(datum);
        }
        return res;
    }

    halfKaSparseBatch* getBatch(void* dataloader)
    {
        return static_cast<dataLoader*>(dataloader)->next();
    }

    void destructBatch(halfKaSparseBatch* batch)
    {
        delete batch;
    }

    int main();
}

int main()
{
    dataLoader dataloader(std::filesystem::current_path() / ".." / "dataset" / "training", 1024, 6);

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
