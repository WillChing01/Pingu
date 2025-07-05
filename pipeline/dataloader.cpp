#include "utils.h"

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

template <typename Datum>
struct chunkLoader {
    std::filesystem::path path;

    size_t chunkIndex = 0;
    std::vector<std::filesystem::path> chunkFiles = {};

    // preload only one chunk to save RAM.
    Datum* chunkBuffer[2] = {nullptr, nullptr};
    size_t chunkSizeBuffer[2] = {0, 0};
    bool chunkToggle = 0;

    std::atomic<bool> finishFlag = true;

    Datum* chunk = nullptr;
    size_t chunkSize = 0;

    std::mt19937_64 _mt;

    chunkLoader(const std::filesystem::path& x) : path(x) {
        _mt = std::mt19937_64{std::random_device{}()};

        chunkFiles = getFiles(path, ".dat");
        std::shuffle(chunkFiles.begin(), chunkFiles.end(), _mt);

        loadNext();
    }

    void loadNext() {
        delete[] chunkBuffer[chunkToggle];

        if (chunkIndex == chunkFiles.size()) {
            chunkBuffer[chunkToggle] = nullptr;
            finishFlag = true;
            return;
        }

        chunkSizeBuffer[chunkToggle] = std::filesystem::file_size(chunkFiles[chunkIndex]) / sizeof(Datum);
        std::ifstream data(chunkFiles[chunkIndex], std::ios::binary);

        chunkBuffer[chunkToggle] = new Datum[chunkSizeBuffer[chunkToggle]];
        data.read((char*)chunkBuffer[chunkToggle], chunkSizeBuffer[chunkToggle] * sizeof(Datum));

        std::shuffle(&chunkBuffer[chunkToggle][0], &chunkBuffer[chunkToggle][chunkSizeBuffer[chunkToggle]], _mt);

        finishFlag = true;
    }

    std::pair<Datum*, size_t> next() {
        if (chunkIndex == chunkFiles.size()) {
            return {nullptr, 0};
        }

        // wait for pre-loaded chunk.
        while (!finishFlag) {
        }

        chunk = chunkBuffer[chunkToggle];
        chunkSize = chunkSizeBuffer[chunkToggle];

        // start pre-loading the next chunk.
        ++chunkIndex;
        chunkToggle = !chunkToggle;
        finishFlag = false;
        std::thread(&chunkLoader::loadNext, this).detach();

        return {chunk, chunkSize};
    }

    ~chunkLoader() {
        while (!finishFlag) {
        }
        delete[] chunkBuffer[0];
        delete[] chunkBuffer[1];
    }
};

template <typename Datum, typename Batch>
struct dataLoader {
    size_t batchSize;
    size_t numWorkers;

    chunkLoader<Datum>* chunkloader = nullptr;
    Datum* chunk = nullptr;
    size_t chunkSize;

    size_t qLength = 8;
    size_t batchCounter = 0;
    std::queue<Batch*>* batchQueue;
    std::mutex* _m;

    std::atomic<bool>* stopFlags;
    std::atomic<bool>* finishFlags;

    dataLoader(const std::filesystem::path& path, size_t x, size_t y) : batchSize(x), numWorkers(y) {
        batchQueue = new std::queue<Batch*>[numWorkers];
        _m = new std::mutex[numWorkers];

        stopFlags = new std::atomic<bool>[numWorkers];
        finishFlags = new std::atomic<bool>[numWorkers];

        std::fill(stopFlags, stopFlags + numWorkers, true);
        std::fill(finishFlags, finishFlags + numWorkers, true);

        chunkloader = new chunkLoader<Datum>(path);
        getNextChunk();
    }

    void getNextChunk() {
        stopThreads();

        std::pair<Datum*, size_t> res = chunkloader->next();
        chunk = res.first;
        chunkSize = res.second;

        batchCounter = 0;
        startThreads();
    }

    void startThreads() {
        std::fill(stopFlags, stopFlags + numWorkers, false);
        std::fill(finishFlags, finishFlags + numWorkers, false);
        for (size_t i = 0; i < numWorkers; ++i) {
            std::thread(&dataLoader::processBatches, this, i).detach();
        }
    }

    void processBatches(int worker) {
        for (size_t i = batchSize * worker; i < chunkSize; i += batchSize * numWorkers) {
            if (stopFlags[worker]) {
                break;
            }

            bool pushed = false;
            while (!pushed && !stopFlags[worker]) {
                std::unique_lock<std::mutex> lock(_m[worker]);
                if (batchQueue[worker].size() < qLength) {
                    lock.unlock();
                    Batch* batch = new Batch(std::min(chunkSize - i, batchSize), chunk + i);
                    lock.lock();
                    batchQueue[worker].push(batch);
                    pushed = true;
                }
            }
        }
        finishFlags[worker] = true;
    }

    void stopThreads() {
        std::fill(stopFlags, stopFlags + numWorkers, true);
        for (size_t i = 0; i < numWorkers; ++i) {
            while (!finishFlags[i]) {
            }
            while (!batchQueue[i].empty()) {
                Batch* batch = batchQueue[i].front();
                batchQueue[i].pop();
                delete batch;
            }
        }
    }

    Batch* next() {
        bool isChunkFinished = batchCounter == ((chunkSize / batchSize) + (bool)(chunkSize % batchSize));
        if (isChunkFinished) {
            getNextChunk();
            if (chunk == nullptr) {
                return nullptr;
            }
            return next();
        }

        int worker = batchCounter++ % numWorkers;

        while (true) {
            std::unique_lock<std::mutex> lock(_m[worker]);
            if (batchQueue[worker].size()) {
                Batch* batch = batchQueue[worker].front();
                batchQueue[worker].pop();
                return batch;
            }
        }
    }

    ~dataLoader() {
        stopThreads();
        delete[] batchQueue;
        delete[] _m;
        delete[] stopFlags;
        delete[] finishFlags;
        delete chunkloader;
    }
};
