#include "../../pipeline/dataloader.h"
#include "../../pipeline/utils.h"
#include "../include/datum.h"

inline float datumToLabel(const Datum& datum) {
    const float alpha = 0.2f;
    const float beta = 0.75f;

    const float naiveRatio = 1.f / (datum.totalPly - datum.ply);
    const float localRatio = (float)datum.timeSpent / (float)datum.timeLeft;
    const float globalRatio = (float)datum.timeSpent / (float)datum.totalTimeSpent;

    return alpha * naiveRatio + (1.f - alpha) * (beta * localRatio + (1.f - beta) * globalRatio);
};

struct Batch {
    float* tensor;
    float* scaledEval;
    float* scaledPly;
    float* scaledIncrement;
    float* scaledOpponentTime;
    float* label;
    int size;

    Batch(size_t batchSize, Datum* data) : size(batchSize) {
        tensor = new float[batchSize * 64 * 14];
        scaledEval = new float[batchSize];
        scaledPly = new float[batchSize];
        scaledIncrement = new float[batchSize];
        scaledOpponentTime = new float[batchSize];
        label = new float[batchSize];

        for (size_t i = 0; i < batchSize; ++i) {
            reformat(i, data[i]);
        }
    }

    void reformat(size_t idx, const Datum& datum) {
        auto callback = [this, idx](int pieceType, int square) {
            this->tensor[idx * 64 * 14 + pieceType * 64 + square] = 1;
        };

        parsePos(datum.pos, callback);

        if (datum.side) {
            float* start = tensor + 64 * 14 * idx + 64 * 12;
            std::fill(start, start + 64, 1.f);
        }

        if (datum.inCheck) {
            float* start = tensor + 64 * 14 * idx + 64 * 13;
            std::fill(start, start + 64, 1.f);
        }

        scaledEval[idx] = 1.f / (1.f + std::exp(-datum.qSearch / 400.f));
        scaledPly[idx] = std::min((float)datum.ply / 100.f, 1.f);
        scaledIncrement[idx] = std::min((float)datum.increment / (float)datum.timeLeft, 1.f);
        scaledOpponentTime[idx] = std::min(0.5f * (float)datum.opponentTime / (float)datum.timeLeft, 1.f);

        label[idx] = datumToLabel(datum);
    }

    ~Batch() {
        delete[] tensor;
        delete[] scaledEval;
        delete[] scaledPly;
        delete[] scaledIncrement;
        delete[] scaledOpponentTime;
        delete[] label;
    }
};

using dataLoader = DataLoader<Datum, Batch>;

extern "C" {
void* constructDataLoader(const char* path, const int batchSize, const int numWorkers) {
    return new dataLoader(std::filesystem::path(std::string(path)), batchSize, numWorkers);
}

void destructDataLoader(void* dataloader) { delete static_cast<dataLoader*>(dataloader); }

U64 length(const char* path) {
    U64 res = 0;
    for (const auto& file : getFiles(std::filesystem::path(path), ".dat")) {
        res += std::filesystem::file_size(file) / sizeof(Datum);
    }
    return res;
}

Batch* getBatch(void* dataloader) { return static_cast<dataLoader*>(dataloader)->next(); }

void destructBatch(Batch* batch) { delete batch; }
}
