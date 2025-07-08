#include "../../pipeline/dataloader.h"
#include "../../pipeline/utils.h"
#include "../include/datum.h"

inline double datumToLabel(const Datum& datum) {
    const double alpha = 0.2;
    const double beta = 0.75;

    const double naiveRatio = 1. / (datum.totalPly - datum.ply);
    const double localRatio = (double)datum.timeSpent / (double)datum.timeLeft;
    const double globalRatio = (double)datum.timeSpent / (double)datum.totalTimeSpent;

    return alpha * naiveRatio + (1. - alpha) * (beta * localRatio + (1. - beta) * globalRatio);
};

struct Batch {
    U64* tensor;
    double* scaledEval;
    double* scaledPly;
    double* scaledIncrement;
    double* scaledOpponentTime;
    double* label;
    int size;

    Batch(size_t batchSize, Datum* data) : size(batchSize) {
        tensor = new U64[batchSize * 64 * 14];
        scaledEval = new double[batchSize];
        scaledPly = new double[batchSize];
        scaledIncrement = new double[batchSize];
        scaledOpponentTime = new double[batchSize];
        label = new double[batchSize];

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
            U64* start = tensor + 64 * 14 * idx + 64 * 12;
            std::fill(start, start + 64, 1.);
        }

        if (datum.inCheck) {
            U64* start = tensor + 64 * 14 * idx + 64 * 13;
            std::fill(start, start + 64, 1.);
        }

        scaledEval[idx] = 1. / (1. + std::exp(-datum.qSearch / 400.));
        scaledPly[idx] = std::min((double)datum.ply / 100., 1.);
        scaledIncrement[idx] = std::min((double)datum.increment / (double)datum.timeLeft, 1.);
        scaledOpponentTime[idx] = std::min(0.5 * (double)datum.opponentTime / (double)datum.timeLeft, 1.);

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
