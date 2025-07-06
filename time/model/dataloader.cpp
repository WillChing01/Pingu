#include "../../pipeline/dataloader.h"
#include "../include/datum.h"

struct Batch {
    int size;

    Batch(size_t batchSize, Datum* data) : size(batchSize) {}

    ~Batch() {}
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
