#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <filesystem>
#include <vector>

typedef unsigned long long U64;

std::vector<std::filesystem::path> getFiles(const std::filesystem::path& path, const std::filesystem::path& ext) {
    std::vector<std::filesystem::path> res = {};

    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        const std::filesystem::path entryPath = entry.path();
        if (entryPath.extension() == ext) {
            res.push_back(entryPath);
        }
    }

    return res;
}

#endif // UTILS_H_INCLUDED
