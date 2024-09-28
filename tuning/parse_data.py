"""Parse and format data in datasets folder"""

import os
import sys
import numpy as np
import multiprocessing

DATASET_DTYPE = np.short

def fenToSparse(fen):
    """Return non-zero indices of input matrix derived from FEN."""

    fen = fen.split(' ')[0]
    indices = []

    square = 56
    pieceTypes = "KkQqRrBbNnPp"

    for i in range(len(fen)):
        if fen[i] == '/':
            square -= 16
        elif fen[i].isdigit():
            square += int(fen[i])
        else:
            indices.append(64*pieceTypes.find(fen[i]) + square)
            square += 1

    return indices

def sparseToArray(indices):
    x = np.zeros(32, dtype = DATASET_DTYPE)
    x[0:len(indices)] = np.array(indices, copy=True)
    x[len(indices):].fill(indices[0])
    return x

def parseFile(startIndex, file_name, dataset_file, dataset_shape):
    dataset = np.memmap(dataset_file, mode = "r+", dtype = DATASET_DTYPE, shape = dataset_shape)

    i = startIndex
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            fen, evaluation, result = line.rstrip().split("; ")

            fenData = sparseToArray(fenToSparse(fen))
            evalData = np.array([int(evaluation)], dtype = DATASET_DTYPE)
            resultData = np.array([int(round(2. * float(result)))], dtype = DATASET_DTYPE)

            dataset[i] = np.array(np.concatenate([fenData, evalData, resultData]), copy = True)
            i += 1

    dataset.flush()
    dataset._mmap.close()

    return True

def main():
    user_args = sys.argv[1:]
    if len(user_args) != 2 or user_args[0] != "-N" or not user_args[1].isdigit():
        print("error: incorrect format of args")
        print("usage: parse_data.py -N <num_threads>")
        print("e.g. parse_data.py -N 2")
        return None
    elif int(user_args[1]) > multiprocessing.cpu_count():
        print("error: not enough cpu threads")
        return None
    num_cpu = int(user_args[1])

    directory = os.getcwd() + "/datasets/"
    fileData = []
    n = 0

    print("Searching for data...")

    for (root, dirs, files) in os.walk(directory):
        if len(files) == 0:
            continue
        for file in files:
            fileData.append([n, root + "/" + file])
            tokens = file.split("_")
            num = ""
            for element in tokens:
                if element[0] == "n":
                    num = element[1:]
                    break
            n += int(num)

    print("Found", n, "pieces of data in", len(fileData), "files.")

    print("Storing data...")

    dataset_file = "dataset_" + str(n) + "_34.dat"
    dataset_shape = (n, 34)

    try:
        dataset = np.memmap(dataset_file, mode = "r", dtype = DATASET_DTYPE, shape = dataset_shape)
        dataset._mmap.close()
    except:
        dataset = np.memmap(dataset_file, mode = "w+", dtype = DATASET_DTYPE, shape = dataset_shape)
        dataset._mmap.close()

    pool = multiprocessing.Pool(num_cpu)
    result = [None for i in range(num_cpu)]

    file_number = 1
    total_files = len(fileData)
    fileData.reverse()
    while True:
        all_ready = True
        for i in range(num_cpu):
            if result[i] is None or result[i].ready():
                try:
                    data = fileData.pop()
                    args = (data[0], data[1], dataset_file, dataset_shape)
                    print("Reading file", file_number, "/", total_files, ";", data[1])
                    result[i] = pool.apply_async(parseFile, args = args)
                    file_number += 1
                except:
                    pass
            else:
                all_ready = False
        if all_ready and len(fileData) == 0:
            break

    print("Process complete.")

if __name__ == "__main__":
    main()
