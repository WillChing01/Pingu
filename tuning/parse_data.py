"""
Parse and format data in book files.
"""

import os
import sys
import numpy as np
import multiprocessing

INPUT_DTYPE = np.short
LABEL_DTYPE = np.float32

def fenToSparse(fen):
    """
        Return non-zero indices of input matrix derived from FEN.
    """

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
    x = np.zeros(32, dtype = INPUT_DTYPE)
    x[0:len(indices)] = np.array(indices, copy=True)
    x[len(indices):].fill(indices[0])
    return x

def parseFile(startIndex, fileName, input_name, input_shape, label_name, label_shape):
    sparse_input = np.memmap(input_name, mode = "r+", dtype = INPUT_DTYPE, shape = input_shape)
    labels = np.memmap(label_name, mode = "r+", dtype = LABEL_DTYPE, shape = label_shape)

    i = startIndex
    with open(fileName, "r") as f:
        lines = f.readlines()
        for line in lines:
            fen, evaluation, result = line.rstrip().split("; ")
            array = sparseToArray(fenToSparse(fen))
            sparse_input[i] = np.array(array, copy = True)
            labels[i][0] = int(evaluation)
            labels[i][1] = float(result)
            i += 1

    sparse_input.flush()
    labels.flush()

    sparse_input._mmap.close()
    labels._mmap.close()

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

    input_name = "sparse_input_"+str(n)+"_32.dat"
    input_shape = (n, 32)

    label_name = "labels_"+str(n)+"_2.dat"
    label_shape = (n, 2)

    try:
        sparse_input = np.memmap(input_name, mode = "r", dtype = INPUT_DTYPE, shape = input_shape)
        labels = np.memmap(label_name, mode = "r", dtype = LABEL_DTYPE, shape = label_shape)
    except:
        sparse_input = np.memmap(input_name, mode = "w+", dtype = INPUT_DTYPE, shape = input_shape)
        labels = np.memmap(label_name, mode = "w+", dtype = LABEL_DTYPE, shape = label_shape)

    sparse_input._mmap.close()
    labels._mmap.close()

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
                    args = (data[0], data[1], input_name, input_shape, label_name, label_shape)
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
