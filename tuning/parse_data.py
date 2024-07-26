"""
Parse and format data in book files.
"""

import os
import numpy as np

TRAINING_RATIO = 0.95
VALIDATION_RATIO = 1 - TRAINING_RATIO

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
    x = np.zeros(32, dtype = np.short)
    x[0:len(indices)] = np.array(indices, copy=True)
    x[len(indices):].fill(indices[0])
    return x

def main():

    directory = os.getcwd() + "/datasets/"

    n = 0
    total_files = 0
    for (root, dirs, files) in os.walk(directory):
        if len(files) == 0:
            continue
        i = 0
        total_files += len(files)
        for file in files:
            tokens = file.split("_")
            num = ""
            for element in tokens:
                if element[0] == "n":
                    num = element[1:]
                    break
            i += int(num)
        n += i
        print("Found " + str(len(files)) + " files containing " + str(i) + " pieces of data in " + root)

    print("Found " + str(n) + " pieces of data in total.")

    print("Storing data...")

    sparse_input = np.memmap("sparse_input_"+str(n)+"_32.dat", mode = "w+", dtype = np.short, shape = (n, 32))
    labels = np.memmap("labels_"+str(n)+"_2.dat", mode = "w+", dtype = np.float32, shape = (n, 2))

    file_count = 0
    i = 0
    for (root, dirs, files) in os.walk(directory):
        if len(files) == 0:
            continue
        for file in files:
            file_count += 1
            print("Reading file", file_count, "/", total_files, ";", file)
            with open(root + "/" + file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    fen, evaluation, result = line.rstrip().split("; ")
                    array = sparseToArray(fenToSparse(fen))
                    sparse_input[i] = np.array(array, copy=True)
                    labels[i][0] = int(evaluation)
                    labels[i][1] = float(result)
                    i += 1
                sparse_input.flush()
                labels.flush()

    print("Successfully stored data.")

    print("Process complete.")

if __name__ == "__main__":
    main()
