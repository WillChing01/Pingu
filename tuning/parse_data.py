"""
Parse and format data in book files.
"""

import os
import numpy as np

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

    files = [i for i in os.listdir(directory) if i.split(".")[-1] == "txt" and i[0:7] == "gensfen"]

    print("Found", len(files), "files.")

    print("Reading files...")

    n = 0
    for file in files:
        with open(directory+file, "r") as f:
            lines = f.readlines()
            n += len(lines)

    print("Collected", n, "pieces of data.")

    print("Storing data...")

    sparse_input = np.memmap("sparse_input_"+str(n)+"_32.dat", mode = "w+", dtype = np.short, shape = (n, 32))
    labels = np.memmap("labels_"+str(n)+"_1.dat", mode = "w+", dtype = np.short, shape = (n, 1))

    file_count = 0
    i = 0
    for file in files:
        file_count += 1
        print("Reading file", file_count, ";", file)
        with open(directory+file, "r") as f:
            lines = f.readlines()
            for x in lines:
                fen, eval_ = x.rstrip().split("; ")
                arr = sparseToArray(fenToSparse(fen))
                sparse_input[i] = np.array(arr, copy=True)
                labels[i][0] = int(eval_)
                i += 1
                if i % 1000000 == 0:
                    print(i)
    sparse_input.flush()
    labels.flush()

    validation_ratio = 0.05
    validation_num = int(round(validation_ratio * n))

    print("Generating", validation_num, "validation samples...")

    indices = np.array([i for i in range(n)])
    np.random.shuffle(indices)

    validation_input = np.memmap("validation_input_"+str(validation_num)+"_32.dat", mode = "w+", dtype = np.short, shape = (validation_num, 32))
    validation_labels = np.memmap("validation_labels_"+str(validation_num)+"_1.dat", mode = "w+", dtype = np.short, shape = (validation_num, 1))
    
    for i in range(0, validation_num):
        validation_input[i] = np.array(sparse_input[indices[i]], copy=True)
        validation_labels[i][0] = labels[indices[i]][0]
    validation_input.flush()
    validation_labels.flush()

    print("Generating", n - validation_num, "training samples...")

    training_input = np.memmap("training_input_"+str(n - validation_num)+"_32.dat", mode = "w+", dtype = np.short, shape = (n - validation_num, 32))
    training_labels = np.memmap("training_labels_"+str(n - validation_num)+"_1.dat", mode = "w+", dtype = np.short, shape = (n - validation_num, 1))

    for i in range(validation_num, n):
        if (i - validation_num) % 1000000 == 0:
            print("Progress", i - validation_num)
        training_input[i-validation_num] = np.array(sparse_input[indices[i]], copy=True)
        training_labels[i-validation_num][0] = labels[indices[i]][0]
    training_input.flush()
    training_labels.flush()

    print("Process complete.")

if __name__ == "__main__":
    main()
