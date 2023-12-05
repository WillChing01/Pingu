"""
Parse and format data in book files.
"""

import os
import random
import pickle

def fenToSparse(fen):
    """
        Return non-zero indices of input matrix derived from FEN.
    """

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

def main():

    directory = os.getcwd() + "/datasets/"

    files = [i for i in os.listdir(directory) if i.split(".")[-1] == "txt" and i[0:7] == "gensfen"]

    data = []

    for file in files:
        print("Reading", file, "...")
        with open(directory+file, "r") as f:
            lines = f.readlines()
            for x in lines:
                data.append(x)

    print("Collected", len(data), "pieces of data")

    print("Shuffling data...")
    random.shuffle(data)

    print("Merging data...")

    validationName = "validation_data.pickle"
    trainingName = "training_data.pickle"

    validation_ratio = 0.1
    num_validation = round(len(data) * validation_ratio)

    print("Creating", num_validation, "validation samples...")

    validation_data = []

    for i in range(num_validation):
        fen, eval = data[i].rstrip().split("; ")
        fen = fen.split(" ")[0]
        validation_data.append([fenToSparse(fen), int(eval)])

    with open(validationName, "wb") as f:
        pickle.dump(validation_data, f)

    print("Creating", len(data) - num_validation, "training samples...")

    training_data = []

    for i in range(num_validation, len(data)):
        fen, eval = data[i].rstrip().split("; ")
        fen = fen.split(" ")[0]
        training_data.append([fenToSparse(fen), int(eval)])

    with open(trainingName, "wb") as f:
        pickle.dump(training_data, f)

    print("Process complete.")

if __name__ == "__main__":
    main()
