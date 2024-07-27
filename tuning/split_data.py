import numpy as np
import os

TRAINING_RATIO = 0.95

def main():
    files = []
    for (root, dir, f) in os.walk(os.getcwd()):
        files = f
        break

    sparse_input_file = None
    labels_file = None
    n = 0

    for file in files:
        if file[-4:] != ".dat":
            continue
        parsed = file[:-4].split("_")
        if parsed[:2] == ["sparse", "input"]:
            sparse_input_file = file
            n = int(parsed[-2])
        elif parsed[0] == "labels":
            labels_file = file
            n = int(parsed[-2])

    if sparse_input_file is None or labels_file is None:
        print("Files not found, run parse_data.py first.")
        return None

    print("Generating and shuffling indices...")

    dtype = np.uint32 if n-1 <= np.iinfo(np.uint32).max else np.uint64
    indices = np.arange(n, dtype = dtype)

    rng = np.random.default_rng()
    rng.shuffle(indices)

    print("Splitting indices between training and validation sets...")

    training_num = int(round(n * TRAINING_RATIO))

    print("Generating", training_num, "training data.")
    print("Generating", n - training_num, "validation data.")

    np.save('training_indices', indices[:training_num])
    np.save('validation_indices', indices[training_num:])

    print("Process complete.")

    return 0

if __name__ == "__main__":
    main()
