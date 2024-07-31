import numpy as np
import os
import pickle
from tqdm import tqdm

N = 0
TRAINING_RATIO = 0.95

CHUNK_SIZE = 25000000

DATASET_DTYPE = np.short
DATASET_SHAPE = (N, 34)

def indicesToData(indices_file, output_file):
    with open(indices_file, 'rb') as f:
        indices = pickle.load(f)

    dataset_file = "dataset_" + str(DATASET_SHAPE[0]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
    dataset = np.memmap(dataset_file, mode = "r", dtype = DATASET_DTYPE, shape = DATASET_SHAPE)

    chunk = np.array(dataset[indices], copy = True)
    np.save(output_file, chunk)

    dataset._mmap.close()

    return None

def main():
    rng = np.random.default_rng()

    estimate_num_training = int(round(TRAINING_RATIO * N))
    estimate_num_validation = N - estimate_num_training

    num_training_chunks = max(1, int(round(estimate_num_training / CHUNK_SIZE)))
    num_validation_chunks = max(1, int(round(estimate_num_validation / CHUNK_SIZE)))

    print("Assigning indices to chunks...")

    chunk_indices = [[[] for i in range(num_training_chunks)], [[] for i in range(num_validation_chunks)]]
    is_validation = rng.choice(np.array([False, True], dtype = bool), N, p = [TRAINING_RATIO, 1. - TRAINING_RATIO])

    for i in tqdm(range(N)):
        chunk_choice = num_validation_chunks if is_validation[i] else num_training_chunks
        chunk_id = rng.integers(chunk_choice)
        chunk_indices[bool(is_validation[i])][chunk_id].append(i)

    temp_training_dir = os.getcwd() + "/temp-training/"
    temp_validation_dir = os.getcwd() + "/temp-validation/"

    if not os.path.exists(temp_training_dir):
        os.makedirs(temp_training_dir)
    if not os.path.exists(temp_validation_dir):
        os.makedirs(temp_validation_dir)

    for i in range(num_training_chunks):
        file_name = temp_training_dir + "chunk_indices_" + str(i) + ".pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(chunk_indices[0][i], f)

    for i in range(num_validation_chunks):
        file_name = temp_validation_dir + "chunk_indices_" + str(i) + ".pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(chunk_indices[1][i], f)

    num_validation = sum(is_validation)
    num_training = N - num_validation

    print("")
    print("-------------------------")
    print("Training : Validation")
    print(num_training, ":", num_validation)
    print("Validation Ratio:", num_validation / N)
    print("-------------------------")
    print("")

    del chunk_indices
    del is_validation

    print("Distributing data to chunks...")

    training_dir = os.getcwd() + "/training/"
    validation_dir = os.getcwd() + "/validation/"

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    training_files = []
    for (root, dir, files) in os.walk(temp_training_dir):
        training_files = files
        break

    for file in tqdm(training_files):
        chunk_index = file.split(".")[0].split("_")[-1]
        indicesToData(temp_training_dir + file, training_dir + "chunk_" + chunk_index)

    validation_files = []
    for (root, dir, files) in os.walk(temp_validation_dir):
        validation_files = files
        break

    for file in tqdm(validation_files):
        chunk_index = file.split(".")[0].split("_")[-1]
        indicesToData(temp_validation_dir + file, validation_dir + "chunk_" + chunk_index)

    print("Cleaning up temp files...")

    for file in training_files:
        os.remove(temp_training_dir + file)

    for file in validation_files:
        os.remove(temp_validation_dir + file)

    os.rmdir(temp_training_dir)
    os.rmdir(temp_validation_dir)

    print("Process complete.")

    return 0

if __name__ == "__main__":
    main()
