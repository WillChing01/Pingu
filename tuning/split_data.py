import numpy as np
import os
import re
from tqdm import tqdm
from utils import DATASET_DTYPE

TRAINING_RATIO = 0.95

CHUNK_SIZE = 25000000

def main():
    DATASET_SHAPE = None
    for item in os.listdir(os.getcwd()):
        if DATASET_SHAPE := tuple(re.findall(r"^dataset_(\d+)_(\d+)\.dat$", item)):
            N = DATASET_SHAPE[0]
            break

    if DATASET_SHAPE is None:
        return

    rng = np.random.default_rng()

    is_validation = rng.random(N, dtype = np.float32).astype(np.float16) > TRAINING_RATIO
    num_validation = np.sum(is_validation)
    num_training = N - num_validation

    print("-------------------------")
    print("Training : Validation")
    print(num_training, ":", num_validation)
    print("Validation Ratio:", num_validation / N)
    print("-------------------------")

    print("Assigning chunks...")

    num_training_chunks = max(1, int(round(num_training / CHUNK_SIZE)))
    num_validation_chunks = max(1, int(round(num_validation / CHUNK_SIZE)))

    training_chunk_id = rng.integers(num_training_chunks, size = num_training, dtype = np.int16)
    validation_chunk_id = rng.integers(num_validation_chunks, size = num_validation, dtype = np.int16)

    training_chunk_sizes = np.bincount(training_chunk_id)
    validation_chunk_sizes = np.bincount(validation_chunk_id)

    print("Writing to chunks...")

    training_dir = os.getcwd() + "/training/"
    validation_dir = os.getcwd() + "/validation/"

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    training_memmaps = []
    training_indices = [0 for i in range(num_training_chunks)]
    training_index = 0

    validation_memmaps = []
    validation_indices = [0 for i in range(num_validation_chunks)]
    validation_index = 0

    for i in range(num_training_chunks):
        file_name = "chunk_" + str(i) + "_" + str(training_chunk_sizes[i]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
        training_memmaps.append(np.memmap(training_dir + file_name, mode = "w+", dtype = DATASET_DTYPE, shape = (training_chunk_sizes[i], DATASET_SHAPE[1])))
    for i in range(num_validation_chunks):
        file_name = "chunk_" + str(i) + "_" + str(validation_chunk_sizes[i]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
        validation_memmaps.append(np.memmap(validation_dir + file_name, mode = "w+", dtype = DATASET_DTYPE, shape = (validation_chunk_sizes[i], DATASET_SHAPE[1])))

    dataset_file = "dataset_" + str(DATASET_SHAPE[0]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
    dataset = np.memmap(dataset_file, mode = "r", dtype = DATASET_DTYPE, shape = DATASET_SHAPE)

    buffer_size = 10000

    for i in tqdm(range(N)):
        if i % buffer_size == 0:
            dataset._mmap.close()
            dataset = np.memmap(dataset_file, mode = "r", dtype = DATASET_DTYPE, shape = DATASET_SHAPE)
        if is_validation[i]:
            chunk_id = validation_chunk_id[validation_index]
            np.copyto(validation_memmaps[chunk_id][validation_indices[chunk_id]], dataset[i])

            if validation_indices[chunk_id] % buffer_size == 0:
                file_name = "chunk_" + str(chunk_id) + "_" + str(validation_chunk_sizes[chunk_id]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
                validation_memmaps[chunk_id]._mmap.close()
                validation_memmaps[chunk_id] = np.memmap(validation_dir + file_name, mode = "r+", dtype = DATASET_DTYPE, shape = (validation_chunk_sizes[chunk_id], DATASET_SHAPE[1]))

            validation_indices[chunk_id] += 1
            validation_index += 1
        else:
            chunk_id = training_chunk_id[training_index]
            np.copyto(training_memmaps[chunk_id][training_indices[chunk_id]], dataset[i])

            if training_indices[chunk_id] % buffer_size == 0:
                file_name = "chunk_" + str(chunk_id) + "_" + str(training_chunk_sizes[chunk_id]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
                training_memmaps[chunk_id]._mmap.close()
                training_memmaps[chunk_id] = np.memmap(training_dir + file_name, mode = "r+", dtype = DATASET_DTYPE, shape = (training_chunk_sizes[chunk_id], DATASET_SHAPE[1]))

            training_indices[chunk_id] += 1
            training_index += 1

    print("Process complete.")

    return 0

if __name__ == "__main__":
    main()
