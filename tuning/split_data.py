import numpy as np
import os
import pickle
import array
from tqdm import tqdm

N = 1180700000
TRAINING_RATIO = 0.95

CHUNK_SIZE = 25000000

DATASET_DTYPE = np.short
DATASET_SHAPE = (N, 34)

def main():
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

    for i in tqdm(range(N)):
        if i % 10000 == 0:
            dataset._mmap.close()
            dataset = np.memmap(dataset_file, mode = "r", dtype = DATASET_DTYPE, shape = DATASET_SHAPE)
        if is_validation[i]:
            chunk_id = validation_chunk_id[validation_index]
            validation_memmaps[chunk_id][validation_indices[chunk_id]] = np.array(dataset[i], copy = True)

            if validation_indices[chunk_id] % 1000 == 0:
                file_name = "chunk_" + str(chunk_id) + "_" + str(validation_chunk_sizes[chunk_id]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
                validation_memmaps[chunk_id]._mmap.close()
                validation_memmaps[chunk_id] = np.memmap(validation_dir + file_name, mode = "r+", dtype = DATASET_DTYPE, shape = (validation_chunk_sizes[chunk_id], DATASET_SHAPE[1]))

            validation_indices[chunk_id] += 1
            validation_index += 1
        else:
            chunk_id = training_chunk_id[training_index]
            training_memmaps[chunk_id][training_indices[chunk_id]] = np.array(dataset[i], copy = True)

            if training_indices[chunk_id] % 1000 == 0:
                file_name = "chunk_" + str(chunk_id) + "_" + str(training_chunk_sizes[chunk_id]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
                training_memmaps[chunk_id]._mmap.close()
                training_memmaps[chunk_id] = np.memmap(training_dir + file_name, mode = "r+", dtype = DATASET_DTYPE, shape = (training_chunk_sizes[chunk_id], DATASET_SHAPE[1]))

            training_indices[chunk_id] += 1
            training_index += 1

    print("Process complete.")

    return 0


# def indicesToData(indices_file, output_file):
#     with open(indices_file, 'rb') as f:
#         indices = pickle.load(f)

#     dataset_file = "dataset_" + str(DATASET_SHAPE[0]) + "_" + str(DATASET_SHAPE[1]) + ".dat"
#     dataset = np.memmap(dataset_file, mode = "r", dtype = DATASET_DTYPE, shape = DATASET_SHAPE)

#     chunk = np.array(dataset[indices], copy = True)
#     np.save(output_file, chunk)

#     dataset._mmap.close()

#     return None

# def main():
#     rng = np.random.default_rng()

#     estimate_num_training = int(round(TRAINING_RATIO * N))
#     estimate_num_validation = N - estimate_num_training

#     num_training_chunks = max(1, int(round(estimate_num_training / CHUNK_SIZE)))
#     num_validation_chunks = max(1, int(round(estimate_num_validation / CHUNK_SIZE)))

#     print("Assigning indices to chunks...")

#     chunk_indices = [[array.array('L') for i in range(num_training_chunks)], [array.array('L') for i in range(num_validation_chunks)]]
#     random_nums = rng.random(size = N, dtype = np.float32).astype(np.float16)
#     num_training = 0
#     num_validation = 0

#     for i in tqdm(range(N)):
#         is_validation = bool(random_nums[i] > TRAINING_RATIO)

#         if is_validation:
#             num_validation += 1
#         else:
#             num_training += 1

#         chunk_choice = num_validation_chunks if is_validation else num_training_chunks
#         chunk_id = rng.integers(chunk_choice)
#         chunk_indices[is_validation][chunk_id].append(i)

#     del random_nums

#     temp_training_dir = os.getcwd() + "/temp-training/"
#     temp_validation_dir = os.getcwd() + "/temp-validation/"

#     if not os.path.exists(temp_training_dir):
#         os.makedirs(temp_training_dir)
#     if not os.path.exists(temp_validation_dir):
#         os.makedirs(temp_validation_dir)

#     for i in range(num_training_chunks):
#         file_name = temp_training_dir + "chunk_indices_" + str(i) + ".pickle"
#         with open(file_name, 'wb') as f:
#             pickle.dump(chunk_indices[0][i], f)

#     for i in range(num_validation_chunks):
#         file_name = temp_validation_dir + "chunk_indices_" + str(i) + ".pickle"
#         with open(file_name, 'wb') as f:
#             pickle.dump(chunk_indices[1][i], f)

#     print("")
#     print("-------------------------")
#     print("Training : Validation")
#     print(num_training, ":", num_validation)
#     print("Validation Ratio:", num_validation / N)
#     print("-------------------------")
#     print("")

#     del chunk_indices

#     print("Distributing data to chunks...")

#     training_dir = os.getcwd() + "/training/"
#     validation_dir = os.getcwd() + "/validation/"

#     if not os.path.exists(training_dir):
#         os.makedirs(training_dir)
#     if not os.path.exists(validation_dir):
#         os.makedirs(validation_dir)

#     training_files = []
#     for (root, dir, files) in os.walk(temp_training_dir):
#         training_files = files
#         break

#     for file in tqdm(training_files):
#         chunk_index = file.split(".")[0].split("_")[-1]
#         indicesToData(temp_training_dir + file, training_dir + "chunk_" + chunk_index)

#     validation_files = []
#     for (root, dir, files) in os.walk(temp_validation_dir):
#         validation_files = files
#         break

#     for file in tqdm(validation_files):
#         chunk_index = file.split(".")[0].split("_")[-1]
#         indicesToData(temp_validation_dir + file, validation_dir + "chunk_" + chunk_index)

#     print("Cleaning up temp files...")

#     for file in training_files:
#         os.remove(temp_training_dir + file)

#     for file in validation_files:
#         os.remove(temp_validation_dir + file)

#     os.rmdir(temp_training_dir)
#     os.rmdir(temp_validation_dir)

#     print("Process complete.")

#     return 0

if __name__ == "__main__":
    main()
