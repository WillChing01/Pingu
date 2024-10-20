"""Parse and format data in datasets folder"""

import os
import re
import sys
import numpy as np
import multiprocessing
import huggingface_hub
from parallelbar import progress_starmap
from utils import REPO_ID, REPO_TYPE, DATASET_DTYPE


def download_files(token: str, directory: str) -> None:
    huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        cache_dir=f"\\\\?\\{directory}",
        local_dir=f"\\\\?\\{directory}",
        token=token,
    )


def parse_file(start_index, file_name, dataset_file, dataset_shape):
    piece_types = {y: x for x, y in enumerate("KkQqRrBbNnPp")}
    side_mapping = {y: x for x, y in enumerate("wb")}

    index = 0
    with open(file_name, "r") as f:
        lines = f.readlines()
        length = len(lines)

        data_chunk = np.full((length, 35), -1, dtype=DATASET_DTYPE)

        for line in lines:
            fen, eval_, result = line.rstrip().split("; ")
            fen, side_to_move = fen.split(" ")[0:2]

            side_to_move = side_mapping[side_to_move]
            eval_ = int(eval_)
            result = round(2 * float(result))

            if side_to_move == 1:
                eval_ = -eval_
                result = 2 - result

            data_chunk[index][[-3, -2, -1]] = side_to_move, eval_, result

            i = 0
            square = 56
            for x in fen:
                if x == "/":
                    square -= 16
                elif x.isdigit():
                    square += int(x)
                else:
                    data_chunk[index][i] = 64 * piece_types[x] + square
                    if x == "K":
                        data_chunk[index][[0, i]] = data_chunk[index][[i, 0]]
                    elif x == "k":
                        data_chunk[index][[1, i]] = data_chunk[index][[i, 1]]
                    i += 1
                    square += 1

            index += 1

    dataset = np.memmap(
        dataset_file, mode="r+", dtype=DATASET_DTYPE, shape=dataset_shape
    )

    dataset[start_index : start_index + length] = data_chunk

    dataset.flush()
    dataset._mmap.close()


def main():
    user_args = sys.argv[1:]
    if not re.search(r"^-N [1-9][0-9]* -T \S+$", " ".join(user_args)):
        print("error: incorrect format of args")
        print("usage: parse_data.py -N <num_threads> -T <api_token>")
        return None
    elif int(user_args[1]) > multiprocessing.cpu_count():
        print("error: not enough cpu threads")
        return None
    num_cpu = int(user_args[1])
    token = user_args[3]

    try:
        huggingface_hub.login(token=token)
    except ValueError:
        print("Invalid token.")
        return

    directory = f"{os.getcwd()}\\datasets\\"

    if not os.path.isdir(directory):
        os.mkdir(directory)

    if not os.listdir(directory):
        download_files(token, directory)

    file_data = []
    n = 0

    for root, _, files in os.walk(directory):
        if len(files) == 0:
            continue
        for file in files:
            if file.split(".")[-1] != "txt":
                continue
            file_data.append((n, f"{root}/{file}"))
            n += int(re.findall(r"_n([1-9][0-9]*)_", file)[0])

    print("Found", n, "pieces of data in", len(file_data), "files.")

    dataset_file = "dataset_" + str(n) + "_35.dat"
    dataset_shape = (n, 35)

    try:
        dataset = np.memmap(
            dataset_file, mode="r", dtype=DATASET_DTYPE, shape=dataset_shape
        )
        dataset._mmap.close()
    except:
        dataset = np.memmap(
            dataset_file, mode="w+", dtype=DATASET_DTYPE, shape=dataset_shape
        )
        dataset._mmap.close()

    args = [(x, y, dataset_file, dataset_shape) for x, y in file_data]

    progress_starmap(parse_file, args, n_cpu=num_cpu)

    os.rmdir(directory)


if __name__ == "__main__":
    main()
