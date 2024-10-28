"""Parse and format data in datasets folder"""

import os
import re
import sys
import numpy as np
import multiprocessing
import huggingface_hub
from tqdm import tqdm
from utils import REPO_ID, REPO_TYPE, DATASET_DTYPE


PIECE_TYPES: dict[str, int] = {y: x for x, y in enumerate("KkQqRrBbNnPp")}
SIDE_MAPPING: dict[str, int] = {y: x for x, y in enumerate("wb")}


def download_files(token: str, directory: str) -> None:
    huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        cache_dir=f"\\\\?\\{directory}",
        local_dir=f"\\\\?\\{directory}",
        token=token,
    )


def progress_bar(total: int, q: multiprocessing.Queue) -> None:
    with tqdm(total=total, desc="Overall progress") as progress:
        update: int
        for update in iter(q.get, None):
            progress.update(update)


def save_chunks(
    dataset_file: str, dataset_shape: tuple[int, int], q: multiprocessing.Queue
) -> None:
    start_index: int
    for start_index, chunk in iter(q.get, None):
        dataset = np.memmap(
            dataset_file, mode="r+", dtype=DATASET_DTYPE, shape=dataset_shape
        )
        dataset[start_index : start_index + chunk.shape[0]] = chunk
        dataset.flush()
        dataset._mmap.close()


def parse_file(
    start_index: int,
    file_name: str,
    q: multiprocessing.Queue,
    progress_q: multiprocessing.Queue,
) -> None:
    length: int = int(re.findall(r"_n([1-9][0-9]*)_", file_name)[0])
    chunk = np.full((length, 35), -1, dtype=DATASET_DTYPE)
    with open(file_name, "r") as f:
        index: int
        line: str
        for index, line in enumerate(f):
            fen: str
            eval_: str
            result: str
            side_to_move: str
            fen, eval_, result = line.rstrip().split("; ")
            fen, side_to_move = fen.split(" ")[0:2]

            parsed_side_to_move: int = SIDE_MAPPING[side_to_move]
            parsed_eval: int = int(eval_)
            parsed_result: int = round(2 * float(result))

            if side_to_move == 1:
                parsed_eval = -parsed_eval
                parsed_result = 2 - parsed_result

            chunk[index][[-3, -2, -1]] = parsed_side_to_move, parsed_eval, parsed_result

            i: int = 0
            square: int = 56
            x: str
            for x in fen:
                if x == "/":
                    square -= 16
                elif x.isdigit():
                    square += int(x)
                else:
                    chunk[index][i] = 64 * PIECE_TYPES[x] + square
                    if x == "K":
                        chunk[index][[0, i]] = chunk[index][[i, 0]]
                    elif x == "k":
                        chunk[index][[1, i]] = chunk[index][[i, 1]]
                    i += 1
                    square += 1

    q.put((start_index, chunk))
    progress_q.put(length)


def main(args):
    if not re.search(r"^-N [1-9][0-9]* -T \S+$", " ".join(args)):
        print("error: incorrect format of args")
        print("usage: parse_data.py -N <num_threads> -T <api_token>")
        return None
    elif int(args[1]) > multiprocessing.cpu_count():
        print("error: not enough cpu threads")
        return None
    num_cpu = int(args[1])
    token = args[3]

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

    q = multiprocessing.Manager().Queue()
    chunk_saver = multiprocessing.Process(
        target=save_chunks, args=(dataset_file, dataset_shape, q)
    )
    chunk_saver.start()

    progress_q = multiprocessing.Manager().Queue()
    progress = multiprocessing.Process(target=progress_bar, args=(n, progress_q))
    progress.start()

    args = [(x, y, q, progress_q) for x, y in file_data]
    pool = multiprocessing.Pool(num_cpu)
    pool.starmap(parse_file, args)

    q.put(None)
    chunk_saver.join()

    progress_q.put(None)
    progress.join()

    os.rmdir(directory)


if __name__ == "__main__":
    main(sys.argv[1:])
