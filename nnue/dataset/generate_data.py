"""Continuously generate self-play data"""

import multiprocessing
import os
import re
import requests
import sys
import time
from tqdm import tqdm
import zipfile

import huggingface_hub

from repo import REPO_ID, PATH_IN_REPO, REPO_TYPE

BASE_CONFIG = {
    "nodes": 25000,
    "positions": 1000000,
    "hash": 64,
    "maxply": 150,
    "evalbound": 8192,
}

CONFIGS = {
    "default": BASE_CONFIG
    | {
        "book": "None",
        "randomply": 6,
    },
    "noob3": BASE_CONFIG
    | {
        "book": "noob_3moves.epd",
        "randomply": 2,
    },
    "endgames": BASE_CONFIG
    | {
        "book": "endgames.epd",
        "randomply": 2,
    },
}


def get_book(book: str) -> bool:
    if book == "None":
        return True

    url = f"https://raw.githubusercontent.com/WillChing01/pingu-books/refs/heads/master/{book}"
    res = requests.get(url)

    retries = 0
    while retries < 3 and res.status_code != 200:
        time.sleep(3)
        res = requests.get(url)
        retries += 1

    if res.status_code != 200:
        return False

    with open(book, "wb") as f:
        f.write(res.content)

    return True


def progress_bar(total_positions: int, q: multiprocessing.Queue) -> None:
    n = 0
    with tqdm(total=total_positions, desc=f"Overall progress") as progress:
        for update in iter(q.get, None):
            progress.update(min(update, total_positions - n))
            n = min(n + update, total_positions)


def upload_file(config: dict, file_name: str, token: str) -> None:
    book_name = config["book"].split(".")[0]
    zip_name = f"{file_name.replace('.txt', '')}.zip"
    zipfile.ZipFile(
        zip_name, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ).write(file_name)

    retries = 0
    while retries < 3:
        try:
            huggingface_hub.upload_file(
                path_or_fileobj=zip_name,
                path_in_repo=f"/{PATH_IN_REPO}/{book_name}/{zip_name}",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
            )
            break
        except:
            print(f"Error uploading file - {file_name}")
            time.sleep(3)
            retries += 1

    os.remove(file_name)
    os.remove(zip_name)


def gensfen_worker(config: dict, q: multiprocessing.Queue, token: str) -> None:
    import engine

    cmd = f"Pingu.exe gensfen nodes {config['nodes']} positions {config['positions']} randomply {config['randomply']} maxply {config['maxply']} evalbound {config['evalbound']} hash {config['hash']} book {config['book']}"
    e = engine.Engine(name=cmd, path="\\..\\..\\")
    previous_n = 0
    while True:
        res = e.readline()
        if "Finished generating positions" in res:
            file_name = res.split(" - ")[1]
            upload_file(config, file_name, token)
            break
        elif "Finished game" in res:
            n = min(int(res.split("; ")[1].split(" ")[0]), config["positions"])
            q.put(n - previous_n)
            previous_n = n


def main():
    user_args = sys.argv[1:]
    if not re.search(r"^-N [1-9][0-9]* -P [1-9][0-9]* -T \S+$", " ".join(user_args)):
        print("error: incorrect format of args")
        print(
            "usage: generate_data.py -N <num_threads> -P <max_positions> -T <api_token>"
        )
        return
    elif int(user_args[1]) > multiprocessing.cpu_count():
        print("error: not enough cpu threads")
        return
    num_cpu = int(user_args[1])
    total_positions = int(user_args[3])
    token = user_args[5]

    sys.path.insert(1, f"{os.getcwd()}\\..\\..\\testing")

    try:
        huggingface_hub.login(token=token)
    except ValueError:
        print("Invalid token.")
        return

    for _, config in CONFIGS.items():
        if not get_book(config["book"]):
            print(f"error: could not download book - {book}")
            return

    q = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(num_cpu)
    results = [None for i in range(num_cpu)]
    total = 0

    progress = multiprocessing.Process(target=progress_bar, args=(total_positions, q))
    progress.start()

    counter = 0
    finished = False
    while not finished:
        for i in range(num_cpu):
            if results[i] is None or results[i].ready():
                if results[i] is not None:
                    total += BASE_CONFIG["positions"]
                if total < total_positions:
                    counter = (counter + 1) % len(CONFIGS)
                    config = CONFIGS[list(CONFIGS)[counter]]
                    results[i] = pool.apply_async(
                        gensfen_worker, args=(config, q, token)
                    )
                    time.sleep(5)
        if total >= total_positions:
            finished = all(x is None or x.ready() for x in results)
        time.sleep(0.5)

    q.put(None)
    progress.join()

    for _, config in CONFIGS.items():
        book = config["book"]
        if os.path.exists(book):
            os.remove(book)


if __name__ == "__main__":
    main()
