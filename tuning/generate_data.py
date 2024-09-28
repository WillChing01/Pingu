"""Continuously generate self-play data"""

import os
import re
import sys
import time
from tqdm import tqdm
import multiprocessing
import huggingface_hub

HASH = 64

MINDEPTH = 8
MAXDEPTH = 12
POSITIONS = 100000
RANDOMPLY = 4
MAXPLY = 150
EVALBOUND = 8192
BOOK = "noob_3moves.epd"

REPO_ID = "WillChing01/pingu"
PATH_IN_REPO = "Pingu_4.0.0"
REPO_TYPE = "dataset"

def gensfen_worker(position: int, token: str) -> None:
    import engine
    cmd = f"Pingu.exe gensfen mindepth {MINDEPTH} maxdepth {MAXDEPTH} positions {POSITIONS} randomply {RANDOMPLY} maxply {MAXPLY} evalbound {EVALBOUND} hash {HASH} book {BOOK}"
    e = engine.Engine(name=cmd, path="\\..\\")
    previous_n = 0
    with tqdm(total=POSITIONS, position=position, desc=f"Worker {position}", leave=False) as progress:
        while True:
            res = e.readline()
            if "Finished generating positions" in res:
                fileName = res.split(" - ")[1]
                try:
                    huggingface_hub.upload_file(
                        path_or_fileobj=fileName,
                        path_in_repo=f"/{PATH_IN_REPO}/{fileName}",
                        repo_id=REPO_ID,
                        repo_type=REPO_TYPE,
                        token=token
                    )
                    os.remove(fileName)
                except:
                    print(f"Error uploading file - {fileName}")
                progress.update(POSITIONS - previous_n)
                break
            elif "Finished game" in res:
                n = min(int(res.split("; ")[1].split(" ")[0]), POSITIONS)
                progress.update(n - previous_n)
                previous_n = n

def main():
    user_args = sys.argv[1:]
    if not re.search("^-N [1-9][0-9]* -P [1-9][0-9]* -T \S+$", " ".join(user_args)):
        print("error: incorrect format of args")
        print("usage: generate_data.py -N <num_threads> -P <max_positions> -T <api_token>")
        return
    elif int(user_args[1]) > multiprocessing.cpu_count():
        print("error: not enough cpu threads")
        return
    num_cpu = int(user_args[1])
    total_positions = int(user_args[3])
    token = user_args[5]

    sys.path.insert(1, os.getcwd() + "\\..\\testing\\")

    try:
        huggingface_hub.login(token=token)
    except ValueError:
        print("Invalid token.")
        return

    pool = multiprocessing.Pool(num_cpu)
    result = [None for i in range(num_cpu)]
    total = 0

    with tqdm(total=total_positions, position=num_cpu, desc=f"Overall progress") as progress:
        while True:
            for i in range(num_cpu):
                if result[i] is None or result[i].ready():
                    if result[i] is not None:
                        total += POSITIONS
                        progress.update(max(min(POSITIONS, total_positions - total), 0))
                        if total >= total_positions:
                            return
                    result[i] = pool.apply_async(gensfen_worker, args=(i,token))
                    time.sleep(5)
            time.sleep(0.5)

if __name__ == "__main__":
    main()
