import os
import requests
import subprocess
import multiprocessing

DIR = "_raw"
OUTPUT_DIR = "_preprocessed"


def download_exe():
    url = "https://www.cs.kent.ac.uk/~djb/pgn-extract/pgn-extract.exe"
    with requests.get(url) as res:
        res.raise_for_status()
        with open("pgn-extract.exe", "wb") as f:
            f.write(res.content)
            print("Downloaded pgn-extract")


def preprocess(file):
    args = [
        "pgn-extract",
        f"{DIR}/{file}",
        "-o",
        f"{OUTPUT_DIR}/{file}",
        "-Wlalg",
        "--nomovenumbers",
        "-w",
        "999999",
        "-t",
        "tags.txt",
        "-s",
        "--nonags",
    ]
    subprocess.run(args, cwd=os.getcwd())


def preprocess_files():
    with multiprocessing.Pool() as pool:
        pool.map(preprocess, [x for x in os.listdir(DIR) if x.endswith(".pgn")])


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    download_exe()
    preprocess_files()


if __name__ == "__main__":
    main()
