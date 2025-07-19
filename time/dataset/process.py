import os
import subprocess
import multiprocessing

DIR = "_preprocessed"
OUTPUT_DIR = "_processed"


def process(file):
    subprocess.run(["../../process-time-pgn.exe", "out", OUTPUT_DIR, "in", file])


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    files = [x.path for x in os.scandir(DIR) if x.is_file()]

    with multiprocessing.Pool() as pool:
        pool.map(process, files)


if __name__ == "__main__":
    main()
