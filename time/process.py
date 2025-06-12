import os
import subprocess

DIR = "_preprocessed"
OUTPUT_DIR = "_processed"


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    files = [x.path for x in os.scandir(DIR) if x.is_file()]

    subprocess.run(
        ["../Pingu.exe", "process-time-pgn", "out", OUTPUT_DIR, "in", *files]
    )


if __name__ == "__main__":
    main()
