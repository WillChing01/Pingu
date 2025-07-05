import os
import pandas as pd
import random
from tqdm import tqdm

DIR = "_processed"

TRAINING_RATIO = 0.95

TRAINING_CHUNKS = 14
VALIDATION_CHUNKS = 1


HEADERS = "fen,isDraw,isWin,ply,totalPly,qSearch,inCheck,increment,timeLeft,timeSpent,totalTimeSpent,startTime,opponentTime"


def write_headers():
    for chunk in range(TRAINING_CHUNKS):
        with open(f"training/chunk_{chunk}.csv", "w") as f:
            f.write(HEADERS + "\n")

    for chunk in range(VALIDATION_CHUNKS):
        with open(f"validation/chunk_{chunk}.csv", "w") as f:
            f.write(HEADERS + "\n")


def count_rows(csv):
    chunk_size = 256 * 1024 * 1024
    total = 0
    with open(csv, "rb") as f:
        while chunk := f.read(chunk_size):
            total += chunk.count(b"\n")
    return total - 1


def main():
    for d in ["training", "validation"]:
        os.makedirs(d, exist_ok=True)

    write_headers()

    training_writers = [
        open(f"training/chunk_{i}.csv", "a") for i in range(TRAINING_CHUNKS)
    ]
    validation_writers = [
        open(f"validation/chunk_{i}.csv", "a") for i in range(VALIDATION_CHUNKS)
    ]

    for file in [x.path for x in os.scandir(DIR) if x.is_file()]:
        total_rows = count_rows(file)
        chunksize = 100000
        total_chunks = total_rows // chunksize + bool(total_rows % chunksize)
        reader = pd.read_csv(file, chunksize=chunksize)
        for chunk in tqdm(
            reader, total=total_chunks, desc=os.path.basename(file), leave=True
        ):
            rows = chunk.to_csv(index=False, header=False).splitlines()
            for row in rows:
                if random.random() < TRAINING_RATIO:
                    idx = random.randint(0, TRAINING_CHUNKS - 1)
                    training_writers[idx].write(row + "\n")
                else:
                    idx = random.randint(0, VALIDATION_CHUNKS - 1)
                    validation_writers[idx].write(row + "\n")
        os.remove(file)

    for writer in training_writers + validation_writers:
        writer.close()


if __name__ == "__main__":
    main()
