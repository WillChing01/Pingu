import os
import multiprocessing
import zipfile

DIR = f"{os.getcwd()}\\_raw"


def unpack(file):
    print(file)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(DIR)


def main():
    files = []
    for cwd, _, f in os.walk(DIR):
        for file in f:
            if file.endswith(".zip"):
                files.append(f"{cwd}\\{file}")

    with multiprocessing.Pool(processes=8) as pool:
        pool.map(unpack, files)


if __name__ == "__main__":
    main()
