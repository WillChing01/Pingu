import os
import shutil
import subprocess

CONFIG = {
    "download": {
        "cwd": f"{os.getcwd()}\\dataset",
        "dirs": ["_raw"],
        "args": ["python", "download.py"],
    },
    "unpack": {
        "cwd": f"{os.getcwd()}\\dataset",
        "args": ["python", "unpack.py"],
    },
    "parse": {
        "cwd": f"{os.getcwd()}\\dataset",
        "dirs": ["training", "validation"],
        "make": True,
        "args": ["parse.exe", "-N", "6"],
    },
    "train": {
        "cwd": f"{os.getcwd()}\\model",
        "dirs": ["checkpoints"],
        "make": True,
        "args": ["python", "train.py"],
    },
    "export": {
        "cwd": f"{os.getcwd()}\\model",
        "args": ["python", "export.py"],
    }
}


def main():
    for name, config in CONFIG.items():
        print(name)

        for dir in config.get("dirs", []):
            path = f"{config['cwd']}\\{dir}"
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.mkdir(path)

        if config.get("make"):
            for args in (["make", "clean"], ["make"]):
                subprocess.run(args, cwd=config["cwd"])

        subprocess.run(config["args"], cwd=config["cwd"], shell=True)

        if config.get("make"):
            subprocess.run(["make", "clean"], cwd=config["cwd"])


if __name__ == "__main__":
    main()
