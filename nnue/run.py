import os
import subprocess

CONFIG = {
    "download": {
        "dir": f"{os.getcwd()}\\dataset",
        "cmd": (
            ("cmd", "/c", "del", "/Q", "/S", "\\_raw"),
            ("python", "download.py", "-d", "_raw"),
        ),
    },
    "parse": {
        "dir": f"{os.getcwd()}\\dataset",
        "cmd": (
            ("cmd", "/c", "del", "/Q", "/S", "\\training"),
            ("cmd", "/c", "del", "/Q", "/S", "\\validation"),
            ("make", "clean"),
            ("make",),
            ("parse.exe", "-N", "6"),
            ("make", "clean"),
        ),
    },
    "train": {
        "dir": f"{os.getcwd()}\\model",
        "cmd": (
            ("cmd", "/c", "del", "/Q", "/S", "\\checkpoints"),
            ("make", "clean"),
            ("make",),
            ("python", "train.py"),
            ("make", "clean"),
            ("python", "export.py"),
        ),
    },
}


def main():
    for process in ("download", "parse", "train"):
        config = CONFIG[process]
        for x in config["cmd"]:
            subprocess.run(x, cwd=config["dir"])


if __name__ == "__main__":
    main()
