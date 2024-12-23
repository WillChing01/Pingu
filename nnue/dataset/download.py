import os

import huggingface_hub

from repo import REPO_ID, REPO_TYPE

DIR = f"{os.getcwd()}\\_raw"


def main():
    huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        cache_dir=f"\\\\?\\{DIR}",
        local_dir=f"\\\\?\\{DIR}",
        allow_patterns="*.zip",
    )


if __name__ == "__main__":
    main()
