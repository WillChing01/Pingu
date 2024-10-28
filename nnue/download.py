import huggingface_hub
import os
import sys
from utils import REPO_ID, REPO_TYPE


def main(args):
    directory = f"{os.getcwd()}\\{args[1]}"
    huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        cache_dir=f"\\\\?\\{directory}",
        local_dir=f"\\\\?\\{directory}",
        allow_patterns="*.txt",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
