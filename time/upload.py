import os
import re
import sys
import time
import zipfile

import huggingface_hub

REPO_ID = "WillChing01/time-management"
PATH_IN_REPO = "/lichess"
REPO_TYPE = "dataset"


def upload_file(file_name: str, token: str) -> None:
    zip_name = os.path.basename(file_name).replace(".csv", ".zip")
    zipfile.ZipFile(
        zip_name, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ).write(file_name, arcname=os.path.basename(file_name))

    retries = 0
    while retries < 3:
        try:
            huggingface_hub.upload_file(
                path_or_fileobj=zip_name,
                path_in_repo=f"{PATH_IN_REPO}/{zip_name}",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
            )
            os.remove(zip_name)
            break
        except:
            print(f"Error uploading file - {file_name}")
            time.sleep(3)
            retries += 1


def main():
    user_args = sys.argv[1:]
    if not re.search(r"^-T \S+$", " ".join(user_args)):
        print("error: incorrect format of args")
        print("usage: upload.py -T <api_token>")
        return
    token = user_args[1]

    try:
        huggingface_hub.login(token=token)
    except ValueError:
        print("Invalid token.")
        return

    DIR = "_processed"

    files = [x.path for x in os.scandir(DIR) if x.is_file() and x.path.endswith(".csv")]

    for file in files:
        upload_file(file, token)


if __name__ == "__main__":
    main()
