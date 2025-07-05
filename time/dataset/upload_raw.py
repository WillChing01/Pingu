import sys
import re
import huggingface_hub
import time


def upload_file(
    file_name: str,
    name: str,
    path_in_repo: str,
    repo_id: str,
    repo_type: str,
    token: str,
) -> None:
    retries = 0
    while retries < 3:
        try:
            huggingface_hub.upload_file(
                path_or_fileobj=file_name,
                path_in_repo=f"{path_in_repo}/{name}",
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )
            break
        except:
            print(f"Error uploading file - {file_name}")
            time.sleep(3)
            retries += 1


def validate_login(token):
    try:
        huggingface_hub.login(token=token)
        return True
    except Exception as e:
        print(e)
        return False


REPO_ID = "WillChing01/time-management"
PATH_IN_REPO = "lichess_raw"
REPO_TYPE = "dataset"


def main():
    user_args = sys.argv[1:]
    if not re.search(r"^\S+ -T \S+$", " ".join(user_args)):
        print("error: incorrect format of args")
        print("usage: upload_raw.py -T <api_token>")

    token = user_args[2]
    if not validate_login(token):
        return

    file = user_args[0]
    upload_file(file, file.split("/")[-1], PATH_IN_REPO, REPO_ID, REPO_TYPE, token)


if __name__ == "__main__":
    main()
