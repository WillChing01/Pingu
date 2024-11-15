import glob

from config import MODEL_PATH


def get_files():
    return glob.glob(f"{MODEL_PATH}\\*.tar")


def get_epoch(file_name):
    return int(file_name.split("\\")[-1].split("_")[0])


def get_vloss(file_name):
    return float(
        file_name.split("\\")[-1].split(".")[0].split("_")[-1].replace(",", ".")
    )


def format_loss(loss):
    N_DIGITS = 10
    return str(round(loss, N_DIGITS)).replace(".", ",")
