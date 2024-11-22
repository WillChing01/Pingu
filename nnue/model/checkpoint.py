import glob
import os

import torch

from config import CONFIG
from model import network


def get_files():
    return glob.glob(f"{CONFIG['path']}\\*.tar")


def get_epoch(file_name):
    return int(file_name.split("\\")[-1].split("_")[0])


def get_vloss(file_name):
    return float(
        file_name.split("\\")[-1].split(".tar")[0].split("_")[-1].replace(",", ".")
    )


def format_loss(loss):
    N_DIGITS = 10
    return str(round(loss, N_DIGITS)).replace(".", ",")


"""Load/Save Model"""


def load_model():
    start_epoch = 1
    model = network().to(CONFIG["device"])
    optimizer = CONFIG["optimizer"]["optim"](
        model.parameters(), **CONFIG["optimizer"]["kwargs"]
    )

    if model_files := get_files():
        latest_file = max(model_files, key=lambda x: get_epoch(x))
        checkpoint = torch.load(latest_file, weights_only=True)

        start_epoch = get_epoch(latest_file) + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (model, optimizer, start_epoch)


def save_model(model, optimizer, t_loss, v_loss):
    latest_epoch = 0
    if model_files := get_files():
        latest_epoch = max(get_epoch(x) for x in model_files)

    if not os.path.isdir(CONFIG["path"]):
        os.mkdir(CONFIG["path"])

    save_file = f"{CONFIG['path']}\\{latest_epoch+1}_tloss_{format_loss(t_loss)}_vloss_{format_loss(v_loss)}.tar"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_file,
    )


"""Early stopping"""


def early_stop():
    RECENT_LOOKBACK = 8
    DISTANT_LOOKBACK = 32

    model_files = get_files()

    if len(model_files) >= DISTANT_LOOKBACK:
        v_loss = [
            get_vloss(x)
            for x in sorted(model_files, key=lambda x: get_epoch(x), reverse=True)
        ]

        recent_loss = sum(v_loss[0:RECENT_LOOKBACK]) / RECENT_LOOKBACK
        distant_loss = sum(v_loss[0:DISTANT_LOOKBACK]) / DISTANT_LOOKBACK

        if recent_loss > distant_loss:
            return True

    return False


"""Find Best Model"""


def load_best():
    if model_files := get_files():
        best_file = min(model_files, key=lambda x: get_vloss(x))
        checkpoint = torch.load(best_file, weights_only=True)

        model = network().to("cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    return None
