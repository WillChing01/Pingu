import contextlib
import glob
import os

import torch
from tqdm import tqdm

from dataloader import DEVICE, DataLoader
from model import INPUT_COUNT, L1_COUNT, OUTPUT_COUNT, HalfKaNetwork

"""Training"""

MAX_EPOCHS = 10000
OPTIMIZER = torch.optim.Adam


def custom_loss(output, targetEval, targetResult):
    K = 1 / 400
    GAMMA = 0.5
    output_scaled = torch.sigmoid(K * output)
    target_scaled = GAMMA * torch.sigmoid(K * targetEval) + (1.0 - GAMMA) * targetResult
    return torch.mean((output_scaled - target_scaled) ** 2)


def run_epoch(model, kind, **kwargs):
    if kind == "training":
        model.train()
        optimizer = kwargs["optimizer"]
        context = contextlib.nullcontext()
    elif kind == "validation":
        model.eval()
        context = torch.no_grad()
    else:
        raise ValueError("kind must be one of [training, validation]")

    loss = 0

    dataLoader = DataLoader(kind)
    length = len(dataLoader)

    progress = tqdm(total=length)

    with context:
        for x, y, evals, results in dataLoader.iterator():
            output = model((x, y))

            l = custom_loss(output, evals, results)
            batch_size = evals.size(0)
            loss += l.item() * batch_size / length

            if kind == "training":
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                model.clamp()

            progress.update(batch_size)

    progress.close()
    print(f"{kind.capitalize()} Loss: {loss:>8f}")
    return loss


"""Load/Save Model"""

MODEL_PATH = f"{os.getcwd()}\\checkpoints"


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


def load_model():
    start_epoch = 1
    model = HalfKaNetwork(INPUT_COUNT, L1_COUNT, OUTPUT_COUNT).to(DEVICE)
    optimizer = OPTIMIZER(model.parameters())

    if model_files := get_files():
        latest_file = max(model_files, key=lambda x: get_epoch(x))
        checkpoint = torch.load(latest_file)

        start_epoch = get_epoch(latest_file) + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (model, optimizer, start_epoch)


def save_model(model, optimizer, t_loss, v_loss):
    latest_epoch = 0
    if model_files := get_files():
        latest_epoch = max(get_epoch(x) for x in model_files)

    save_file = f"{MODEL_PATH}\\{latest_epoch+1}_tloss_{format_loss(t_loss)}_vloss_{format_loss(v_loss)}.tar"
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


def main():
    model, optimizer, start_epoch = load_model()

    for epoch in range(start_epoch, start_epoch + MAX_EPOCHS):
        print(f"Epoch {epoch}\n-------------------------------")

        t_loss = run_epoch(model, "training", **{"optimizer": optimizer})
        v_loss = run_epoch(model, "validation")

        save_model(model, optimizer, t_loss, v_loss)

        if early_stop():
            return


if __name__ == "__main__":
    main()
