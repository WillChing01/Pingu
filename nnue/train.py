import contextlib
import glob
import os

import torch
from torch import nn
from tqdm import tqdm

from dataloader import NUM_FEATURES, dll


"""Dataloader"""

BATCH_SIZE = 1024
NUM_WORKERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATALOADER_CONFIGS = {
    "training": {
        "path": bytes(f"{os.getcwd()}\\dataset\\training", "utf-8"),
    },
    "validation": {
        "path": bytes(f"{os.getcwd()}\\dataset\\validation", "utf-8"),
    },
}


class DataLoader:
    def __init__(self, kind):
        self.path = DATALOADER_CONFIGS[kind]["path"]
        self.length = dll.length(self.path)

    def __len__(self):
        return self.length

    def iterator(self):
        dataLoader = dll.constructDataLoader(self.path, BATCH_SIZE, NUM_WORKERS)

        while batch := dll.getBatch(dataLoader):
            yield batch.contents.reformat(DEVICE)
            dll.destructBatch(batch)

        dll.destructDataLoader(dataLoader)


"""Model definition: (45056 -> 64 -> cReLU(64)) x 2 -> 1"""

INPUT_COUNT = NUM_FEATURES
L1_COUNT = 64
OUTPUT_COUNT = 1


class ClippedReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.mul(x, self.factor)


class PerspectiveNetwork(nn.Module):
    def __init__(self, input_count, output_count):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_count, output_count), ClippedReLU())

    def forward(self, x):
        return self.net(x)


class HalfKaNetwork(nn.Module):
    def __init__(self, input_count, l1_count, output_count):
        super().__init__()

        self.perspectiveNets = [
            PerspectiveNetwork(input_count, l1_count),
            PerspectiveNetwork(input_count, l1_count),
        ]

        self.net = nn.Sequential(nn.Linear(2 * l1_count, output_count), Scale(512))

        self.perspectiveNets[0].apply(self.init_weights)
        self.perspectiveNets[1].apply(self.init_weights)
        self.net.apply(self.init_weights)

    def init_weights(self, x):
        if type(x) == nn.Linear:
            torch.nn.init.kaiming_normal_(x.weight, nonlinearity="relu")
            torch.nn.init.zeros_(x.bias)

    def forward(self, x, y):
        return self.net(
            torch.cat(
                (
                    self.perspectiveNets[0](x),
                    self.perspectiveNets[1](y),
                )
            )
        )


"""Training"""

MAX_EPOCHS = 10000
OPTIMIZER = torch.optim.Adam


def custom_loss(output, targetEval, targetResult):
    K = 0.00475
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
            if kind == "training":
                optimizer.zero_grad()

            output = model(x, y)

            l = custom_loss(output, evals, results)
            batch_size = evals.size(0)
            loss += l.item() * batch_size / length

            if kind == "training":
                l.backward()
                optimizer.step()

            progress.update(batch_size)

    progress.close()
    print(f"{kind.capitalize()} Loss: {loss:>8f}")
    return loss


"""Load/Save Model"""

MODEL_PATH = f"{os.getcwd()}\\models"


def format_loss(loss):
    N_DIGITS = 10
    return str(round(loss, N_DIGITS)).replace(".", ",")


def load_model():
    start_epoch = 1
    model = HalfKaNetwork(INPUT_COUNT, L1_COUNT, OUTPUT_COUNT)

    if model_files := glob.glob(f"{MODEL_PATH}\\*.pth"):
        latest_file = max(model_files, key=lambda x: int(x.split("_")[0]))
        start_epoch = int(latest_file.split("_")[0]) + 1
        model.load_state_dict(torch.load(latest_file))

    optimizer = OPTIMIZER(model.parameters())

    return (model, optimizer, start_epoch)


def save_model(model, t_loss, v_loss):
    latest_epoch = 0
    if model_files := glob.glob(f"{MODEL_PATH}\\*.pth"):
        latest_epoch = max(int(x.split("_")[0]) for x in model_files)

    save_file = (
        f"{latest_epoch+1}_tloss_{format_loss(t_loss)}_vloss_{format_loss(v_loss)}.pth"
    )
    torch.save(model.state_dict(), save_file)


"""Early stopping"""


def early_stop():
    RECENT_LOOKBACK = 4
    DISTANT_LOOKBACK = 16

    if (
        model_files := glob.glob(f"{MODEL_PATH}\\*.pth")
        and len(model_files) >= DISTANT_LOOKBACK
    ):
        format = lambda x: float(x.split(".")[0].split("_")[-1].replace(",", "."))
        v_loss = [
            format(x)
            for x in sorted(
                model_files, key=lambda x: int(x.split("_")[0]), reverse=True
            )
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

        t_loss = run_epoch(model, "training", {"optimizer": optimizer})
        v_loss = run_epoch(model, "validation")

        save_model(model, t_loss, v_loss)

        if early_stop():
            return


if __name__ == "__main__":
    main()
