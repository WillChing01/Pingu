import contextlib

import torch
from tqdm import tqdm

from checkpoint import load_model, save_model, early_stop
from dataloader import DataLoader

"""Training"""

MAX_EPOCHS = 10000


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
    counter = 0

    with context:
        for batch_size, (x, evals, results) in dataLoader.iterator():
            output = model(x)
            l = custom_loss(output, evals, results)
            _loss = l.item()

            if kind == "training":
                optimizer.zero_grad(set_to_none=True)
                l.backward()
                optimizer.step()
                model.clamp()

            loss += _loss * batch_size / length
            progress.update(batch_size)
            counter += 1
            if counter % 100 == 0:
                progress.set_description(f"Loss: {_loss:>8f}")

    progress.set_description(None)
    progress.close()
    print(f"{kind.capitalize()} Loss: {loss:>8f}")
    return loss


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
