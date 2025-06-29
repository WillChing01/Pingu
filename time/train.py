import contextlib

import torch
from tqdm import tqdm

import dataloader
from model import network

MAX_EPOCHS = 10000


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

    d = dataloader.dataloader(kind)
    length = (
        dataloader.TRAINING_LENGTH
        if kind == "training"
        else dataloader.VALIDATION_LENGTH
    )

    progress = tqdm(total=length)
    counter = 0

    with context:
        for batch_size, cnn_inputs, scalar_inputs, labels in d:
            output = model(
                cnn_inputs.to(device="cuda"), scalar_inputs.to(device="cuda")
            )
            l = torch.mean((labels.to(device="cuda") - output) ** 2)
            _loss = l.item()

            if kind == "training":
                optimizer.zero_grad(set_to_none=True)
                l.backward()
                optimizer.step()

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
    # model, optimizer, start_epoch = load_model()
    _model = network().to(device="cuda")
    start_epoch = 1
    optimizer = torch.optim.Adam(_model.parameters())

    for epoch in range(start_epoch, start_epoch + MAX_EPOCHS):
        print(f"Epoch {epoch}\n-------------------------------")

        t_loss = run_epoch(_model, "training", **{"optimizer": optimizer})
        v_loss = run_epoch(_model, "validation")

        # save_model(model, optimizer, t_loss, v_loss)

        # if early_stop():
        #     return


if __name__ == "__main__":
    main()
