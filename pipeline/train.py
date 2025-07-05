import contextlib

import torch
from tqdm import tqdm

from checkpoint import Checkpoint


class Trainer(Checkpoint):
    MAX_EPOCHS = 10000

    def __init__(
        self,
        checkpoint_path,
        device,
        model_class,
        optimizer_class,
        optimizer_kwargs,
        dataloader_class,
    ):
        super().__init__(
            checkpoint_path, device, model_class, optimizer_class, optimizer_kwargs
        )

        self.dataloader_class = dataloader_class
        self.model, self.optimizer, self.start_epoch = self.load_model()

    def forward(self, datum):
        raise NotImplementedError

    def custom_loss(self, output, datum):
        raise NotImplementedError

    def run_epoch(self, kind):
        if kind == "training":
            self.model.train()
            context = contextlib.nullcontext()
        elif kind == "validation":
            self.model.eval()
            context = torch.no_grad()
        else:
            raise ValueError("kind must be one of [training, validation]")

        loss = 0

        dataloader = self.dataloader_class(kind)
        length = len(dataloader)

        progress = tqdm(total=length)
        counter = 0

        with context:
            for batch_size, datum in dataloader.iterator():
                output = self.forward(datum)
                l = self.custom_loss(output, datum)
                _loss = l.item()

                if kind == "training":
                    self.optimizer.zero_grad(set_to_none=True)
                    l.backward()
                    self.optimizer.step()
                    if hasattr(self.model, "clamp"):
                        self.model.clamp()

                loss += _loss * batch_size / length
                progress.update(batch_size)
                counter += 1
                if counter % 100 == 0:
                    progress.set_description(f"Loss: {_loss:>8f}")

        progress.set_description(None)
        progress.close()
        print(f"{kind.capitalize()} Loss: {loss:>8f}")
        return loss

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.MAX_EPOCHS):
            print(f"Epoch {epoch}\n-------------------------------")

            t_loss = self.run_epoch("training")
            v_loss = self.run_epoch("validation")

            self.save_model(self.model, self.optimizer, t_loss, v_loss)

            if self.early_stop():
                return
