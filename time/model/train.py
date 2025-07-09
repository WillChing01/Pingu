import torch
from ...pipeline.train import Trainer
from model import TimeNetwork


class TimeTrainer(Trainer):
    def __init__(self):
        super().__init__(
            "/checkpoints",
            "cuda",
            TimeNetwork,
            torch.optim.Adam,
            {},
            None,
        )

    def forward(self, datum):
        return self.model.forward(datum["tensor"], datum["scalar"])

    def custom_loss(self, output, datum):
        return torch.mean((output - datum["label"]) ** 2)


def main():
    TimeTrainer().train()


if __name__ == "__main__":
    main()
