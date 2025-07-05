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
        return None

    def custom_loss(self, output, datum):
        return None


def main():
    TimeTrainer().train()


if __name__ == "__main__":
    main()
