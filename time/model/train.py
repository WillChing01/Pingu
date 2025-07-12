import torch

import os
import subprocess
import sys

sys.path.insert(0, os.getcwd() + "\\..\\..\\")

from pipeline.train import Trainer
from dataloader import DataLoader
from model import TimeNetwork


class TimeTrainer(Trainer):
    def __init__(self):
        super().__init__(
            os.getcwd() + "\\checkpoints",
            "cuda",
            TimeNetwork,
            torch.optim.Adam,
            {},
            DataLoader,
        )

    def forward(self, datum):
        return self.model.forward(datum["tensor"], datum["scalar"])

    def custom_loss(self, output, datum):
        return torch.mean((output - datum["label"]) ** 2)


def main():
    # subprocess.run(["make", "clean"], cwd=os.getcwd())
    # subprocess.run(["make"], cwd=os.getcwd())

    TimeTrainer().train()


if __name__ == "__main__":
    main()
