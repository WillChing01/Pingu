import torch

import os
import subprocess
import sys

sys.path.insert(0, os.getcwd() + "\\..\\..\\")

from pipeline.train import Trainer
from dataloader import DataLoader
from model import SimpleTimeNetwork


class TimeTrainer(Trainer):
    def __init__(self):
        super().__init__(
            os.getcwd() + "\\checkpoints",
            "cuda",
            SimpleTimeNetwork,
            torch.optim.Adam,
            {"lr": 0.0001},
            DataLoader,
        )

    def forward(self, datum):
        assert torch.isfinite(datum["tensor"]).all(), "Non-finite values in tensor"
        assert torch.isfinite(datum["scalar"]).all(), "Non-finite values in scalar"
        return self.model.forward(datum["tensor"], datum["scalar"])

    def custom_loss(self, output, datum):
        assert torch.isfinite(datum["label"]).all(), "Non-finite values in label"
        label = torch.clamp(datum["label"], 0, 1)
        return torch.mean((output - label) ** 2)


def main():
    # subprocess.run(["make", "clean"], cwd=os.getcwd())
    # subprocess.run(["make"], cwd=os.getcwd())

    TimeTrainer().train()


if __name__ == "__main__":
    main()
