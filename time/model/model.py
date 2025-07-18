import torch
from torch import nn

from pipeline.export import export_layer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))

    def export(self, prefix=""):
        return export_layer(f"{prefix}block_in", self.block[0]) | export_layer(
            f"{prefix}block_out", self.block[2]
        )


class TimeNetwork(nn.Module):
    def __init__(self, cnn_channels=14, scalar_channels=4):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.block_1 = ResidualBlock(32)
        self.block_2 = ResidualBlock(32)

        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + scalar_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, board, scalar_inputs):
        cnn_output = self.downsample(self.block_2(self.block_1(self.initial(board))))
        return self.head(torch.concat((scalar_inputs, cnn_output), dim=1))


class SimpleTimeNetwork(nn.Module):
    def __init__(self, cnn_channels=14, scalar_channels=4):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(cnn_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.block = ResidualBlock(24)

        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(96 + scalar_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, board, scalar_inputs):
        cnn_output = self.downsample(self.block(self.initial(board)))
        return self.head(torch.concat((scalar_inputs, cnn_output), dim=1))

    def export(self, prefix=""):
        return (
            export_layer(f"{prefix}initial", self.initial[0])
            | self.block.export(prefix=prefix)
            | export_layer(f"{prefix}head", self.head[0])
            | export_layer(f"{prefix}head", self.head[2])
        )


def network():
    return SimpleTimeNetwork()
