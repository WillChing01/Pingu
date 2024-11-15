import torch as torch
from torch import nn

"""Model definition: (45056 -> 64 -> cReLU(64)) x 2 -> 1"""

INPUT_COUNT = 45056
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


class Concat(nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()

        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, x):
        return torch.cat(
            (self.model_1.forward(x[0]), self.model_2.forward(x[1])), dim=-1
        )


class PerspectiveNetwork(nn.Module):
    def __init__(self, input_count, output_count):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_count, output_count), ClippedReLU())

    def forward(self, x):
        return self.net(x)


class HalfKaNetwork(nn.Module):
    def __init__(self, input_count, l1_count, output_count):
        super().__init__()

        self.net = nn.Sequential(
            Concat(
                PerspectiveNetwork(input_count, l1_count),
                PerspectiveNetwork(input_count, l1_count),
            ),
            nn.Linear(2 * l1_count, output_count),
            Scale(127 * 64),
        )
        self.net.apply(self.init_weights)

    def init_weights(self, x):
        if type(x) == nn.Linear:
            torch.nn.init.kaiming_normal_(x.weight, nonlinearity="relu")
            torch.nn.init.zeros_(x.bias)

    def clamp(self):
        self.net.apply(self._clamp)

    def _clamp(self, x):
        if type(x) == nn.Linear:
            if x.weight.shape == (OUTPUT_COUNT, 2 * L1_COUNT):
                weight_limit = 127 / 64
                bias_limit = 32767 / (64 * 127)
            elif x.weight.shape == (L1_COUNT, INPUT_COUNT):
                weight_limit = 32767 / 127
                bias_limit = 32767 / 127
            torch.clamp(x.weight, min=-weight_limit, max=weight_limit)
            torch.clamp(x.bias, min=-bias_limit, max=bias_limit)

    def forward(self, x):
        return self.net(x)
