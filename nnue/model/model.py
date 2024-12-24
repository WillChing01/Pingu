import torch as torch
from torch import nn

from config import CONFIG


class ClippedReLU(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.clamp(x, 0.0, self.factor)


class PerspectiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        config = CONFIG["modules"][0]
        self.net = nn.Sequential(
            *(
                x
                for i in range(len(config) - 1)
                for x in (nn.Linear(*config[i : i + 2]), ClippedReLU(127))
            )
        )

    def forward(self, x):
        return torch.cat((self.net(x[0]), self.net(x[1])), dim=-1)


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        config = CONFIG["modules"][1]
        self.net = nn.Sequential(
            *(
                tuple(
                    x
                    for i in range(len(config) - 2)
                    for x in (nn.Linear(*config[i : i + 2]), ClippedReLU(127))
                )
                + (nn.Linear(*config[-2:]),)
            )
        )

    def forward(self, x):
        return self.net(x)


class HalfKaNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_stacks = CONFIG["stacks"]
        self.stacks = [Stack() for _ in range(self.num_stacks)]

        self.add_module("perspective", PerspectiveNetwork())
        for i in range(self.num_stacks):
            self.add_module(f"stack_{i}", self.stacks[i])

        self.apply(self.init_weights)

    def init_weights(self, x):
        if type(x) == nn.Linear:
            torch.nn.init.kaiming_normal_(x.weight, nonlinearity="relu")
            torch.nn.init.zeros_(x.bias)

    def clamp(self):
        def _clamp(x):
            if type(x) == nn.Linear:
                if quant := CONFIG["quant"].get(x.weight.shape):
                    w = quant["w"]["clamp"] / quant["w"]["factor"]
                    b = quant["b"]["clamp"] / quant["b"]["factor"]
                    x.weight.data = torch.clamp(x.weight.data, min=-w, max=w)
                    x.bias.data = torch.clamp(x.bias.data, min=-b, max=b)

        self.apply(_clamp)

    def quantize(self):
        ret = []

        def _quantize(x):
            if isinstance(x, nn.Linear):
                if quant := CONFIG["quant"].get(x.weight.shape):
                    w = quant["w"]["factor"], quant["w"]["clamp"]
                    b = quant["b"]["factor"], quant["b"]["clamp"]

                    q_w = (
                        torch.clamp(x.weight * w[0], min=-w[1], max=w[1]).round().int()
                    )
                    q_b = torch.clamp(x.bias * b[0], min=-b[1], max=b[1]).round().int()

                    ret.append((q_w, q_b))

        self.apply(_quantize)

        return ret

    def forward(self, x, piece_counts):
        p_out = self.perspective.forward(x)
        out = torch.concat(tuple(stack.forward(p_out) for stack in self.stacks), dim=-1)
        indices = torch.floor((piece_counts - 1) * self.num_stacks / 32).long()
        return torch.gather(out, -1, indices)


def network():
    return HalfKaNetwork()
