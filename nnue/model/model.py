import torch as torch
from torch import nn

from config import CONFIG


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
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        return torch.cat((self.model.forward(x[0]), self.model.forward(x[1])), dim=-1)


class PerspectiveNetwork(nn.Module):
    def __init__(self, module_config):
        super().__init__()

        self.net = nn.Sequential(
            *(
                x
                for i in range(len(module_config) - 1)
                for x in (nn.Linear(*module_config[i : i + 2]), ClippedReLU())
            )
        )

    def forward(self, x):
        return self.net(x)


class HalfKaNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        modules = CONFIG["modules"]
        self.net = nn.Sequential(
            *(
                (Concat(PerspectiveNetwork(modules[0])),)
                + tuple(
                    x
                    for i in range(len(modules[1]) - 2)
                    for x in (nn.Linear(*modules[1][i : i + 2]), ClippedReLU())
                )
                + (nn.Linear(*modules[1][-2:]), Scale(CONFIG["quant"]["scaling"]))
            )
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
            if quant := CONFIG["quant"].get(x.weight.shape):
                weight_limit = quant["w"]["clamp"] / quant["w"]["factor"]
                bias_limit = quant["b"]["clamp"] / quant["b"]["factor"]
                torch.clamp(x.weight, min=-weight_limit, max=weight_limit)
                torch.clamp(x.bias, min=-bias_limit, max=bias_limit)

    def forward(self, x):
        return self.net(x)


def quantize(model):
    def quant(x):
        if q := CONFIG["quant"].get(x.weight.shape):
            weights = torch.clamp(
                torch.round(q["w"]["factor"] * x.weight).int(),
                min=-q["w"]["clamp"],
                max=q["w"]["clamp"],
            )
            bias = torch.clamp(
                torch.round(q["b"]["factor"] * x.bias).int(),
                min=-q["b"]["clamp"],
                max=q["b"]["clamp"],
            )
            return weights, bias
        raise ValueError(f"no quantization scheme for {x.weight.shape}")

    ret = []
    for x in model.net:
        if isinstance(x, Concat):
            ret.append(*quantize(x.model))
        elif isinstance(x, nn.Linear):
            ret.append(quant(x))

    return tuple(ret)


def network():
    return HalfKaNetwork()