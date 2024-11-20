import torch as torch
from torch import nn

from config import CONFIG


class ClippedReLU(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.clamp(x, 0.0, self.factor)


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.mul(x, self.factor)


class PerspectiveNetwork(nn.Module):
    def __init__(self, module_config):
        super().__init__()

        self.net = nn.Sequential(
            *(
                x
                for i in range(len(module_config) - 1)
                for x in (nn.Linear(*module_config[i : i + 2]), ClippedReLU(127))
            )
        )

    def forward(self, x):
        return torch.cat((self.net(x[0]), self.net(x[1])), dim=-1)


class HalfKaNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        modules = CONFIG["modules"]
        self.net = nn.Sequential(
            *(
                (PerspectiveNetwork(modules[0]),)
                + tuple(
                    x
                    for i in range(len(modules[1]) - 2)
                    for x in (nn.Linear(*modules[1][i : i + 2]), ClippedReLU(127))
                )
                + (nn.Linear(*modules[1][-2:]),)
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
                w = quant["w"]["clamp"]
                b = quant["b"]["clamp"]
                x.weight.data = torch.clamp(x.weight.data, min=-w, max=w)
                x.bias.data = torch.clamp(x.bias.data, min=-b, max=b)

    def forward(self, x):
        return self.net(x)


class QuantHalfKaNetwork(HalfKaNetwork):
    def __init__(self, state_dict):
        super().__init__()
        self.load_state_dict(state_dict)
        self.net.apply(self.quantize)

    def quantize(self, x):
        if isinstance(x, ClippedReLU):
            x.factor = 127
        elif isinstance(x, Scale):
            x.factor = 1
        elif isinstance(x, nn.Linear):
            if q := CONFIG["quant"].get(x.weight.shape):
                x.weight = nn.Parameter(
                    torch.clamp(
                        torch.round(q["w"]["factor"] * x.weight),
                        min=-q["w"]["clamp"],
                        max=q["w"]["clamp"],
                    )
                )
                x.bias = nn.Parameter(
                    torch.clamp(
                        torch.round(q["b"]["factor"] * x.bias),
                        min=-q["b"]["clamp"],
                        max=q["b"]["clamp"],
                    )
                )


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
        if isinstance(x, PerspectiveNetwork):
            ret.append(*quantize(x.net))
        elif isinstance(x, nn.Linear):
            ret.append(quant(x))

    return tuple(ret)


def network():
    return HalfKaNetwork()
