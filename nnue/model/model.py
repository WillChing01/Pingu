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
        if isinstance(x, nn.Linear):
            if q := CONFIG["quant"].get(x.weight.shape):
                w = q["w"]["clamp"]
                b = q["b"]["clamp"]
                x.weight.data = torch.clamp(x.weight.data, min=-w, max=w).round()
                x.bias.data = torch.clamp(x.bias.data, min=-b, max=b).round()

    def get_quant_params(self):
        ret = []
        self.net.apply(lambda x: self.gather(x, ret))
        return ret

    def gather(self, x, ret):
        if isinstance(x, nn.Linear):
            ret.append((x.weight.int(), x.bias.int()))


def network():
    return HalfKaNetwork()
