import torch

import os

from config import CONFIG
from checkpoint import load_best

TYPES = {
    8: "char",
    16: "short",
    32: "int",
}

DTYPES = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
}


def write_to_binary(name, t, **kwargs):
    if kwargs["transpose"]:
        t = torch.transpose(t, dim0=0, dim1=1)

    file_name = (
        f'{name}_{TYPES[kwargs["dtype"]]}_{"_".join(str(x) for x in t.size())}.bin'
    )
    file = f"{os.getcwd()}\\..\\..\\weights\\nnue\\{file_name}"
    with open(file, "wb") as f:
        f.write(t.to(DTYPES[kwargs["dtype"]]).contiguous().numpy().tobytes())


def main():
    model = load_best()
    quant = model.quantize()

    for name, layers in quant.items():
        for ind, (w, b) in enumerate(layers):
            shape = list(w.shape)
            if name == "stacks":
                shape[0] //= CONFIG["stacks"]
            q = CONFIG["quant"][tuple(shape)]
            write_to_binary(f"{name}_w{ind}", w, **q["w"])
            write_to_binary(f"{name}_b{ind}", b, **q["b"])


if __name__ == "__main__":
    main()
