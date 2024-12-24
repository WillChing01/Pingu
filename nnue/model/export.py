import torch

import os

from config import CONFIG
from checkpoint import load_best

TYPES = {
    8: "char",
    16: "short",
    32: "int",
}


def convert(name, t, **kwargs):
    dtype = kwargs["dtype"]
    if kwargs["transpose"]:
        t = torch.transpose(t, dim0=0, dim1=1)

    def convert_dtype(t):
        if t.dim() == 1:
            return f"std::array<{TYPES[dtype]}, {len(t)}>"
        return f"std::array<{convert_dtype(t[0])}, {len(t)}>"

    def convert_tensor(t):
        if t.dim() == 1:
            return f"\u007b\u007b{', '.join(str(x) for x in t.tolist())}\u007d\u007d"
        return f"\u007b\u007b{', '.join(convert_tensor(x) for x in t)}\u007d\u007d"

    return f"alignas(32) const {convert_dtype(t)} {name} = {convert_tensor(t)}"


def main():
    model = load_best()
    quant = model.quantize()

    with open(f"{os.getcwd()}\\..\\..\\include\\weights.h", "w") as f:
        f.write("#ifndef WEIGHTS_H_INCLUDED\n#define WEIGHTS_H_INCLUDED\n\n")
        f.write("#include <array>\n\n")
        for name, layers in quant.items():
            for ind, (w, b) in enumerate(layers):
                shape = list(w.shape)
                if name == "stacks":
                    shape[0] //= CONFIG["stacks"]
                q = CONFIG["quant"][tuple(shape)]
                f.write(f"{convert(f'{name}_w{ind}', w, **q['w'])};\n\n")
                f.write(f"{convert(f'{name}_b{ind}', b, **q['b'])};\n\n")
        f.write("#endif // WEIGHTS_H_INCLUDED\n")


if __name__ == "__main__":
    main()
