import torch

import os

from config import CONFIG
from checkpoint import load_best
from model import quantize


def convert_shape(t, type):
    if t.dim() == 1:
        return f"std::array<{type}, {len(t)}>"
    return f"std::array<{convert_shape(t[0], type)}, {len(t)}>"


def convert_tensor(t):
    if t.dim() == 1:
        return f"\u007b\u007b{', '.join(str(x) for x in t.tolist())}\u007d\u007d"
    return f"\u007b\u007b{', '.join(convert_tensor(t[i]) for i in range(t.shape[0]))}\u007d\u007d"


def main():
    quant = quantize(load_best())

    ind = 0
    with open(f"{os.getcwd()}\\..\\..\\include\\weights.h", "w") as f:
        f.write(
            "#ifndef WEIGHTS_H_INCLUDED\n#define WEIGHTS_H_INCLUDED\n\n#include <array>\n\n"
        )
        for ind, (w, b) in enumerate(quant):
            q = CONFIG["quant"][w.shape]
            if q["transpose"]:
                w = torch.transpose(w, dim0=0, dim1=1)
            f.write(
                f"const {convert_shape(w, q['type'])} w_{ind} = {convert_tensor(w)};\n"
            )
            f.write(
                f"const {convert_shape(b, q['type'])} b_{ind} = {convert_tensor(b)};\n\n"
            )
        f.write("#endif // WEIGHTS_H_INCLUDED\n")


if __name__ == "__main__":
    main()
