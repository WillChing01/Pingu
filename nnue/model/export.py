import torch

import os

from config import CONFIG
from checkpoint import load_best
from model import quantize

TYPES = {
    8: "char",
    16: "short",
    32: "int",
}


def convert(name, t, **kwargs):
    if kwargs["transpose"]:
        torch.transpose(t, dim0=0, dim1=1)

    dtype = kwargs["dtype"]
    avx = kwargs["avx"]

    def convert_dtype(t):
        if t.dim() == 1:
            return (
                f"std::array<__m256i, {len(t) // (256 // dtype)}>"
                if avx
                else f"std::array<{TYPES[dtype]}, {len(t)}>"
            )
        return f"std::array<{convert_dtype(t[0])}, {len(t)}>"

    def convert_tensor(t):
        if t.dim() == 1:
            contents = (
                ", ".join(
                    f"_mm256_setr_epi{dtype}({', '.join(str(x) for x in t[i:i+(256 // dtype)].tolist())})"
                    for i in range(len(t) // (256 // dtype))
                )
                if avx
                else ", ".join(str(x) for x in t.tolist())
            )
            return f"\u007b\u007b{contents}\u007d\u007d"
        return f"\u007b\u007b{', '.join(convert_tensor(x) for x in t)}\u007d\u007d"

    return f"const {convert_dtype(t)} {name} = {convert_tensor(t)}"


def main():
    quant = quantize(load_best())

    ind = 0
    with open(f"{os.getcwd()}\\..\\..\\include\\weights.h", "w") as f:
        f.write("#ifndef WEIGHTS_H_INCLUDED\n#define WEIGHTS_H_INCLUDED\n\n")
        f.write("#include <array>\n#include <immintrin.h>\n\n")
        for ind, (w, b) in enumerate(quant):
            q = CONFIG["quant"][w.shape]
            f.write(f"{convert(f'w_{ind}', w, **q['w'])};\n\n")
            f.write(f"{convert(f'b_{ind}', b, **q['b'])};\n\n")
        f.write("#endif // WEIGHTS_H_INCLUDED\n")


if __name__ == "__main__":
    main()
