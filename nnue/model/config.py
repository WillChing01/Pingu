import os

import torch

"""Model definition: (45056 -> 64 -> cReLU(64)) x 2 -> 1"""


CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optimizer": torch.optim.Adam,
    "path": f"{os.getcwd()}\\checkpoints",
    "modules": ((45056, 64), (2 * 64, 1)),
    "quant": {
        "scaling": 64 * 127,
        (64, 45056): {
            "type": "short",
            "w": {
                "factor": 127,
                "clamp": 32767,
            },
            "b": {
                "factor": 127,
                "clamp": 32767,
            },
            "transpose": True,
        },
        (1, 128): {
            "type": "char",
            "w": {
                "factor": 64,
                "clamp": 127,
            },
            "b": {
                "factor": 64 * 127,
                "clamp": 32767,
            },
            "transpose": False,
        },
    },
}
