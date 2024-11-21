import os

import torch

"""Model definition: (45056 -> 64 -> cReLU(64)) x 2 -> 1"""


CONFIG = {
    "device": "cuda",
    "optimizer": torch.optim.Adam,
    "path": f"{os.getcwd()}\\checkpoints",
    "modules": ((45056, 64), (2 * 64, 1)),
    "quant": {
        "scaling": 64 * 127,
        (64, 45056): {
            "w": {
                "dtype": 16,
                "factor": 127,
                "clamp": 32767,
                "transpose": True,
            },
            "b": {
                "dtype": 16,
                "factor": 127,
                "clamp": 32767,
                "transpose": False,
            },
        },
        (1, 128): {
            "w": {
                "dtype": 8,
                "factor": 64,
                "clamp": 127,
                "transpose": False,
            },
            "b": {
                "dtype": 32,
                "factor": 64 * 127,
                "clamp": 32767,
                "transpose": False,
            },
        },
    },
}
