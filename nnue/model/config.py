import os

import torch

CONFIG = {
    "device": "cuda",
    "optimizer": {
        "optim": torch.optim.Adam,
        "kwargs": {"eps": 1e-07},
    },
    "path": f"{os.getcwd()}\\checkpoints",
    "modules": ((45056, 32), (2 * 32, 1)),
    "stacks": 4,
    "quant": {
        (32, 45056): {
            "w": {
                "dtype": 16,
                "factor": 64,
                "clamp": 32767,
                "transpose": True,
            },
            "b": {
                "dtype": 16,
                "factor": 64,
                "clamp": 32767,
                "transpose": False,
            },
        },
        (1, 64): {
            "w": {
                "dtype": 8,
                "factor": 1,
                "clamp": 127,
                "transpose": False,
            },
            "b": {
                "dtype": 32,
                "factor": 1,
                "clamp": 32767,
                "transpose": False,
            },
        },
    },
}
