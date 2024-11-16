import os

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTIMIZER = torch.optim.Adam

MODEL_PATH = f"{os.getcwd()}\\checkpoints"

INPUT_COUNT = 45056
L1_COUNT = 64
OUTPUT_COUNT = 1

QUANT_CONFIG = {
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
}
