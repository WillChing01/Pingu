import os

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = f"{os.getcwd()}\\checkpoints"

INPUT_COUNT = 45056
L1_COUNT = 64
OUTPUT_COUNT = 1
