"""
Notes on input features and target outputs

Inputs:

- Board position from side-to-move perspective
- Evaluation (via qSearch) => sigmoid using scaling factor for nnue training
- Are we in check [0,1]
- Current ply => ply / (ply+40)
- Binary flag for whether we have increment
- Increment / ourTime if binary flag is 1 => sigmoid(log(increment / ourTime))
- Ratio of opponent time => sigmoid(log(opponentTime / ourTime))
- Ratio of ourTime / startTime

"""

import math
import os
import random
import numpy as np
import pandas as pd
import torch

from parse import count_rows

PIECE_TYPES = {x: i for i, x in enumerate("KkQqRrBbNnPp")}

clamp = lambda x: min(max(x, 0), 1)
sigmoid = lambda x: 1 / (1 + math.exp(-x))


def get_label(datum):
    alpha = 0.2
    beta = 0.75

    return alpha / (datum.totalPly - datum.ply) + (1 - alpha) * (
        beta * datum.timeSpent / datum.timeLeft
        + (1 - beta) * datum.timeSpent / datum.totalTimeSpent
    )


def datum_to_input(datum):
    pos, side = datum.fen.split(" ")[:2]

    board = torch.zeros((14, 8, 8), dtype=torch.bool)
    board[-2] = side == "b"
    board[-1] = datum.inCheck

    square = 56
    for x in pos:
        if x == "/":
            square -= 16
        elif x.isdigit():
            square += int(x)
        else:
            board[PIECE_TYPES[x]][square // 8][square % 8] = 1
            square += 1

    return (
        board,
        (
            clamp(datum.ply / 100),
            sigmoid(datum.qSearch / 400),
            clamp(datum.increment / datum.timeLeft),
            clamp(0.5 * datum.opponentTime / datum.timeLeft),
        ),
    ), get_label(datum)


LENGTH_BY_PATH = {
    kind: {
        x.path: count_rows(x.path) for x in os.scandir(kind) if x.path.endswith(".csv")
    }
    for kind in ["training", "validation"]
}

TRAINING_LENGTH = sum(LENGTH_BY_PATH["training"].values())
VALIDATION_LENGTH = sum(LENGTH_BY_PATH["validation"].values())

HEADERS = "fen,isDraw,isWin,ply,totalPly,qSearch,inCheck,increment,timeLeft,timeSpent,totalTimeSpent,startTime,opponentTime".split(
    ","
)


def dataloader(kind, batch_size=1024):
    assert kind in ["training", "validation"]

    paths = [x.path for x in os.scandir(kind) if x.path.endswith(".csv")]
    random.shuffle(paths)

    for path in paths:
        df = pd.read_csv(path, names=HEADERS, header=0)
        indices = np.random.permutation(LENGTH_BY_PATH[kind][path])

        for i in range(0, len(indices), batch_size):
            data = df.iloc[indices[i : i + batch_size]]
            batch = [datum_to_input(datum) for datum in data.itertuples(index=False)]
            yield batch
