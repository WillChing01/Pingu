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

import torch

PIECE_TYPES = {x: i for x, i in enumerate("KkQqRrBbNnPp")}

clamp = lambda x: min(max(x, 0), 1)


def get_label(datum):
    alpha = 0.2
    beta = 0.75

    return alpha / (datum.totalPly - datum.ply) + (1 - alpha) * (
        beta * datum.timeSpent / datum.timeLeft
        + (1 - beta) * datum.timeSpent / datum.totalTimeSpent
    )


def datum_to_input(datum):
    pos, side = datum.fen.split(" ")[:2]

    board = torch.tensor([[[0] * 8] * 8] * 14, dtype=torch.bool)
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
            torch.sigmoid(datum.qSearch / 400),
            clamp(datum.increment / datum.timeLeft),
            clamp(0.5 * datum.opponentTime / datum.timeLeft),
        ),
    ), get_label(datum)
