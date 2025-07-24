import os
import sys

import torch

sys.path.insert(0, os.getcwd() + "\\..\\..\\")

from pipeline.checkpoint import Checkpoint
from model import SimpleTimeNetwork

# TODO correct this for checks and side to move
def parse_fen(fen):
    pieceTypes = "KkQqRrBbNnPp"
    square = 56
    board = torch.zeros((1, 14, 8, 8))
    for x in fen:
        if x == " ":
            break
        elif x == "/":
            square -= 16
        elif x.isdigit():
            square += int(x)
        else:
            board[0][pieceTypes.find(x)][square // 8][square % 8] = 1
            square += 1
    return board


def main():
    board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    scalars = torch.tensor([[0.553543903151, 0.01, 0, 0.5]])
    checkpoint = Checkpoint("checkpoints", "cpu", SimpleTimeNetwork, None, {})
    model = checkpoint.load_best()
    print(model.forward(board, scalars))


if __name__ == "__main__":
    main()
