import torch

from model import QuantHalfKaNetwork
from checkpoint import load_best


def fen_to_half_ka(fen):
    fen, side = fen.split(" ")[0:2]

    whiteFeatures = []
    blackFeatures = []

    features = []
    kingPos = [-1, -1]

    pieceTypes = "KkQqRrBbNnPp"
    square = 56
    for x in fen:
        if x == "/":
            square -= 16
        elif x.isdigit():
            square += int(x)
        else:
            pieceType = pieceTypes.find(x)
            if pieceType == 0:
                kingPos[0] = square
            elif pieceType == 1:
                kingPos[1] = square
            else:
                features.append(64 * pieceType + square)
            square += 1

    whiteFeatures.append(704 * kingPos[0] + kingPos[1])
    blackFeatures.append(704 * (kingPos[1] ^ 56) + (kingPos[0] ^ 56))
    for x in features:
        pieceType = x // 64
        if pieceType < 2:
            continue

        square = x % 64
        whiteFeatures.append(704 * kingPos[0] + 64 * (pieceType - 1) + square)
        blackFeatures.append(
            704 * (kingPos[1] ^ 56)
            + 64 * (pieceType - 2 * (pieceType % 2))
            + (square ^ 56)
        )

    whiteInput, blackInput = torch.zeros(45056), torch.zeros(45056)
    whiteInput[whiteFeatures] = 1
    blackInput[blackFeatures] = 1

    return (whiteInput, blackInput) if side == "w" else (blackInput, whiteInput)


def main():
    model = load_best()
    model.eval()
    quant = QuantHalfKaNetwork(model.state_dict())
    while True:
        fen = input("fen: ")
        x = fen_to_half_ka(fen)
        print(model.forward(x))
        print(quant.forward(x))


if __name__ == "__main__":
    main()
