import torch

from checkpoint import load_best


def fenToHalfKa(fen):
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
    blackFeatures.append(704 * kingPos[1] + (kingPos[0] ^ 56))
    for x in features:
        pieceType = x // 64
        if pieceType < 2:
            continue

        square = x % 64
        whiteFeatures.append(704 * kingPos[0] + 64 * (pieceType - 1) + square)

        if pieceType % 2 == 0:
            pieceType += 1
        else:
            pieceType -= 1
        blackFeatures.append(704 * kingPos[1] + 64 * (pieceType - 1) + (square ^ 56))

    whiteInput, blackInput = torch.zeros(45056), torch.zeros(45056)
    whiteInput[whiteFeatures] = 1
    blackInput[blackFeatures] = 1

    return (whiteInput, blackInput) if side == "w" else (blackInput, whiteInput)


def main():
    model = load_best()
    model.eval()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    score = model.forward(fenToHalfKa(fen))
    print(score)


if __name__ == "__main__":
    main()
