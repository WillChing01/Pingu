import numpy as np
import torch

REPO_ID = "WillChing01/pingu"
PATH_IN_REPO = "Pingu_4.0.0"
REPO_TYPE = "dataset"

DATASET_DTYPE = np.short


def sparse_to_halfKA(sparse):
    """convert sparse into halfKA on torch device"""

    whiteKingPos = sparse[0]
    blackKingPos = sparse[1]

    def toWhitePerspective(x):
        pieceType = x // 64 - 1
        square = x % 64
        return whiteKingPos * 704 + pieceType * 64 + square

    def toBlackPerspective(x):
        pieceType = (x // 64) - 2 * ((x // 64) % 2)
        square = (x % 64) ^ 56
        return blackKingPos * 704 + pieceType * 64 + square

    # remove filler values
    padding = (sparse < 0).short().argmin()
    index = padding[0] - 1 if padding.size() else 31

    whiteSparse = sparse[1:]
    whiteSparse[index:] = whiteSparse[0]
    whiteIndices = torch.vmap(toWhitePerspective)(whiteSparse)

    sparse[[0, 1]] = sparse[[1, 0]]

    blackSparse = sparse[1:]
    blackSparse[index:] = blackSparse[0]
    blackIndices = torch.vmap(toBlackPerspective)(blackSparse)

    return torch.stack((whiteIndices, blackIndices))


def batch_to_halfKA(batch, side):
    res = torch.vmap(sparse_to_halfKA)(batch)
    res[side == 1, [0, 1], :] = res[side == 1, [1, 0], :]
    return res
