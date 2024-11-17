import os
import ctypes

import numpy as np
import torch

from config import CONFIG

NUM_FEATURES = CONFIG["modules"][0][0]
BATCH_SIZE = 32768
NUM_WORKERS = 4

DATALOADER_CONFIGS = {
    "training": {
        "path": bytes(f"{os.getcwd()}\\..\\dataset\\training", "utf-8"),
    },
    "validation": {
        "path": bytes(f"{os.getcwd()}\\..\\dataset\\validation", "utf-8"),
    },
}


class HalfKaSparseBatch(ctypes.Structure):
    _fields_ = [
        ("indices", ctypes.POINTER(ctypes.c_ulonglong)),
        ("firstFeatures", ctypes.POINTER(ctypes.c_ulonglong)),
        ("secondFeatures", ctypes.POINTER(ctypes.c_ulonglong)),
        ("result", ctypes.POINTER(ctypes.c_double)),
        ("eval", ctypes.POINTER(ctypes.c_short)),
        ("totalFeatures", ctypes.c_int),
        ("size", ctypes.c_int),
    ]

    def reformat(self, device):
        values = torch.ones(self.totalFeatures, dtype=torch.float32, device=device)

        firstFeaturesIndices = torch.stack(
            (
                torch.from_numpy(
                    np.ctypeslib.as_array(self.indices, shape=(self.totalFeatures,))
                ).to(device),
                torch.from_numpy(
                    np.ctypeslib.as_array(
                        self.firstFeatures, shape=(self.totalFeatures,)
                    )
                ).to(device),
            )
        )

        secondFeaturesIndices = torch.stack(
            (
                torch.from_numpy(
                    np.ctypeslib.as_array(self.indices, shape=(self.totalFeatures,))
                ).to(device),
                torch.from_numpy(
                    np.ctypeslib.as_array(
                        self.secondFeatures, shape=(self.totalFeatures,)
                    )
                ).to(device),
            )
        )

        firstBatch = torch.sparse_coo_tensor(
            firstFeaturesIndices,
            values,
            (self.size, NUM_FEATURES),
            device=device,
            check_invariants=False,
            is_coalesced=True,
        )

        secondBatch = torch.sparse_coo_tensor(
            secondFeaturesIndices,
            values,
            (self.size, NUM_FEATURES),
            device=device,
            check_invariants=False,
            is_coalesced=True,
        )

        evals = (
            torch.from_numpy(np.ctypeslib.as_array(self.eval, shape=(self.size,)))
            .reshape((self.size, 1))
            .to(device)
        )

        results = (
            torch.from_numpy(np.ctypeslib.as_array(self.result, shape=(self.size,)))
            .reshape((self.size, 1))
            .to(device)
        )

        return (firstBatch, secondBatch), evals, results


dll = ctypes.CDLL(os.getcwd() + "\\dataloader.dll")

dll.constructDataLoader.restype = ctypes.c_void_p
dll.constructDataLoader.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

dll.destructDataLoader.restype = None
dll.destructDataLoader.argtypes = [ctypes.c_void_p]

dll.length.restype = ctypes.c_ulonglong
dll.length.argtypes = [ctypes.c_char_p]

dll.getBatch.restype = ctypes.POINTER(HalfKaSparseBatch)
dll.getBatch.argtypes = [ctypes.c_void_p]

dll.destructBatch.restype = None
dll.destructBatch.argtypes = [ctypes.POINTER(HalfKaSparseBatch)]


class DataLoader:
    def __init__(self, kind):
        self.path = DATALOADER_CONFIGS[kind]["path"]
        self.length = dll.length(self.path)

    def __len__(self):
        return self.length

    def iterator(self):
        dataLoader = dll.constructDataLoader(self.path, BATCH_SIZE, NUM_WORKERS)

        while batch := dll.getBatch(dataLoader):
            yield batch.contents.reformat(CONFIG["device"])
            dll.destructBatch(batch)

        dll.destructDataLoader(dataLoader)
