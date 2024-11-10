import os
import ctypes

import numpy as np
import torch

NUM_FEATURES = 45056
BATCH_SIZE = 1024
NUM_WORKERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
                ).to(device, non_blocking=True),
                torch.from_numpy(
                    np.ctypeslib.as_array(
                        self.firstFeatures, shape=(self.totalFeatures,)
                    )
                ).to(device, non_blocking=True),
            )
        )

        secondFeaturesIndices = torch.stack(
            (
                torch.from_numpy(
                    np.ctypeslib.as_array(self.indices, shape=(self.totalFeatures,))
                ).to(device, non_blocking=True),
                torch.from_numpy(
                    np.ctypeslib.as_array(
                        self.secondFeatures, shape=(self.totalFeatures,)
                    )
                ).to(device, non_blocking=True),
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

        evals = torch.from_numpy(
            np.ctypeslib.as_array(self.eval, shape=(self.size,))
        ).to(device, non_blocking=True)

        results = torch.from_numpy(
            np.ctypeslib.as_array(self.result, shape=(self.size,))
        ).to(device, non_blocking=True)

        return (firstBatch, secondBatch, evals, results)


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
            yield batch.contents.reformat(DEVICE)
            dll.destructBatch(batch)

        dll.destructDataLoader(dataLoader)
