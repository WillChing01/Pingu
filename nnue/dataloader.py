import os
import ctypes
import sys
import numpy as np
import torch


BATCH_SIZE = 1024
NUM_FEATURES = 45056


class HalfKaSparseBatch(ctypes.Structure):
    _fields_ = [
        ("indices", ctypes.POINTER(ctypes.c_ulonglong)),
        ("firstFeatures", ctypes.POINTER(ctypes.c_ulonglong)),
        ("secondFeatures", ctypes.POINTER(ctypes.c_ulonglong)),
        ("result", ctypes.POINTER(ctypes.c_double)),
        ("eval", ctypes.POINTER(ctypes.c_short)),
        ("totalFeatures", ctypes.c_int),
    ]

    def reformat(self, device):
        values = torch.ones(self.totalFeatures, dtype=torch.float32)

        firstFeaturesIndices = torch.stack(
            (
                torch.from_numpy(np.ctypeslib.as_array(self.indices, shape=(self.totalFeatures,))),
                torch.from_numpy(np.ctypeslib.as_array(self.firstFeatures, shape=(self.totalFeatures,))),
            )
        )

        secondFeaturesIndices = torch.stack(
            (
                torch.from_numpy(np.ctypeslib.as_array(self.indices, shape=(self.totalFeatures,))),
                torch.from_numpy(np.ctypeslib.as_array(self.secondFeatures, shape=(self.totalFeatures,))),
            )
        )

        firstBatch = torch.sparse_coo_tensor(
            firstFeaturesIndices,
            values,
            (BATCH_SIZE, NUM_FEATURES),
            device=device,
            check_invariants=False,
            is_coalesced=True,
        )

        secondBatch = torch.sparse_coo_tensor(
            secondFeaturesIndices,
            values,
            (BATCH_SIZE, NUM_FEATURES),
            device=device,
            check_invariants=False,
            is_coalesced=True,
        )

        evals = torch.from_numpy(np.ctypeslib.as_array(self.eval, shape=(BATCH_SIZE,)))

        results = torch.from_numpy(
            np.ctypeslib.as_array(self.result, shape=(BATCH_SIZE,))
        )

        return (firstBatch, secondBatch, evals, results)


dll = ctypes.CDLL(os.getcwd() + "\\dataloader.dll")

dll.constructDataLoader.restype = ctypes.c_void_p
dll.constructDataLoader.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

dll.destructDataLoader.restype = None
dll.destructDataLoader.argtypes = [ctypes.c_void_p]

dll.getBatch.restype = ctypes.POINTER(HalfKaSparseBatch)
dll.getBatch.argtypes = [ctypes.c_void_p]

dll.destructBatch.restype = None
dll.destructBatch.argtypes = [ctypes.POINTER(HalfKaSparseBatch)]

res = dll.constructDataLoader(
    bytes(os.getcwd() + "\\dataset\\training", "utf-8"), 1024, 6
)

batch = dll.getBatch(res)

x = batch.contents.reformat("cpu")

print(x)
print(sys.getsizeof(x))

dll.destructBatch(batch)

dll.destructDataLoader(res)
