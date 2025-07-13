import ctypes
import os
import numpy as np
import torch
from tqdm import tqdm


DATALOADER_CONFIG = {
    "training": {
        "path": bytes(f"{os.getcwd()}\\..\\dataset\\training", "utf-8"),
    },
    "validation": {
        "path": bytes(f"{os.getcwd()}\\..\\dataset\\validation", "utf-8"),
    },
}


class Batch(ctypes.Structure):
    _fields_ = [
        ("tensor", ctypes.POINTER(ctypes.c_float)),
        ("scaledEval", ctypes.POINTER(ctypes.c_float)),
        ("scaledPly", ctypes.POINTER(ctypes.c_float)),
        ("scaledIncrement", ctypes.POINTER(ctypes.c_float)),
        ("scaledOpponentTime", ctypes.POINTER(ctypes.c_float)),
        ("label", ctypes.POINTER(ctypes.c_float)),
        ("size", ctypes.c_int),
    ]

    def reformat(self, device):
        tensor = torch.from_numpy(
            np.ctypeslib.as_array(self.tensor, shape=(self.size, 14, 8, 8))
        ).to(device)

        scalar = torch.stack(
            (
                torch.from_numpy(
                    np.ctypeslib.as_array(self.scaledEval, shape=(self.size,))
                ).to(device),
                torch.from_numpy(
                    np.ctypeslib.as_array(self.scaledPly, shape=(self.size,))
                ).to(device),
                torch.from_numpy(
                    np.ctypeslib.as_array(self.scaledIncrement, shape=(self.size,))
                ).to(device),
                torch.from_numpy(
                    np.ctypeslib.as_array(self.scaledOpponentTime, shape=(self.size,))
                ).to(device),
            ),
            dim=1,
        )

        label = (
            torch.from_numpy(np.ctypeslib.as_array(self.label, shape=(self.size,)))
            .reshape((self.size, 1))
            .to(device)
        )

        datum = {
            "tensor": tensor,
            "scalar": scalar,
            "label": label,
        }

        return self.size, datum


dll = ctypes.CDLL(os.getcwd() + "\\dataloader.dll")

dll.constructDataLoader.restype = ctypes.c_void_p
dll.constructDataLoader.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

dll.destructDataLoader.restype = None
dll.destructDataLoader.argtypes = [ctypes.c_void_p]

dll.length.restype = ctypes.c_ulonglong
dll.length.argtypes = [ctypes.c_char_p]

dll.getBatch.restype = ctypes.POINTER(Batch)
dll.getBatch.argtypes = [ctypes.c_void_p]

dll.destructBatch.restype = None
dll.destructBatch.argtypes = [ctypes.POINTER(Batch)]


class DataLoader:
    def __init__(self, kind, device, batch_size=8192, num_workers=4):
        self.path = DATALOADER_CONFIG[kind]["path"]
        self.length = dll.length(self.path)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.length

    def iterator(self):
        dl = dll.constructDataLoader(self.path, self.batch_size, self.num_workers)

        while batch := dll.getBatch(dl):
            yield batch.contents.reformat(self.device)
            dll.destructBatch(batch)

        dll.destructDataLoader(dl)

    def test(self):
        progress = tqdm(total=len(self))
        for batch_size, datum in self.iterator():
            assert torch.isfinite(datum["tensor"]).all(), "tensor is NaN"
            assert torch.isfinite(datum["scalar"]).all(), "scalar is NaN"
            assert torch.isfinite(datum["label"]).all(), "label is NaN"
            progress.update(batch_size)
