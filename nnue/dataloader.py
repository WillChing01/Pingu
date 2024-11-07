import os
import ctypes


class HalfKaSparseBatch(ctypes.Structure):
    _fields_ = [
        ("indices", ctypes.Pointer(ctypes.ulonglong)),
        ("firstFeatures", ctypes.Pointer(ctypes.ulonglong)),
        ("secondFeatures", ctypes.Pointer(ctypes.ulonglong)),
        ("result", ctypes.Pointer(ctypes.double)),
        ("eval", ctypes.Pointer(ctypes.short)),
        ("totalFeatures", ctypes.c_int),
    ]

    def reformat(self, device):
        return


dataloader = ctypes.CDLL(os.getcwd() + "\\dataloader.dll")

dataloader.constructDataLoader.restype = ctypes.c_void_p
dataloader.constructDataLoader.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

dataloader.destructDataLoader.argtypes = [ctypes.c_void_p]

res = dataloader.constructDataLoader(
    bytes(os.getcwd() + "\\dataset\\training", "utf-8"), 1024, 6
)

dataloader.destructDataLoader(res)
