import os
import ctypes


dataloader = ctypes.CDLL(os.getcwd() + "\\dataloader.dll")

dataloader.main.restype = ctypes.c_int

res = dataloader.main()
