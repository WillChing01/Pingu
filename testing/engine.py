"""
Expects to find 'Pingu.exe' in parent folder.

"""

import os
from subprocess import Popen, PIPE

ENGINE_EXE = "Pingu.exe"
REL_PATH = "\\..\\"

class Engine:

    def __init__(self, name = ENGINE_EXE, path = REL_PATH):
        self._process = Popen(os.getcwd() + path + name, stdin = PIPE, stdout = PIPE)

    def stdin(self, string):
        self._process.stdin.write((string + "\n").encode())
        self._process.stdin.flush()
        return 0

    def readline(self):
        return self._process.stdout.readline().rstrip().decode()

    def uciCommand(self):
        self.stdin("uci")
        while True:
            res = self.readline()
            if res == "uciok": break
        return 0

    def isReadyCommand(self):
        self.stdin("isready")
        while True:
            res = self.readline()
            if res == "readyok": break
        return 0

    def quitCommand(self):
        self.stdin("quit")
        return 0
