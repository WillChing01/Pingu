"""Expects to find 'Pingu.exe' in parent folder."""

from subprocess import Popen, PIPE
import sys

ARGS = ["..\\Pingu.exe" if sys.platform == "win32" else "../Pingu"]


class Engine:

    def __init__(self, args=ARGS):
        self._process = Popen(args, stdin=PIPE, stdout=PIPE)

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
            if res == "uciok":
                break
        return 0

    def isReadyCommand(self):
        self.stdin("isready")
        while True:
            res = self.readline()
            if res == "readyok":
                break
        return 0

    def quitCommand(self):
        self.stdin("quit")
        return 0
