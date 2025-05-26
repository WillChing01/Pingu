import os
import requests
import chess.pgn
import zstandard
import io
from tqdm import tqdm

OUTPUT_DIR = f"{os.getcwd()}\\_raw"


class LoggingTextIOWrapper:
    def __init__(self, stream: io.TextIOWrapper):
        self._stream = stream
        self.buffer = []

    def readline(self, *args, **kwargs):
        if line := self._stream.readline(*args, **kwargs):
            self.buffer.append(line)
            return line
        return None

    def __getattr__(self, name):
        return getattr(self._stream, name)


def get_lichess_files():
    BASE_URL = "https://database.lichess.org/standard/"
    files = []

    finish = 2025, 4
    start = 2017, 4
    current = list(finish)

    while True:
        file = f"lichess_db_standard_rated_{current[0]}-{current[1]:02}.pgn"
        url = f"{BASE_URL}{file}.zst"
        files.append((url, file))

        if current[1] == 1:
            current[1] = 12
            current[0] -= 1
        else:
            current[1] -= 1

        if current[0] < start[0] or current[0] == start[0] and current[1] < start[1]:
            break

    return files


def is_valid_game(headers):
    titles = [headers.get("WhiteTitle") or "", headers.get("BlackTitle") or ""]
    if any(x == "BOT" for x in titles):
        return False

    elo = [int(headers.get("WhiteElo") or 0), int(headers.get("BlackElo") or 0)]
    if not all(x > 2500 for x in elo):
        return False

    time_control = headers.get("TimeControl")
    if not (base_time := time_control.split("+")[0]).isdigit():
        return False
    if int(base_time) <= 60:
        return False

    return True


def stream_file(url, output_file):
    encoding = "utf-8"
    with open(f"{OUTPUT_DIR}\\{output_file}", "w", encoding=encoding) as f:
        with requests.get(url, stream=True) as res:
            res.raise_for_status()
            total = int(res.headers.get("Content-Length") or 0)
            with tqdm.wrapattr(res.raw, "read", total=total) as res_raw:
                dctx = zstandard.ZstdDecompressor()
                reader = dctx.stream_reader(res_raw)
                stream = io.TextIOWrapper(reader, encoding=encoding)
                wrapped_stream = LoggingTextIOWrapper(stream)
                while headers := chess.pgn.read_headers(wrapped_stream):
                    if is_valid_game(headers):
                        f.writelines(wrapped_stream.buffer)
                    wrapped_stream.buffer.clear()


def main():
    files = get_lichess_files()
    for url, file in files:
        stream_file(url, file)


if __name__ == "__main__":
    lines = main()
