import os
import requests
import chess.pgn
import zstandard
import io
from tqdm import tqdm

DIR = f"{os.getcwd()}\\_raw"


def main():
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    with requests.get(url, stream=True) as res:
        res.raise_for_status()
        file_size = int(res.headers.get("Content-Length", 0))
        with tqdm.wrapattr(res.raw, "read", total=file_size) as res_raw:
            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(res_raw)
            stream = io.TextIOWrapper(reader, encoding="utf-8")
            while headers := chess.pgn.read_headers(stream):
                elo = [headers.get("WhiteElo") or 0, headers.get("BlackElo") or 0]
                if not all(int(x) > 2500 for x in elo):
                    continue


if __name__ == "__main__":
    lines = main()
