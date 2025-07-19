import glob
import os
from pathlib import Path
import re
import subprocess


def find_clang_format():
    extensions_dir = Path(os.environ["USERPROFILE"]) / ".vscode" / "extensions"
    search_pattern = str(
        extensions_dir / "ms-vscode.cpptools-*" / "LLVM" / "bin" / "clang-format.exe"
    )
    matches = glob.glob(search_pattern)

    if not matches:
        raise FileNotFoundError(
            "Could not find clang-format.exe in VSCode extensions folder."
        )

    return matches[0]


def wrap_header(header):
    return f"#ifndef WEIGHTS_H_INCLUDED\n#define WEIGHTS_H_INCLUDED\n\n{header}\n#endif // WEIGHTS_H_INCLUDED\n"


def parse(dir, file):
    shape_regex = r"_(\d+)"

    length = 1
    for x in re.findall(shape_regex, file):
        length *= int(x)

    dtype = re.split(shape_regex, file, maxsplit=1)[0].split("_")[-1]
    symbol = f"_binary_weights_{dir}_{file}_bin_start"

    return f"extern const {dtype} {symbol}[];\n"


def main():
    header = ""

    for dir, _, f in os.walk("."):
        for file in f:
            if file.endswith(".bin"):
                header += parse(os.path.basename(dir), file.split(".bin")[0])

    weights_file = f"{os.getcwd()}\\..\\include\\weights.h"
    with open(weights_file, "w") as f:
        f.write(wrap_header(header))

    subprocess.run([find_clang_format(), "-i", weights_file])


if __name__ == "__main__":
    main()
