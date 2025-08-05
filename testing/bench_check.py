import sys
from time import time
import re
import engine


def main():
    try:
        with open("bench_value.txt", "r") as file:
            expected = int(file.readline().strip())
    except:
        print(
            'Error: expected to find "bench_value.txt" with bench value as first line'
        )

    timeout = 30
    startTime = time()

    args = engine.ARGS + ["bench"]
    e = engine.Engine(args=args)

    while True:
        res = e.readline()

        if not res.startswith("info"):
            pattern = r"^(\d+) nodes \d+ nps$"
            if not (search := re.search(pattern, res)):
                print(f'Incorrect bench format. Expected "{pattern}", received "{res}"')
                sys.exit(1)
            if (bench := int(search.group(1))) != expected:
                print(
                    f'Incorrect bench value. Expected "{expected}", received "{bench}"'
                )
                sys.exit(1)
            break

        if time() - startTime > timeout:
            print(f"Engine timeout after {timeout}s")
            sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
