import sys
import time
import engine

TIME_LIMIT = 10


def main():
    e = engine.Engine()
    e.stdin("setoption name Hash value 256")

    errors = 0
    i = 0

    with open("nps_positions.txt", "r") as file:
        for line in file.readlines():
            i += 1

            fen = line.rstrip()

            e.stdin("position fen " + fen)
            e.stdin("setoption name Clear Hash")

            nps = 0

            e.stdin("go infinite")
            startTime = time.perf_counter()
            while time.perf_counter() - startTime < TIME_LIMIT:
                res = e.readline()
                if "info" in res:
                    res = res.split(" ")
                    for j in range(1, len(res)):
                        if res[j - 1] == "nps" and res[j].isdigit():
                            nps = int(res[j])
                            break
            e.stdin("stop")

            if nps > 0:
                print("Position", i, "nps", nps)
            else:
                errors += 1
                print("Incorrect output format, expected integer after 'nps'")

    e.quitCommand()

    sys.exit(errors != 0)


if __name__ == "__main__":
    main()
