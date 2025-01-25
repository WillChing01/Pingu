import sys
import engine


def main():
    e = engine.Engine()

    errors = 0
    i = 0

    with open("perft_positions.txt", "r") as file:
        for line in file.readlines():
            fen = line.rstrip().split("; ")[0]

            e.stdin("position fen " + fen)
            e.stdin("display")

            i += 1

            while True:
                res = e.readline()
                if res.rstrip()[-3:] == "0 1":
                    if res.rstrip() != fen:
                        errors += 1
                        print(
                            "Position",
                            i,
                            "FAIL, expected " + fen + " fen, received " + res.rstrip(),
                        )
                    else:
                        print("Position", i, "PASS")
                    break

    e.quitCommand()

    sys.exit(errors != 0)


if __name__ == "__main__":
    main()
