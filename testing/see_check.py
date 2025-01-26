import sys
import engine


def main():
    e = engine.Engine()

    errors = 0
    i = 0

    with open("see_positions.txt", "r") as file:
        for line in file.readlines():
            fen, move, score = line.rstrip().split("; ")

            e.stdin("position fen " + fen)
            e.stdin("see move " + move)

            i += 1

            while True:
                res = e.readline()
                if "score" in res:
                    res = res.split(" score ")
                    if res[1] != score:
                        errors += 1
                        print("Position", i, "FAIL")
                    else:
                        print("Position", i, "PASS")
                    break

    e.quitCommand()

    sys.exit(errors != 0)


if __name__ == "__main__":
    main()
