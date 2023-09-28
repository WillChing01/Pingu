import os
import sys
import engine

def main():
    e = engine.Engine()

    errors = 0
    i = 0

    with open(os.getcwd() + "\\testing\\" + "perft_positions.txt", "r") as file:
        for line in file.readlines():
            fen, data = line.rstrip().split("; ")
            depth = data.split(" ")[0][1:]

            e.stdin("position fen " + fen)
            e.stdin("test incremental depth " + depth)

            i += 1

            while True:
                res = e.readline()
                if "success" in res:
                    res = res.split(" success ")
                    if res[1].isdigit():
                        if int(res[1]) != 1:
                            errors += 1
                            print("Position", i, "FAIL")
                        else: print("Position", i, "PASS")
                    else:
                        errors += 1
                        print("Incorrect output format, expected integer")
                    break

    e.quitCommand()

    sys.exit(errors != 0)

if __name__ == "__main__":
    main()
