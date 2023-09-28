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
            depth, nodes = data.split(" ")
            depth = depth[1:]

            e.stdin("position fen " + fen)
            e.stdin("perft depth " + depth)

            i += 1

            while True:
                res = e.readline()
                if "nodes" in res:
                    res = res.split(" nodes ")
                    if res[1].isdigit():
                        if res[1] != nodes:
                            errors += 1
                            print("Position", i, "FAIL, expected " + nodes + " nodes, received " + res[1])
                        else: print("Position", i, "PASS")
                    else:
                        errors += 1
                        print("Incorrect output format, expected integer")
                    break

    e.quitCommand()

    sys.exit(errors != 0)

if __name__ == "__main__":
    main()
