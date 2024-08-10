"""

Continuously generate self-play data until manually terminated.

"""

import os
import sys
import time
import multiprocessing

HASH = 64

DEPTH = 8
POSITIONS = 20000
RANDOMPLY = 6
MAXPLY = 200
EVALBOUND = 8192

ARGS = (HASH, DEPTH, POSITIONS, RANDOMPLY, MAXPLY, EVALBOUND)

def gensfen_worker(hash, depth, positions, randomply, maxply, evalbound):
    import engine
    e = engine.Engine("Pingu.exe", "\\..\\..\\")
    e.stdin("setoption name Hash value " + str(hash))
    time.sleep(0.5)

    e.stdin("gensfen depth "+str(depth)+" positions "+str(positions)+" randomply "+str(randomply)+" maxply "+str(maxply)+" evalbound "+str(evalbound))
    while True:
        res = e.readline()
        if res == "Finished generating positions.":
            break
    e.quitCommand()
    time.sleep(0.5)

def main():
    try:
        os.chdir(os.getcwd() + "\\datasets\\")
    except:
        os.mkdir(os.getcwd() + "\\datasets\\")
        os.chdir(os.getcwd() + "\\datasets\\")
    sys.path.insert(1, os.getcwd() + "\\..\\..\\testing\\")

    num_cpu = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(num_cpu)
    result = [False for i in range(num_cpu)]
    total = 0

    for i in range(num_cpu):
        result[i] = pool.apply_async(gensfen_worker, args = ARGS)
        print("Starting worker", i+1)
        time.sleep(5)

    while True:
        for i in range(num_cpu):
            if result[i].ready():
                total += POSITIONS
                print("Generated", total, "positions")
                result[i] = pool.apply_async(gensfen_worker, args = ARGS)
                print("Starting worker", i+1)
                time.sleep(5)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
