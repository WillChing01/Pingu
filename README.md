# Chess_3

UCI compatible chess engine built from scratch.

Many thanks to the advice on [Chess Programming Wiki](www.chessprogramming.org).

Play me on Lichess! My username is [WilliamEngine](https://lichess.org/@/WilliamEngine).

# Features

### Move generation
- Pseudo-legal move generation
- Bitboard representation
- Plain magic bitboards for sliding pieces

### Move ordering
- PV move at root
- Hash move
- Static exchange evaluation
- Killer moves
- Quiet moves ordered at random

### Search
- Negamax with fail-soft alpha-beta
- Quiescence search
- Iterative deepening at root
- Transposition tables
  - Always replace
  - Depth-preferred
- Null move pruning

### Evaluation
- Material
- Piece square tables
  - Tapered eval for king

### Testing
- Perft function (initial and Kiwipete positions)

# Executable

### Github releases
Check out [releases](https://github.com/WillChing01/Chess_3/releases/) to download the .exe for master branch.

### Compile with cmake

__Requires cmake, g++ and mingw__

Clone the repository, go to ```/Chess_3``` directory and run these terminal commands:

```
mkdir build
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Chess_3 --config Release --
```

Alternatively, you can skip downloading mingw and replace "MinGW Makefiles" with another generator that you already have.

The executable will appear in ```/build``` directory.

### Compile manually
__Requires g++__

Clone the repository, go to ```/Chess_3``` directory and run these terminal commands:

```
g++ -Wall -std=gnu++17 -fexceptions -O3 -Iinclude -c main.cpp -o main.o
g++ -o Chess_3.exe main.o -static -static-libstdc++ -static-libgcc -static -s
```

You can delete the ```main.o``` file after compilation.
