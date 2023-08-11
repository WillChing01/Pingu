# Pingu

<img src="pingu.jpeg" alt="pingu" width="300"/>

UCI compatible chess engine built from scratch.

Many thanks to the advice on [Chess Programming Wiki](www.chessprogramming.org).

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
- Piece square tables for quiet moves

### Search
- Main search
  - Fail-soft alpha-beta
  - Principal variation search
  - Null move pruning
  - Late move reductions
- Quiescence search
  - Stand-pat
  - Forward prune SEE < 0
- Transposition tables
  - Always replace
  - Depth-preferred
- Repetition detection
- Iterative deepening at root
- Basic time management

### Evaluation
- Material
- Piece square tables
- Pawn hash table

### Testing
- Perft function

# Executable

### Github releases
Check out [releases](https://github.com/WillChing01/Pingu/releases/) to download the .exe for master branch.

### Compile with cmake

__Requires cmake, g++ and mingw__

Clone the repository, go to ```/Pingu``` directory and run these terminal commands:

```
mkdir build
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Pingu --config Release --
```

Alternatively, you can skip downloading mingw and replace "MinGW Makefiles" with another generator that you already have.

The executable will appear in ```/build``` directory.

### Compile manually
__Requires g++__

Clone the repository, go to ```/Pingu``` directory and run these terminal commands:

```
g++ -Wall -std=gnu++17 -fexceptions -O3 -Iinclude -c main.cpp -o main.o
g++ -o Pingu.exe main.o -static -static-libstdc++ -static-libgcc -static -s
```

You can delete the ```main.o``` file after compilation.
