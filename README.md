# Pingu

<img src="pingu.jpeg" alt="pingu" width="300"/>

UCI chess engine built from scratch.

Pingu is a command-line program - in order to interact with other engines or people it is recommended to use a suitable GUI e.g. [cutechess](https://github.com/cutechess/cutechess).

Play against Pingu on [Lichess](https://lichess.org/@/WilliamEngine)!

Many thanks to the advice on [Chess Programming Wiki](https://www.chessprogramming.org).

# Rating

| Version | CCRL Blitz |
| ------: | ---------: |
| 1.0.0   | 2250       |

More information on [CCRL Blitz](http://ccrl.chessdom.com/ccrl/404/).

# Features

### Move generation
- Staged move generation
- Bitboard representation
- Plain magic bitboards

### Move ordering
1. Hash move (PV move at root)
2. Good captures / promotions
3. Killers
4. Bad captures
5. Quiet moves
- Captures ordered by static exchange evaluation
- Quiets ordered by history heuristic

### Search
- Main search
  - Fail-soft alpha-beta
  - Principal variation search
  - Null move pruning
  - Internal iterative reduction
  - Late move reductions
- Quiescence search
  - Search captures/promotions/check-evasions
  - Stand-pat
  - Forward prune SEE < 0
- Transposition tables
  - Always replace
  - Depth-preferred
- Repetition detection
- Iterative deepening at root
- Basic time management

### Evaluation
- Tapered evaluation
- Material
- Piece square tables
- Pawn hash table

Evaluation features are tuned via Texel's method - logistic regression via gradient descent.

# Installation

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
