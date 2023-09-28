# Pingu

<img src="pingu.jpeg" alt="pingu" width="300"/>

Pingu is a chess engine built from scratch. It communicates via [UCI protocol](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf).

Play against Pingu on [Lichess](https://lichess.org/@/WilliamEngine)!

Many thanks to the advice on [Chess Programming Wiki](https://www.chessprogramming.org).

# Usage

Pingu is a command-line program - in order to interact with other engines or people it is recommended to use a suitable GUI e.g. [cutechess](https://github.com/cutechess/cutechess).

When running Pingu in the command line, type 'help' for a list of available commands.

Pingu accepts many of the usual UCI commands (go/stop/position etc.) and it has some additional custom commands.

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

- Root search
  1. PV move
  2. Others ordered by descending subtree size
- Main search
  1. Hash move
  2. Good captures + promotions
  3. Killers
  4. Bad captures
  5. Quiet moves
- Quiescence search
  1. Good captures + promotions
  2. Check evasions
- Captures ordered by static exchange evaluation
- Quiet moves ordered by history heuristic

### Search

- Root search
  - Iterative deepening
  - Aspiration windows
  - Principal variation search
- Main search
  - Fail-soft alpha-beta
  - Principal variation search
  - Null move pruning
  - Internal iterative reduction
  - Late move reductions
  - Repetition detection
  - Transposition tables
    1. Always replace
    2. Depth-preferred
- Quiescence search
  - Fail-soft alpha-beta
  - Stand-pat
  - Forward prune SEE < 0
- Basic time management

### Evaluation
- Tapered evaluation
- Material
- Piece square tables
- Mobility (bishop/rook)

Evaluation features are tuned via Texel's method - logistic regression via gradient descent.

# Installation

### Github releases

Check out [releases](https://github.com/WillChing01/Pingu/releases/) to download the .exe for master branch.

The ```dev-build``` pre-release contains the most up-to-date executable for Pingu.

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
g++ -o Pingu.exe main.o -static -static-libstdc++ -static-libgcc -s
```

You can delete the ```main.o``` file after compilation.
