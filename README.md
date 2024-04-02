<div align="center">

  <img src="pingu.jpeg" alt="pingu" width="300"/>

  # Pingu

  [![License][license-badge]][license-link]
  [![Release][release-badge]][release-link]
  [![Commits][commits-badge]][commits-link]

Pingu is a chess engine built from scratch. It communicates via [UCI protocol](https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf).

</div>

# Usage

Pingu is a command-line program - in order to interact with other engines or people it is recommended to use a suitable GUI e.g. [cutechess](https://github.com/cutechess/cutechess).

When running Pingu in the command line, type 'help' for a list of available commands.

Pingu accepts many of the usual UCI commands (go/stop/position etc.) and it has some additional custom commands.

# Rating

| Version | CCRL Blitz | CCRL 40/15 |
| ------: | ---------: | ---------: |
| 3.0.0   | N/A        | 2828       |
| 2.0.0   | 2528       | 2614       |
| 1.0.0   | 2162       | N/A        |

More information on [CCRL](https://www.computerchess.org.uk/ccrl/).

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
- Winning captures ordered by MVV/LVA
- Losing captures ordered by static exchange evaluation
- Quiet moves ordered by history heuristic

### Search

- Root search
  - Iterative deepening
  - Aspiration windows
  - Principal variation search
- Main search
  - Fail-soft alpha-beta
  - Principal variation search
  - Reverse futility pruning
  - Null move pruning
  - Internal iterative reduction
  - Late move reductions
  - Futility pruning
  - Late move pruning
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

- NNUE
  - Network structure
    1. 768 -> 64 -> 8 -> 1
    2. Fully connected layers
    3. Clipped ReLU activation
  - Quantized weights
  - AVX2 instructions

NNUE trained from Pingu 2.0.0 self-play data.

# Thanks

[Chess Programming Wiki](https://www.chessprogramming.org) for its useful resources.

[CCRL](https://www.computerchess.org.uk/ccrl/) for testing Pingu.


[commits-badge]:https://img.shields.io/github/commits-since/WillChing01/Pingu/latest?style=for-the-badge
[commits-link]:https://github.com/WillChing01/Pingu/commits/master
[release-badge]:https://img.shields.io/github/v/release/WillChing01/Pingu?style=for-the-badge&label=official%20release
[release-link]:https://github.com/WillChing01/Pingu/releases/latest
[license-badge]:https://img.shields.io/github/license/WillChing01/Pingu?style=for-the-badge&label=license&color=success
[license-link]:https://github.com/WillChing01/Pingu/blob/master/LICENSE
