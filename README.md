# Chess_3

A basic chess engine, currently in the early stages of development.

UCI compatible.

# Features

### Move generation
- Pseudo-legal move generation
- Bitboard representation
- Plain magic bitboards for sliding pieces

### Testing
- Perft function (initial and Kiwipete positions)

### Evaluation
- Material
- Piece square tables

### Search
- Alpha-beta with quiescence search
- Iterative deepening
- Transposition tables
    - Always replace
    - Depth-preferred

### Move ordering
- PV move from previous iteration
- Hash move
- Killer Heuristic
- Static evaluation exchange