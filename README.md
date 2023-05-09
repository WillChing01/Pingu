# Chess_3

A basic chess engine, currently in the early stages of development.

To make a move, enter the start and end squares separated by a space e.g. "e2 e4"

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

### Move ordering
- Search PV (principal variation) first
- Killer Heuristic
- Static evaluation exchange
