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
