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
