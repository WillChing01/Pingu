# Search

## Summary

This document contains general information about Pingu's search function.

For code examples, please refer to ```/include```.

## Contents

- [Move Generation](#move-generation)
- [Move Ordering](#move-ordering)
- [Root Search](#root-search)
- [Main Search](#main-search)
- [Quiescence Search](#quiescence-search)
- [Transposition Table](#transposition-table)
- [Parallel Search](#parallel-search)

## Move Generation

Move generation in Pingu has the following attributes:
- Legal move generation
- King/Knight/Pawn attacks via bitboard calculation
- Queen/Rook/Bishop attacks via plain magic bitboards

For each search type (root/main/quiescence) moves are generated differently. In the main and quiescence searches moves are generated in stages to save computation in cut-nodes. 

### Root search

- Generate all moves at once

### Main search

1. Hash move
2. Good captures + promotions
3. Killer moves (2 per ply)
4. Bad captures + promotions
5. Quiet moves

### Quiescence search

If not in check:
1. Good captures + promotions

If in check:
1. Good captures + promotions
2. Bad captures + promotions
3. Quiet moves

## Move Ordering

Good move ordering is critical for the efficiency of alpha-beta, and it occurs in two phases:

- Staged move generation loosely orders the moves
- Specific ordering techniques tightly order the moves after each stage of move generation

### Root node

- Iterative deepening obtains a PV move which is played first at the next iteration
- The remaining moves are ranked in descending order based on the number of nodes searched in the previous iteration

### Captures

__MVV-LVA__

A fast way to order captures in order of usefulness.

Prioritize the most valuable victim first, breaking ties by prioritizing the least valuable attacker e.g. PxQ, QxQ, NxR, ...

Equal captures of heavy pieces are prioritized over winning captures of small pieces since removing heavy pieces from the board should reduce the subtree size, allowing for a reasonable search score to be quickly established.

__SEE__

SEE (static exchange evaluation) involves swapping off material at a given square to estimate the value of a capture (the SEE score). It is more accurate but slower than MVV-LVA.

The SEE routine in Pingu takes pins into account but it allows capturing kings and ignores the possibility of in-between moves - it is faster but less accurate than quiescence search.

__Summary__

MVV-LVA is used to quickly assess if a capture/promotion is winning or equal.

Failing this, SEE is used as a fallback - if SEE score < 0 then the capture is marked as bad.

1. Good captures + promotions are ordered by MVV-LVA
2. Bad captures + promotions are ordered by SEE score

### Quiet moves

__Killer moves__

The killer moves are ordered to play the most recent killer first.

__History tables__

A table indexed like `history[pieceType][toSquare]`.

At a cut-node, the cut-move has its history incremented by `depth^2`. Quiet moves played in a cut-node which do not fail high are decremented by `depth^2`.

The history table is only updated when `depth >= 5` to avoid filling it with noise at lower depths.

History scores are capped at `±2^20` and if scores exceed this value all entries in the table are divided by 16.

At the start of each game the history table is cleared, and before each search all entries are divided by 8.

__Piece square tables__

The difference in piece square table scores (from the middlegame) provides a useful move ordering technique when history scores are not yet established:

`pst[toSquare] - pst[fromSquare]`

__Summary__

Quiet moves have their history and pst scores added together to provide a score for ordering:

`score = history[pieceType][toSquare] + (pst[toSquare] - pst[fromSquare])`


## Root Search

## Main Search

## Quiescence Search

## Transposition Table

## Parallel Search
