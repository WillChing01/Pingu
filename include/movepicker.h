#ifndef MOVEPICKER_H_INCLUDED
#define MOVEPICKER_H_INCLUDED

// move-picker class manages when moves are generated and when/how they are ordered.

/*

Regular Node:
    - Hash move
    - Good captures/promotions
    - Killer moves (x2)
    - Bad captures/promotions
    - Quiet moves

Quiescence Node:
    - Good captures/promotions
    - Bad captures/promotions (if in check)
    - Quiet moves (if in check)

*/

#include "constants.h"
#include "validate.h"
#include "board.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

enum moveType { HASH_MOVE = 0, GOOD_CAPTURES = 1, KILLER_MOVES = 2, BAD_CAPTURES = 3, QUIET_MOVES = 4 };

enum qMoveType { Q_CAPTURES = 0, Q_BAD_CAPTURES = 1, Q_EVASIONS = 2 };

class MovePicker {
  private:
    Board* b;
    int ply;
    U32 numChecks;
    U32 hashMove = 0;

    int killerIndex = 0;

  public:
    size_t moveIndex = 0;
    moveType stage;
    std::unordered_set<U32> singleQuiets = {};

    MovePicker(Board* _b, int _ply, U32 _numChecks, U32 _hashMove) {
        // initializer for main nodes.
        b = _b;
        ply = _ply;
        numChecks = _numChecks;
        hashMove = _hashMove;

        stage = HASH_MOVE;
    }

    U32 getNext() {
        static const void* stageLabels[5] = {
            &&hash_move, &&good_captures, &&killer_moves, &&bad_captures, &&quiet_moves};

        goto* stageLabels[stage];

    hash_move:
        while (moveIndex == 0 && hashMove != 0) {
            ++moveIndex;

            bool isValid = validate::isValidMove(hashMove, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
            if (!isValid) {
                break;
            }

            U32 capturedPieceType = (hashMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
            U32 pieceType = (hashMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 finishPieceType = (hashMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

            bool isQuiet = (capturedPieceType == 15) && (pieceType == finishPieceType);
            if (isQuiet) {
                singleQuiets.insert(hashMove);
            }

            return hashMove;
        }

        moveIndex = 0;
        b->moveBuffer.clear();
        b->generateCaptures(numChecks);
        b->orderCaptures(ply);
        b->badCaptures[ply].clear();

        stage = GOOD_CAPTURES;

    good_captures:
        while (moveIndex != b->moveCache[ply].size()) {
            U32 move = b->moveCache[ply][moveIndex++].first;
            if (move == hashMove) {
                continue;
            }

            if (shouldCheckSEE(move)) {
                int seeScore = b->see.evaluate(move);
                if (seeScore < 0) {
                    b->badCaptures[ply].push_back(std::pair<U32, int>(move, seeScore));
                    continue;
                }
            }

            return move;
        }

        stage = KILLER_MOVES;

    killer_moves:
        while (killerIndex != 2) {
            U32 move = b->killer.killerMoves[ply][killerIndex++];

            if (singleQuiets.contains(move)) {
                continue;
            }

            bool isValid = validate::isValidMove(move, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
            if (!isValid) {
                continue;
            }

            singleQuiets.insert(move);
            return move;
        }

        moveIndex = 0;
        std::sort(b->badCaptures[ply].begin(), b->badCaptures[ply].end(), [](auto& a, auto& b) {
            return a.second > b.second;
        });

        stage = BAD_CAPTURES;

    bad_captures:
        while (moveIndex != b->badCaptures[ply].size()) {
            U32 move = b->badCaptures[ply][moveIndex++].first;
            if (move == hashMove) {
                continue;
            }

            return move;
        }

        moveIndex = 0;
        b->moveBuffer.clear();
        b->generateQuiets(numChecks);
        b->orderQuiets(ply);

        stage = QUIET_MOVES;

    quiet_moves:
        while (moveIndex != b->moveCache[ply].size()) {
            U32 move = b->moveCache[ply][moveIndex++].first;

            if (singleQuiets.contains(move)) {
                continue;
            }
            return move;
        }

        return 0;
    }
};

class QMovePicker {
  private:
    Board* b;
    U32 numChecks;
    int ply;
    size_t moveIndex = 0;
    qMoveType stage;

  public:
    QMovePicker() {}

    QMovePicker(Board* _b, int _ply, U32 _numChecks) {
        b = _b;
        ply = _ply;
        numChecks = _numChecks;

        stage = Q_CAPTURES;

        b->moveBuffer.clear();
        b->generateCaptures(numChecks);
        b->orderCaptures(ply);
        b->badCaptures[ply].clear();
    }

    U32 getNext() {
        static const void* stageLabels[5] = {&&q_captures, &&q_bad_captures, &&q_evasions};

        goto* stageLabels[stage];

    q_captures:
        while (moveIndex != b->moveCache[ply].size()) {
            U32 move = b->moveCache[ply][moveIndex++].first;

            if (shouldCheckSEE(move)) {
                int seeScore = b->see.evaluate(move);
                if (seeScore < 0) {
                    if (numChecks > 0) {
                        b->badCaptures[ply].push_back(std::pair<U32, int>(move, seeScore));
                    }
                    continue;
                }
            }

            return move;
        }

        if (numChecks == 0) {
            return 0;
        }

        moveIndex = 0;
        std::sort(b->badCaptures[ply].begin(), b->badCaptures[ply].end(), [](auto& a, auto& b) {
            return a.second > b.second;
        });

        stage = Q_BAD_CAPTURES;

    q_bad_captures:
        while (moveIndex != b->badCaptures[ply].size()) {
            return b->badCaptures[ply][moveIndex++].first;
        }

        moveIndex = 0;
        b->moveBuffer.clear();
        b->generateQuiets(numChecks);
        b->orderQuiets(ply);

        stage = Q_EVASIONS;

    q_evasions:
        while (moveIndex != b->moveCache[ply].size()) {
            return b->moveCache[ply][moveIndex++].first;
        }

        return 0;
    }
};

#endif // MOVEPICKER_H_INCLUDED
