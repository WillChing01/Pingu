#ifndef MOVEPICKER_H_INCLUDED
#define MOVEPICKER_H_INCLUDED

//move-picker class manages when moves are generated and when/how they are ordered.

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

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "constants.h"
#include "validate.h"
#include "board.h"

enum nodeType
{
    MAIN_NODE,
    Q_NODE
};

enum moveType
{
    HASH_MOVE,
    GOOD_CAPTURES,
    KILLER_MOVES,
    BAD_CAPTURES,
    QUIET_MOVES,

    Q_MOVES,

    END
};

class MovePicker
{
    private:
        Board* b;
        int ply;
        int numChecks;
        U32 hashMove = 0;

        nodeType node;

        int killerIndex = 0;

        void updateStage()
        {
            //generate and order moves on demand for the next stage.
            switch(stage)
            {
                case HASH_MOVE:
                    moveIndex = 0;
                    stage = GOOD_CAPTURES;
                    b->moveBuffer.clear();
                    b->generateCaptures(numChecks);
                    scoredMoves = b->orderCaptures();
                    break;
                case GOOD_CAPTURES:
                    stage = KILLER_MOVES;
                    break;
                case KILLER_MOVES:
                    stage = BAD_CAPTURES;
                    break;
                case BAD_CAPTURES:
                    moveIndex = 0;
                    stage = QUIET_MOVES;
                    b->moveBuffer.clear();
                    b->generateQuiets(numChecks);
                    scoredMoves = b->orderQuiets();
                    break;
                case QUIET_MOVES:
                    stage = END;
                    break;
                case Q_MOVES:
                    stage = END;
                    break;
                case END:
                    break;
            }
        }

    public:
        int moveIndex = 0;
        moveType stage;
        std::unordered_set<U32> singleQuiets = {};
        std::vector<std::pair<U32, int> > scoredMoves = {};

        MovePicker(Board* _b, int _ply, int _numChecks, U32 _hashMove)
        {
            //initializer for main nodes.
            b = _b;
            ply = _ply;
            numChecks = _numChecks;
            hashMove = _hashMove;

            node = MAIN_NODE;
            stage = HASH_MOVE;
        }

        MovePicker(Board* _b, int _numChecks)
        {
            //initializer for q nodes.
            b = _b;
            numChecks = _numChecks;

            node = Q_NODE;
            stage = Q_MOVES;
            
            if (numChecks == 0)
            {
                b->moveBuffer.clear();
                b->generateCaptures(0);
                scoredMoves = b->orderQMoves();
            }
            else
            {
                b->moveBuffer.clear();
                b->generateCaptures(numChecks);
                b->generateQuiets(numChecks);
                scoredMoves = b->orderQMovesInCheck();
            }
        }

        U32 getNext()
        {
            //get the next move to be played.
            //return 0 if no moves left.

            U32 move = 0;

            switch(stage)
            {
                case HASH_MOVE:
                {
                    if (moveIndex == 1 || hashMove == 0) {updateStage(); return getNext();}

                    move = hashMove;
                    ++moveIndex;

                    bool isValid = validate::isValidMove(move, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
                    if (!isValid) {return getNext();}

                    U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                    U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                    U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

                    bool isQuiet = (capturedPieceType == 15) && (pieceType == finishPieceType);
                    if (isQuiet) {singleQuiets.insert(move);}

                    break;
                }
                case GOOD_CAPTURES:
                {
                    if (moveIndex == (int)scoredMoves.size()) {updateStage(); return getNext();}
                    if (scoredMoves[moveIndex].second < 0) {updateStage(); return getNext();}

                    move = scoredMoves[moveIndex++].first;
                    if (move == hashMove) {return getNext();}

                    break;
                }
                case KILLER_MOVES:
                {
                    if (killerIndex == 2) {updateStage(); return getNext();}

                    move = b->killer.killerMoves[ply][killerIndex++];
                    if (singleQuiets.contains(move)) {return getNext();}

                    bool isValid = validate::isValidMove(move, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
                    if (!isValid) {return getNext();}

                    singleQuiets.insert(move);

                    break;
                }
                case BAD_CAPTURES:
                {
                    if (moveIndex == (int)scoredMoves.size()) {updateStage(); return getNext();}

                    move = scoredMoves[moveIndex++].first;
                    if (move == hashMove) {return getNext();}

                    break;
                }
                case QUIET_MOVES:
                {
                    if (moveIndex == (int)scoredMoves.size()) {updateStage(); return getNext();}

                    move = scoredMoves[moveIndex++].first;
                    if (singleQuiets.contains(move)) {return getNext();}

                    break;
                }
                case Q_MOVES:
                {
                    if (moveIndex == (int)scoredMoves.size()) {updateStage(); return getNext();}

                    move = scoredMoves[moveIndex++].first;
                    break;
                }
                case END:
                    break;
            }

            return move;
        }
};

#endif // MOVEPICKER_H_INCLUDED
