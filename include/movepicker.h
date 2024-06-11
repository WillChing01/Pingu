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
    END
};

class MovePicker
{
    private:
        Board* b;
        int ply;
        int numChecks;
        U32 hashMove;

        nodeType node;

        int moveIndex = 0;
        std::vector<std::pair<U32, int> > badCaptures = {};

        void updateStage()
        {
            //generate and order moves on demand for the next stage.

            moveIndex = 0;
            switch(stage)
            {
                case HASH_MOVE:
                    stage = GOOD_CAPTURES;
                    b->moveBuffer.clear();
                    b->generateCaptures(numChecks);
                    orderMoves();
                    break;
                case GOOD_CAPTURES:
                    switch(node)
                    {
                        case MAIN_NODE:
                            stage = KILLER_MOVES;
                            break;
                        case Q_NODE:
                            stage = numChecks > 0 ? BAD_CAPTURES : END;
                            break;
                    }
                    break;
                case KILLER_MOVES:
                    stage = BAD_CAPTURES;
                    orderMoves();
                    break;
                case BAD_CAPTURES:
                    stage = QUIET_MOVES;
                    b->moveBuffer.clear();
                    b->generateQuiets(numChecks);
                    orderMoves();
                    break;
                case QUIET_MOVES:
                    stage = END;
                    break;
                case END:
                    break;
            }
        }

        void orderMoves()
        {
            //score and then order moves in moveBuffer.
            switch(stage)
            {
                case HASH_MOVE:
                    break;
                case GOOD_CAPTURES:
                {
                    //order by mvv/lva.
                    scoredMoves.clear();
                    for (const auto &move: b->moveBuffer)
                    {
                        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;

                        int score = 32 * (15 - capturedPieceType) + pieceType;
                        if (capturedPieceType == 15)
                        {
                            U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
                            score += (15 - finishPieceType);
                        }

                        scoredMoves.push_back(std::pair<U32, int>(move, score));
                    }
                    std::sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
                    break;
                }
                case KILLER_MOVES:
                    break;
                case BAD_CAPTURES:
                {
                    std::sort(badCaptures.begin(), badCaptures.end(), [](auto &a, auto &b) {return a.second > b.second;});
                    break;
                }
                case QUIET_MOVES:
                {
                    //order by history + pst.
                    scoredMoves.clear();
                    for (const auto &move: b->moveBuffer)
                    {
                        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 startSquare = (move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
                        U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

                        int moveScore = b->history.scores[pieceType][finishSquare];
                        switch(pieceType & 1)
                        {
                            case 0:
                                moveScore += PIECE_TABLES_START[pieceType >> 1][finishSquare ^ 56] - PIECE_TABLES_START[pieceType >> 1][startSquare ^ 56];
                                break;
                            case 1:
                                moveScore += PIECE_TABLES_START[pieceType >> 1][finishSquare] - PIECE_TABLES_START[pieceType >> 1][startSquare];
                                break;
                        }

                        scoredMoves.push_back(std::pair<U32,int>(move, moveScore));
                    }
                    std::sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
                    break;
                }
                case END:
                    break;
            }
        }

    public:
        moveType stage;
        std::unordered_set<U32> singleQuiets = {};
        std::vector<std::pair<U32, int> > scoredMoves = {};

        MovePicker(Board* _b, int _ply, int _numChecks, U32 _hashMove, nodeType _node)
        {
            b = _b;
            ply = _ply;
            numChecks = _numChecks;
            hashMove = _hashMove;

            node = _node;
            switch(node)
            {
                case MAIN_NODE:
                {
                    stage = HASH_MOVE;
                    break;
                }
                case Q_NODE:
                {
                    stage = GOOD_CAPTURES;
                    break;
                }
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

                    move = scoredMoves[moveIndex++].first;
                    if (move == hashMove) {return getNext();}

                    //see if necessary.
                    U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                    U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;

                    bool needToSee = ((pieceType >= _nQueens) && (pieceType < capturedPieceType)) || (capturedPieceType == 15);
                    if (needToSee)
                    {
                        int seeScore = b->see.evaluate(move);
                        if (seeScore < 0)
                        {
                            switch(node)
                            {
                                case MAIN_NODE:
                                    badCaptures.push_back(std::pair<U32, int>(move, seeScore));
                                    break;
                                case Q_NODE:
                                    break;
                            }
                            return getNext();
                        }
                    }

                    break;
                }
                case KILLER_MOVES:
                {
                    if (moveIndex == 2) {updateStage(); return getNext();}

                    move = b->killer.killerMoves[ply][moveIndex++];
                    if (singleQuiets.contains(move)) {return getNext();}

                    bool isValid = validate::isValidMove(move, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
                    if (!isValid) {return getNext();}

                    singleQuiets.insert(move);

                    break;
                }
                case BAD_CAPTURES:
                {
                    if (moveIndex == (int)badCaptures.size()) {updateStage(); return getNext();}

                    move = badCaptures[moveIndex++].first;
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
                case END:
                    break;
            }

            return move;
        }
};

#endif // MOVEPICKER_H_INCLUDED
