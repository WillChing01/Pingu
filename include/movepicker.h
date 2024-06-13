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

enum moveType
{
    HASH_MOVE = 0,
    GOOD_CAPTURES = 1,
    KILLER_MOVES = 2,
    BAD_CAPTURES = 3,
    QUIET_MOVES = 4,

    Q_CAPTURES = 5,
    Q_BAD_CAPTURES = 6,
    Q_EVASIONS = 7
};

class MovePicker
{
    private:
        Board* b;
        int ply;
        U32 numChecks;
        U32 hashMove = 0;

        int killerIndex = 0;

    public:
        size_t moveIndex = 0;
        int stage;
        std::unordered_set<U32> singleQuiets = {};
        std::vector<std::pair<U32, int> > scoredMoves = {};
        std::vector<std::pair<U32, int> > badCaptures = {};

        MovePicker(Board* _b, int _ply, U32 _numChecks, U32 _hashMove)
        {
            //initializer for main nodes.
            b = _b;
            ply = _ply;
            numChecks = _numChecks;
            hashMove = _hashMove;

            stage = HASH_MOVE;
        }

        U32 getNext()
        {
            //get the next move to be played.
            //return 0 if no moves left.

            switch(stage)
            {
                case HASH_MOVE:
                {
                    while (moveIndex == 0 && hashMove != 0)
                    {
                        ++moveIndex;

                        bool isValid = validate::isValidMove(hashMove, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
                        if (!isValid) {break;}

                        U32 capturedPieceType = (hashMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                        U32 pieceType = (hashMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 finishPieceType = (hashMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

                        bool isQuiet = (capturedPieceType == 15) && (pieceType == finishPieceType);
                        if (isQuiet) {singleQuiets.insert(hashMove);}

                        return hashMove;
                    }

                    moveIndex = 0;
                    b->moveBuffer.clear();
                    b->generateCaptures(numChecks);
                    scoredMoves = b->orderCaptures();

                    ++stage;
                    [[fallthrough]];
                }
                case GOOD_CAPTURES:
                {
                    while (moveIndex != scoredMoves.size())
                    {
                        U32 move = scoredMoves[moveIndex++].first;
                        if (move == hashMove) {continue;}

                        //check if see is necessary.
                        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;

                        bool mustCheck = (capturedPieceType == 15) || (pieceType >= _nQueens && seeValues[pieceType >> 1] > seeValues[capturedPieceType >> 1]);
                        if (mustCheck)
                        {
                            int seeScore = b->see.evaluate(move);
                            if (seeScore < 0) {badCaptures.push_back(std::pair<U32, int>(move, seeScore)); continue;}
                        }

                        return move;
                    }

                    ++stage;
                    [[fallthrough]];
                }
                case KILLER_MOVES:
                {
                    while (killerIndex != 2)
                    {
                        U32 move = b->killer.killerMoves[ply][killerIndex++];

                        if (singleQuiets.contains(move)) {continue;}

                        bool isValid = validate::isValidMove(move, numChecks > 0, b->side, b->current, b->pieces, b->occupied);
                        if (!isValid) {continue;}

                        singleQuiets.insert(move);
                        return move;
                    }

                    moveIndex = 0;
                    std::sort(badCaptures.begin(), badCaptures.end(), [](auto &a, auto &b) {return a.second > b.second;});

                    ++stage;
                    [[fallthrough]];
                }
                case BAD_CAPTURES:
                {
                    while (moveIndex != badCaptures.size())
                    {
                        U32 move = badCaptures[moveIndex++].first;

                        if (move == hashMove) {continue;}
                        return move;
                    }

                    moveIndex = 0;
                    b->moveBuffer.clear();
                    b->generateQuiets(numChecks);
                    scoredMoves = b->orderQuiets();

                    ++stage;
                    [[fallthrough]];
                }
                case QUIET_MOVES:
                {
                    while (moveIndex != scoredMoves.size())
                    {
                        U32 move = scoredMoves[moveIndex++].first;

                        if (singleQuiets.contains(move)) {continue;}
                        return move;
                    }

                    return 0;
                    break;
                }
            }

            return 0;
        }
};

class QMovePicker
{
    private:
        Board *b;
        U32 numChecks;
        size_t moveIndex = 0;
        int stage = 0;
        std::vector<std::pair<U32, int> > scoredMoves = {};
        std::vector<std::pair<U32, int> > badCaptures = {};

    public:
        QMovePicker() {}

        QMovePicker(Board* _b, U32 _numChecks)
        {
            b = _b;
            numChecks = _numChecks;

            stage = Q_CAPTURES;
            b->moveBuffer.clear();
            b->generateCaptures(numChecks);
            scoredMoves = b->orderCaptures();
        }

        U32 getNext()
        {
            switch(stage)
            {
                case Q_CAPTURES:
                {
                    while (moveIndex != scoredMoves.size())
                    {
                        U32 move = scoredMoves[moveIndex++].first;

                        //check if see is necessary.
                        U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;

                        bool mustCheck = (capturedPieceType == 15) || (pieceType >= _nQueens && seeValues[pieceType >> 1] > seeValues[capturedPieceType >> 1]);
                        if (mustCheck)
                        {
                            int seeScore = b->see.evaluate(move);
                            if (seeScore < 0)
                            {
                                if (numChecks > 0) {badCaptures.push_back(std::pair<U32, int>(move, seeScore));}
                                continue;
                            }
                        }

                        return move;
                    }
                    
                    if (numChecks == 0) {return 0;}

                    moveIndex = 0;
                    std::sort(badCaptures.begin(), badCaptures.end(), [](auto &a, auto &b) {return a.second > b.second;});

                    ++stage;
                    [[fallthrough]];
                }
                case Q_BAD_CAPTURES:
                {
                    while (moveIndex != badCaptures.size())
                    {
                        return badCaptures[moveIndex++].first;
                    }

                    moveIndex = 0;
                    b->moveBuffer.clear();
                    b->generateQuiets(numChecks);
                    scoredMoves = b->orderQuiets();

                    ++stage;
                    [[fallthrough]];
                }
                case Q_EVASIONS:
                {
                    while (moveIndex != scoredMoves.size())
                    {
                        return scoredMoves[moveIndex++].first;
                    }
                    return 0;
                    break;
                }
            }

            return 0;
        }
};

#endif // MOVEPICKER_H_INCLUDED
