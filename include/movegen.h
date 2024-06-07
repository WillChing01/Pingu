#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include <vector>

#include "constants.h"
#include "king.h"
#include "pawn.h"
#include "knight.h"
#include "magic.h"
#include "util.h"

class CaptureGenerator
{
    private:
        U64* pieces;
        U64* occupied;
        const int* enPassantSquare;
        const bool* side;
        bool isQSearch;

        U64 pinned;
        int numChecks;

        bool finishedRegularCaptures;
        bool finishedEnPassant;
        bool finishedPromotions;
        int promotionPieceType;
        std::vector<U32> promotionMoveBuffer;

        int victimPieceType;
        U64 victimsBB;
        int currentVictimSquare;
        int attackerPieceType;
        U64 attackerBB;

        void update()
        {
            switch(numChecks)
            {
                case 0:
                    updateNoCheck();
                    break;
                case 1:
                    updateSingleCheck();
                    break;
                case 2:
                    updateDoubleCheck();
                    break;
            }
        }

        void updateNoCheck()
        {
            //regular captures.
            while (!finishedRegularCaptures)
            {
                while (true)
                {
                    while (attackerPieceType >= _nKing + 2)
                    {
                        //update attacker and try again.
                        attackerPieceType -= 2;
                        attackerBB = pieces[attackerPieceType];
                        if (!attackerBB) {continue;}

                        //intersect pieces with attacks.
                        switch(attackerPieceType >> 1)
                        {
                            case _nPawns >> 1:
                                attackerBB &= pawnAttacks(1ull << currentVictimSquare, !side);
                                break;
                            case _nKnights >> 1:
                                attackerBB &= knightAttacks(1ull << currentVictimSquare);
                                break;
                            case _nBishops >> 1:
                                attackerBB &= magicBishopAttacks(occupied[0] | occupied[1], currentVictimSquare);
                                break;
                            case _nRooks >> 1:
                                attackerBB &= magicRookAttacks(occupied[0] | occupied[1], currentVictimSquare);
                                break;
                            case _nQueens >> 1:
                                attackerBB &= magicQueenAttacks(occupied[0] | occupied[1], currentVictimSquare);
                                break;
                            case _nKing >> 1:
                                attackerBB &= kingAttacks(1ull << currentVictimSquare) & ~kingAttacks(pieces[_nKing + (int)(!side)]);
                                break;
                        }

                        //found some attackers. time to exit.
                        if (attackerBB) {return;}
                    }

                    //update current victim square and try again.
                    if (!victimsBB) {break;}
                    currentVictimSquare = popLSB(victimsBB);
                    attackerPieceType = _nPawns + 2 + (int)(side);
                }

                //update victim piece type.
                while (!victimsBB && victimPieceType < _nPawns)
                {
                    victimPieceType += 2;
                    victimsBB = pieces[victimPieceType];
                }

                if (victimsBB)
                {
                    currentVictimSquare = popLSB(victimsBB);
                    attackerPieceType = _nPawns + 2 + (int)(side);
                    continue;
                }

                //we have exhausted all regular captures.
                finishedRegularCaptures = true;
                break;
            }

            //enPassant.
            if (!finishedEnPassant)
            {
                finishedEnPassant = true;

                attackerPieceType = _nPawns + (int)(side);
                victimPieceType = _nPawns + (int)(!side);

                currentVictimSquare = *enPassantSquare;
                attackerBB = pawnAttacks(1ull << currentVictimSquare, !side) & pieces[_nPawns+(int)(side)];

                if (attackerBB) {return;}
            }

            //promotion without capture.
            while (!finishedPromotions && promotionPieceType + 2 < _nPawns)
            {
                attackerPieceType = _nPawns + (int)(side);
                victimPieceType = 15;

                U64 p = occupied[0] | occupied[1];
                attackerBB = side ? (RANK_1 & (~p)) << 1 : (RANK_8 & (~p)) >> 1;
                attackerBB &= pieces[_nPawns + (int)(side)];

                if (attackerBB)
                {
                    promotionPieceType += 2;
                    return;
                }

                finishedPromotions = true;
                break;
            }
        }

        void updateSingleCheck()
        {
            //single check - only one victim can be captured.

            while (!finishedRegularCaptures)
            {
                while (attackerPieceType >= _nKing + 2)
                {
                    //update attacker and try again.
                    attackerPieceType -= 2;
                    attackerBB = pieces[attackerPieceType];
                    if (!attackerBB) {continue;}

                    //intersect pieces with attacks.
                    switch(attackerPieceType >> 1)
                    {
                        case _nPawns >> 1:
                            attackerBB &= pawnAttacks(1ull << currentVictimSquare, !side);
                            break;
                        case _nKnights >> 1:
                            attackerBB &= knightAttacks(1ull << currentVictimSquare);
                            break;
                        case _nBishops >> 1:
                            attackerBB &= magicBishopAttacks(occupied[0] | occupied[1], currentVictimSquare);
                            break;
                        case _nRooks >> 1:
                            attackerBB &= magicRookAttacks(occupied[0] | occupied[1], currentVictimSquare);
                            break;
                        case _nQueens >> 1:
                            attackerBB &= magicQueenAttacks(occupied[0] | occupied[1], currentVictimSquare);
                            break;
                        case _nKing >> 1:
                            attackerBB &= kingAttacks(1ull << currentVictimSquare) & ~kingAttacks(pieces[_nKing + (int)(!side)]);
                            break;
                    }

                    //found some attackers. time to exit.
                    if (attackerBB) {return;}
                }

                finishedRegularCaptures = true;
                break;
            }

            if (!finishedEnPassant)
            {
                finishedEnPassant = true;
                if (currentVictimSquare != *enPassantSquare) {return;}

                attackerPieceType = _nPawns + (int)(side);
                attackerBB = pawnAttacks(1ull << currentVictimSquare, !side) & pieces[_nPawns+(int)(side)];

                if (attackerBB) {return;}
            }
        }

        void updateDoubleCheck()
        {
            //only consider king captures.
        }

        U32 getNextPromotion()
        {
            //for non-capture promotions.
            if (!attackerBB)
            {
                update();
                if (!attackerBB) {return 0;}
            }

            int startSquare = popLSB(attackerBB);
            int finishSquare = side ? startSquare >> 1 : startSquare << 1;
            int capturedPieceType = 15;
            int pieceType = _nPawns + (int)(side);

            //check if piece is pinned.
            if (pinned & (1ull << startSquare))
            {
                U64 start = 1ull << startSquare;
                U64 finish = 1ull << finishSquare;
                pieces[pieceType] -= start;
                pieces[pieceType] += finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                bool isBad = util::isInCheck(side, pieces, occupied);

                pieces[pieceType] += start;
                pieces[pieceType] -= finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;

                if (isBad) {return getNextPromotion();}
            }

            //return the move.
            return (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (false << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (promotionPieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);
        }

    public:
        CaptureGenerator() {}

        CaptureGenerator(U64* _pieces, U64* _occupied, const bool* side, const int* _enPassantSquare, bool _isQSearch)
        {
            pieces = _pieces;
            occupied = _occupied;
            side = side;
            enPassantSquare = _enPassantSquare;
            isQSearch = _isQSearch;
        }

        void reset(int _numChecks, U64 _pinned)
        {
            numChecks = _numChecks;
            pinned = _pinned;

            promotionMoveBuffer.clear();

            victimPieceType = _nKing + (int)(!side);
            victimsBB = 0;
            attackerPieceType = _nPawns + (int)(side);
            attackerBB = 0;

            update();
        }

        U32 getNext()
        {
            if (victimPieceType == 15) {return getNextPromotion();}

            if (!isQSearch && promotionMoveBuffer.size() != 0)
            {
                U32 move = promotionMoveBuffer.back();
                promotionMoveBuffer.pop_back();
                return move;
            }

            if (!attackerBB)
            {
                update();
                if (!attackerBB) {return 0;}
            }

            int startSquare = popLSB(attackerBB);
            int finishSquare = currentVictimSquare;
            int capturedPieceType = victimPieceType;
            int pieceType = attackerPieceType;

            //check if the piece is pinned.
            if ((pinned & (1ull << startSquare)) || finishSquare == *enPassantSquare || pieceType >> 1 == _nKing >> 1)
            {
                U64 start = 1ull << startSquare;
                U64 finish = 1ull << finishSquare;
                pieces[pieceType] -= start;
                pieces[pieceType] += finish;
                pieces[capturedPieceType] -= finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                occupied[(int)(!side)] -= finish;
                bool isBad = util::isInCheck(side, pieces, occupied);

                //unmove pieces.
                pieces[pieceType] += start;
                pieces[pieceType] -= finish;
                pieces[capturedPieceType] += finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;
                occupied[(int)(!side)] += finish;

                if (isBad) {return getNext();}
            }

            //check for promotions.
            bool isPromotion = (pieceType == _nPawns + (int)(side)) && ((side && ((1ull << startSquare) & RANK_2)) || (!side && ((1ull << startSquare) & RANK_7)));
            int finishPieceType = isPromotion ? _nQueens + (int)(side) : pieceType;

            if (isPromotion && !isQSearch)
            {
                for (int i=_nKnights+(int)(side);i<=_nRooks+(int)(side);i+=2)
                {
                    promotionMoveBuffer.push_back(
                        (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
                        (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
                        (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
                        (false << MOVEINFO_ENPASSANT_OFFSET) |
                        (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
                        (i << MOVEINFO_FINISHPIECETYPE_OFFSET)
                    );
                }
            }

            //return the move.
            return (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (finishSquare == *enPassantSquare << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);
        }
};

#endif // MOVEGEN_H_INCLUDED
