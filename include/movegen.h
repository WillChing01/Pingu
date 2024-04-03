#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include "constants.h"
#include "king.h"
#include "pawn.h"
#include "knight.h"
#include "magic.h"

class Board;

inline bool isInCheck(bool side)
{
    return false;
}

inline U64 getPinnedPieces(const U64* pieces, const U64* occupied, bool side)
{
    //generate attacks to the king.
    int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
    U64 b = occupied[0] | occupied[1];

    U64 pinned = 0;
    U64 attackers;

    //check for rook-like pins.
    attackers = magicRookAttacks(occupied[(int)(!side)], kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
    while (attackers)
    {
        pinned |= magicRookAttacks(b, popLSB(attackers)) & magicRookAttacks(b, kingPos);
    }

    //check for bishop-like pins.
    attackers = magicBishopAttacks(occupied[(int)(!side)], kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
    while (attackers)
    {
        pinned |= magicBishopAttacks(b, popLSB(attackers)) & magicBishopAttacks(b, kingPos);
    }

    return pinned;
}

//class to generate obviously winning captures/promotions (no need for SEE check).
class ObviousCaptureGenerator
{
    public:
        U64* pieces;
        U64* occupied;
        U64* pinned;
        int* enPassantSquare;
        bool side;
        int numChecks;
        U64 pinned;
        bool enPassant;
        int victimPieceType;
        U64 victimsBB;
        int currentVictimSquare;
        int attackerPieceType;
        U64 attackerBB;

        ObviousCaptureGenerator(U64* _pieces, U64* _occupied, U64* _pinned, int* _enPassantSquare)
        {
            pieces = _pieces;
            occupied = _occupied;
            pinned = _pinned;
            enPassantSquare = _enPassantSquare;
        }

        void reset(int _side, int _numChecks)
        {
            side = _side;
            numChecks = _numChecks;
            enPassant = false;
            victimPieceType = _nKing + (int)(!side);
            victimsBB = 0;
            attackerPieceType = _nKing + (int)(side);
            attackerBB = 0;
            update();
        }

        void updateCheck() {}
        void update()
        {
            if (numChecks > 0) {updateCheck(); return;}
            
            //check if we update the victim.
            if (attackerPieceType >> 1 == victimPieceType >> 1)
            {
                //check if we update victim piece type.
                if (!victimsBB)
                {
                    //check if all captures exhausted.
                    if (victimPieceType == _nPawns + (int)(!side))
                    {
                        if (enPassant) {attackerBB = 0; return;}

                        enPassant = true;
                        //test for enpassant.
                        if (*enPassantSquare != -1)
                        {
                            currentVictimSquare = *enPassantSquare;
                            attackerBB = pawnAttacks(1ull << *enPassantSquare, !side) & pieces[attackerPieceType];
                        }
                        return;
                    }

                    //update victim piece type.
                    while (!victimsBB && victimPieceType < _nPawns + 2)
                    {
                        victimPieceType += 2;
                        victimsBB = pieces[victimPieceType];
                    }
                    if (!victimsBB) {return;}
                }

                //update victim.
                currentVictimSquare = popLSB(victimsBB);
                attackerPieceType = _nPawns + (int)(side) + 2;
            }

            //update attacker.
            attackerPieceType -= 2;

            switch(attackerPieceType >> 1)
            {
                case _nPawns >> 1:
                    attackerBB = pawnAttacks(1ull << currentVictimSquare, !side);
                    break;
                case _nKnights >> 1:
                    attackerBB = knightAttacks(1ull << currentVictimSquare);
                    break;
                case _nBishops >> 1:
                    attackerBB = magicBishopAttacks(occupied[0] | occupied[1], currentVictimSquare);
                    break;
                case _nRooks >> 1:
                    attackerBB = magicRookAttacks(occupied[0] | occupied[1], currentVictimSquare);
                    break;
                case _nQueens >> 1:
                    attackerBB = magicQueenAttacks(occupied[0] | occupied[1], currentVictimSquare);
                    break;
                case _nKing >> 1:
                    attackerBB = kingAttacks(1ull << currentVictimSquare) & ~kingAttacks(pieces[_nKing + (int)(!side)]);
                    break;
            }

            //intersection with attackers.
            attackerBB &= pieces[attackerPieceType];

            //if no valid attacks, iterate again.
            if (!attackerBB) {update();}
        }

        U32 getNext()
        {
            if (!attackerBB)
            {
                update();
                if (!attackerBB) {return 0;}
            }

            int finishSquare = currentVictimSquare;
            int startSquare = popLSB(attackerBB);
            int pieceType = attackerPieceType;
            int finishPieceType = attackerPieceType;
            //check for promotion. only consider queen promotion for qsearch.
            if ((pieceType == _nPawns + (int)(side)) &&
                ((side && ((1ull << startSquare) & RANK_2)) ||
                (!side && ((1ull << startSquare) & RANK_7))))
            {
                finishPieceType = _nQueens + (int)(side);
            }
            int capturedPieceType = victimPieceType;
            bool enPassant = enPassant;

            //check if the piece is pinned.
            if ((*pinned & (1ull << startSquare)) || enPassant || pieceType >> 1 == _nKing >> 1)
            {
                U64 start = 1ull << startSquare;
                U64 finish = 1ull << finishSquare;
                pieces[pieceType] -= start;
                pieces[pieceType] += finish;
                pieces[capturedPieceType] -= finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                occupied[(int)(!side)] -= finish;
                bool isBad = isInCheck(side);

                //unmove pieces.
                pieces[pieceType] += start;
                pieces[pieceType] -= finish;
                pieces[capturedPieceType] += finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;
                occupied[(int)(!side)] += finish;

                if (isBad) {return getNext();}
            }

            //return the move.
            return (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (enPassant << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);
        }
};

#endif // MOVEGEN_H_INCLUDED
