#ifndef SEE_H_INCLUDED
#define SEE_H_INCLUDED

#include <array>

#include "constants.h"
#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "magic.h"

const std::array<int,6> seeValues = 
{{
    20000,
    1000,
    525,
    350,
    350,
    100,
}};

class SEE
{
    private:
        int gain[32] = {};
        const U64* pieces;
        const U64* occupied;

        U64 getLeastValuableAttacker(bool side, U64 attackersBB, U32 &attackingPieceType)
        {
            for (int i=_nPawns+(int)(side); i >= (int)_nKing+(int)(side); i-=2)
            {
                U64 x = attackersBB & pieces[i];
                if (x)
                {
                    attackingPieceType = i >> 1;
                    return x & (-x);
                }
            }
            return 0; // no attacker found.
        }

    public:
        SEE() {}

        SEE(const U64* _pieces, const U64* _occupied)
        {
            pieces = _pieces;
            occupied = _occupied;
        }

        int evaluate(U32 chessMove)
        {
            //perform static evaluation exchange (SEE).
            U32 finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            U32 attackingPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
            U32 capturedPieceType = (chessMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;

            bool side = attackingPieceType & 1;
            int d = 0;
            gain[0] = (capturedPieceType != 15 ? seeValues[capturedPieceType >> 1] : 0)
            + (attackingPieceType == ((chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET) ? 0 : seeValues[attackingPieceType >> 1] - seeValues[_nPawns >> 1]);
            U64 attackingPieceBB = 1ull << ((chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET);
            U64 occ = occupied[0] | occupied[1];
            if (chessMove & MOVEINFO_ENPASSANT_MASK) {occ ^= 1ull << (finishSquare - 8 + side * 16);}

            U64 attackersBB = attackingPieceBB;
            attackersBB |= kingAttacks(1ull << finishSquare) & (pieces[_nKing] | pieces[_nKing+1]);
            attackersBB |= pawnAttacks(1ull << finishSquare, 0) & pieces[_nPawns+1];
            attackersBB |= pawnAttacks(1ull << finishSquare, 1) & pieces[_nPawns];
            attackersBB |= knightAttacks(finishSquare) & (pieces[_nKnights] | pieces[_nKnights+1]);
            attackersBB |= magicRookAttacks(occ, finishSquare) & (pieces[_nRooks] | pieces[_nRooks+1] | pieces[_nQueens] | pieces[_nQueens+1]);
            attackersBB |= magicBishopAttacks(occ, finishSquare) & (pieces[_nBishops] | pieces[_nBishops+1] | pieces[_nQueens] | pieces[_nQueens+1]);

            attackingPieceType = attackingPieceType >> 1;

            do
            {
                d++; side = !side;
                gain[d] = -gain[d-1] + seeValues[attackingPieceType];
                attackersBB ^= attackingPieceBB;
                occ ^= attackingPieceBB;

                //update possible x-ray attacks.
                if (attackingPieceType == (_nRooks >> 1) || attackingPieceType == (_nQueens >> 1) || d == 1)
                {
                    //rook-like xray.
                    attackersBB |= magicRookAttacks(occ, finishSquare) & (pieces[_nRooks] | pieces[_nRooks+1] | pieces[_nQueens] | pieces[_nQueens+1]) & occ;
                }
                if (attackingPieceType == (_nPawns >> 1) || attackingPieceType == (_nBishops >> 1) || attackingPieceType == (_nQueens >> 1))
                {
                    //bishop-like xray.
                    attackersBB |= magicBishopAttacks(occ, finishSquare) & (pieces[_nBishops] | pieces[_nBishops+1] | pieces[_nQueens] | pieces[_nQueens+1]) & occ;
                }

                attackingPieceBB = getLeastValuableAttacker(side, attackersBB, attackingPieceType);
                if (((finishSquare >> 3) == 0 || (finishSquare >> 3) == 7) && (attackingPieceType == _nPawns >> 1))
                {
                    gain[d] += seeValues[_nQueens >> 1] - seeValues[_nPawns >> 1];
                    attackingPieceType = _nQueens >> 1;
                }
                if (gain[d] < 0) {break;}
            } while (attackingPieceBB);
            while (--d) {gain[d-1] = -std::max(-gain[d-1], gain[d]);}

            return gain[0];
        }
};

#endif // SEE_H_INCLUDED
