#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "constants.h"
#include "bitboard.h"

#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "magic.h"

#include "killer.h"
#include "history.h"
#include "see.h"

#include "evaluation.h"
#include "nnue.h"

#include "transposition.h"

#include "format.h"

inline std::string positionToFen(const U64* pieces, const gameState &current, bool side);

struct moveInfo
{
    U32 pieceType;
    U32 startSquare;
    U32 finishSquare;
    bool enPassant;
    U32 capturedPieceType;
    U32 finishPieceType;
};

struct captureCounter
{
    bool side;
    int numChecks;
    U64 pinned;
    bool enPassant;
    int victimPieceType;
    U64 victimsBB;
    int currentVictimSquare;
    int attackerPieceType;
    U64 attackerBB;
};

class Board {
    public:
        U64 pieces[12]={};

        U64 occupied[2]={0,0};
        U64 attacked[2]={0,0};

        std::vector<gameState> stateHistory;
        std::vector<U32> moveHistory;
        std::vector<U32> hashHistory;
        std::vector<int> irrevMoveInd;

        std::vector<U32> captureBuffer;
        std::vector<U32> quietBuffer;
        std::vector<U32> moveBuffer;
        std::vector<std::pair<U32,int> > scoredMoves;

        Killer killer;

        gameState current = {
            .canKingCastle = {true,true},
            .canQueenCastle = {true,true},
            .enPassantSquare = -1,
        };

        moveInfo currentMove = {};

        NNUE nnue;

        const int piecePhases[6] = {0,4,2,1,1,0};

        int phase = 24;

        //overall zHash is XOR of these two.
        U64 zHashPieces = 0;
        U64 zHashState = 0;

        //history table, history.scores[pieceType][to_square]
        History history;

        //temp variable for move appending.
        U32 newMove;

        captureCounter cc;

        Board()
        {
            //start position default.
            setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        };

        void zHashHardUpdate()
        {
            zHashPieces = 0;
            zHashState = 0;

            for (int i=0;i<12;i++)
            {
                U64 temp = pieces[i];
                while (temp)
                {
                    zHashPieces ^= randomNums[ZHASH_PIECES[i] + popLSB(temp)];
                }
            }

            if (moveHistory.size() & 1) {zHashPieces ^= randomNums[ZHASH_TURN];}

            if (current.enPassantSquare != -1) {zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];}

            if (current.canKingCastle[0]) {zHashState ^= randomNums[ZHASH_CASTLES[0]];}
            if (current.canKingCastle[1]) {zHashState ^= randomNums[ZHASH_CASTLES[1]];}
            if (current.canQueenCastle[0]){zHashState ^= randomNums[ZHASH_CASTLES[2]];}
            if (current.canQueenCastle[1]){zHashState ^= randomNums[ZHASH_CASTLES[3]];}
        }

        void phaseHardUpdate()
        {
            phase = 0;
            for (int i=0;i<12;i++)
            {
                U64 temp = pieces[i];
                while (temp)
                {
                    phase += piecePhases[i >> 1];
                    popLSB(temp);
                }
            }
        }

        void nnueHardUpdate()
        {
            nnue.refreshInput(positionToFen(pieces, current, moveHistory.size() & 1));
        }

        void setPositionFen(const std::string &fen)
        {
            //reset history.
            stateHistory.clear();
            moveHistory.clear();
            hashHistory.clear();
            irrevMoveInd.clear();

            std::vector<std::string> temp; temp.push_back("");

            for (int i=0;i<(int)fen.length();i++)
            {
                if (fen[i] == ' ') {temp.push_back("");}
                else {temp.back() += fen[i];}
            }

            //piece placement.
            for (int i=0;i<12;i++) {pieces[i] = 0;}

            U32 square = 56;
            for (int i=0;i<(int)temp[0].length();i++)
            {
                if (temp[0][i] == '/') {square -= 16;}
                else if ((int)(temp[0][i] - '0') < 9) {square += (int)(temp[0][i] - '0');}
                else if (temp[0][i] == 'K') {pieces[_nKing] += 1ull << square++;}
                else if (temp[0][i] == 'Q') {pieces[_nQueens] += 1ull << square++;}
                else if (temp[0][i] == 'R') {pieces[_nRooks] += 1ull << square++;}
                else if (temp[0][i] == 'B') {pieces[_nBishops] += 1ull << square++;}
                else if (temp[0][i] == 'N') {pieces[_nKnights] += 1ull << square++;}
                else if (temp[0][i] == 'P') {pieces[_nPawns] += 1ull << square++;}
                else if (temp[0][i] == 'k') {pieces[_nKing+1] += 1ull << square++;}
                else if (temp[0][i] == 'q') {pieces[_nQueens+1] += 1ull << square++;}
                else if (temp[0][i] == 'r') {pieces[_nRooks+1] += 1ull << square++;}
                else if (temp[0][i] == 'b') {pieces[_nBishops+1] += 1ull << square++;}
                else if (temp[0][i] == 'n') {pieces[_nKnights+1] += 1ull << square++;}
                else if (temp[0][i] == 'p') {pieces[_nPawns+1] += 1ull << square++;}
            }

            updateOccupied();

            //side to move.
            moveHistory.push_back(0);
            if (temp[1] == "w") {moveHistory.push_back(0);}

            current = {
                .canKingCastle = {false,false},
                .canQueenCastle = {false,false},
                .enPassantSquare = -1,
            };

            //castling rights.
            for (int i=0;i<(int)temp[2].length();i++)
            {
                if (temp[2][i] == 'K') {current.canKingCastle[0] = true;}
                else if (temp[2][i] == 'k') {current.canKingCastle[1] = true;}
                else if (temp[2][i] == 'Q') {current.canQueenCastle[0] = true;}
                else if (temp[2][i] == 'q') {current.canQueenCastle[1] = true;}
            }

            //en passant square.
            if (temp[3] != "-") {current.enPassantSquare = toSquare(temp[3]);}

            zHashHardUpdate();
            phaseHardUpdate();
            nnue.refreshInput(fen);

            //hash and state history.
            stateHistory.push_back(current);
            hashHistory.push_back(zHashPieces ^ zHashState);
            if (temp[1] == "w")
            {
                stateHistory.push_back(current);
                hashHistory.push_back(zHashPieces ^ zHashState);
            }
        }

        bool isValidPawnMove(bool inCheck)
        {
            //called by isValidMove, so currentMove is up-to-date.
            
            if (currentMove.enPassant)
            {
                //enPassant capture.

                //check enPassant square.
                if (current.enPassantSquare != (int)(currentMove.finishSquare)) {return false;}

                //no need to check if captured piece present, guaranteed with ep square.
            }
            else
            {
                //regular move, capture or push.

                //check if finishSquare is empty or capturedPiece.
                if ((currentMove.capturedPieceType == 15 &&
                    (bool)((occupied[0] | occupied[1]) & (1ull << currentMove.finishSquare))) ||
                    (currentMove.capturedPieceType != 15 &&
                    !(bool)(pieces[currentMove.capturedPieceType] & (1ull << currentMove.finishSquare))))
                {
                    return false;
                }

                //if double push, check that middle square is clear.
                if (abs((int)(currentMove.finishSquare) - (int)(currentMove.startSquare)) == 16)
                {
                    if (currentMove.finishSquare > currentMove.startSquare)
                    {
                        //white double push.
                        if ((occupied[0] | occupied[1]) & (1ull << (currentMove.startSquare + 8))) {return false;}
                    }
                    else
                    {
                        //black double push.
                        if ((occupied[0] | occupied[1]) & (1ull << (currentMove.startSquare - 8))) {return false;}
                    }
                }
            }

            //check if piece is pinned or if enPassant.
            U64 pinned = getPinnedPieces(currentMove.pieceType & 1);
            if ((pinned & (1ull << currentMove.startSquare)) ||
                currentMove.enPassant ||
                inCheck)
            {
                U64 start = 1ull << currentMove.startSquare;
                U64 finish = 1ull << currentMove.finishSquare;
                bool side = currentMove.pieceType & 1;
                pieces[currentMove.pieceType] -= start;
                pieces[currentMove.pieceType] += finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                if (currentMove.capturedPieceType != 15)
                {
                    pieces[currentMove.capturedPieceType] -= 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(side)));
                    occupied[(int)(!side)] -= 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(side)));
                }
                bool isBad = isInCheck(side);

                //unmove pieces.
                pieces[currentMove.pieceType] += start;
                pieces[currentMove.pieceType] -= finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;
                if (currentMove.capturedPieceType != 15)
                {
                    pieces[currentMove.capturedPieceType] += 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(side)));
                    occupied[(int)(!side)] += 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(side)));
                }
                if (isBad) {return false;}
            }
            
            return true;
        }

        bool isValidCastles(bool inCheck)
        {
            //called by isValidMove, so currentMove is up-to-date.
            if (inCheck) {return false;}
            bool side = currentMove.pieceType & 1;

            //check that castling is allowed.
            if (currentMove.finishSquare - currentMove.startSquare == 2)
            {
                //kingside.
                if (!current.canKingCastle[(int)(side)]) {return false;}
                
                //castling squares not occupied or attacked.
                updateAttacked(!side);
                U64 p = occupied[0] | occupied[1];
                if ((KING_CASTLE_OCCUPIED[(int)(side)] & p) ||
                    (KING_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)])) {return false;}
            }
            else
            {
                //queenside.
                if (!current.canQueenCastle[(int)(side)]) {return false;}

                //castling squares not occupied or attacked.
                updateAttacked(!side);
                U64 p = occupied[0] | occupied[1];
                if ((QUEEN_CASTLE_OCCUPIED[(int)(side)] & p) ||
                    (QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)])) {return false;}
            }

            return true;
        }

        bool isValidMove(U32 chessMove, bool inCheck)
        {
            //verifies if a move is valid in this position.
            //move is assumed to be legal from some other arbitrary position in the search tree.
            unpackMove(chessMove);

            //check for correct side-to-move.
            if ((currentMove.pieceType & 1) != (moveHistory.size() & 1)) {return false;}

            //check that startSquare contains piece.
            if (!(bool)(pieces[currentMove.pieceType] & (1ull << currentMove.startSquare))) {return false;}

            //check for pawn move.
            if ((currentMove.pieceType >> 1) == (_nPawns >> 1)) {return isValidPawnMove(inCheck);}
            //check for castles.
            if (((currentMove.pieceType >> 1) == (_nKing >> 1)) &&
                (abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare)) == 2))
            {
                return isValidCastles(inCheck);
            }

            //ordinary non-pawn capture/quiet.

            //check that finishSquare is empty or contains capturedPiece.
            if ((currentMove.capturedPieceType == 15 &&
                (bool)((occupied[0] | occupied[1]) & (1ull << currentMove.finishSquare))) ||
                (currentMove.capturedPieceType != 15 &&
                !(bool)(pieces[currentMove.capturedPieceType] & (1ull << currentMove.finishSquare))))
            {
                return false;
            }

            //startSquare -> finishSquare is valid path for that piece.
            //knight moves are path legal.
            switch(currentMove.pieceType >> 1)
            {
                case _nKing >> 1:
                    if (!(kingAttacks(pieces[currentMove.pieceType]) & ~kingAttacks(pieces[_nKing + !(bool)(currentMove.pieceType & 1)]) & (1ull << currentMove.finishSquare))) {return false;}
                    break;
                case _nQueens >> 1:
                    if (!(magicQueenAttacks(occupied[0] | occupied[1], currentMove.startSquare) & (1ull << currentMove.finishSquare))) {return false;}
                    break;
                case _nRooks >> 1:
                    if (!(magicRookAttacks(occupied[0] | occupied[1], currentMove.startSquare) & (1ull << currentMove.finishSquare))) {return false;}
                    break;
                case _nBishops >> 1:
                    if (!(magicBishopAttacks(occupied[0] | occupied[1], currentMove.startSquare) & (1ull << currentMove.finishSquare))) {return false;}
                    break;
            }

            //check if the piece to move is pinned and verify the move if necessary.
            U64 pinned = getPinnedPieces(currentMove.pieceType & 1);
            if ((pinned & (1ull << currentMove.startSquare)) ||
                ((currentMove.pieceType >> 1) == (_nKing >> 1)) ||
                inCheck)
            {
                U64 start = 1ull << currentMove.startSquare;
                U64 finish = 1ull << currentMove.finishSquare;
                bool side = currentMove.pieceType & 1;
                pieces[currentMove.pieceType] -= start;
                pieces[currentMove.pieceType] += finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                if (currentMove.capturedPieceType != 15)
                {
                    pieces[currentMove.capturedPieceType] -= finish;
                    occupied[(int)(!side)] -= finish;
                }
                bool isBad = isInCheck(side);

                //unmove pieces.
                pieces[currentMove.pieceType] += start;
                pieces[currentMove.pieceType] -= finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;
                if (currentMove.capturedPieceType != 15)
                {
                    pieces[currentMove.capturedPieceType] += finish;
                    occupied[(int)(!side)] += finish;
                }
                if (isBad) {return false;}
            }

            //all checks passed!
            return true;
        }

        bool isCheckingMove(U32 chessMove)
        {
            //verifies if a legal move gives check.
            U32 pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            bool side = pieceType & 1;

            switch(pieceType >> 1)
            {
                case _nQueens >> 1:
                    if (magicRookAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing+(int)(!side)]) {return true;}
                    if (magicBishopAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing+(int)(!side)]) {return true;}
                    break;
                case _nRooks >> 1:
                    if (magicRookAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing+(int)(!side)]) {return true;}
                    break;
                case _nBishops >> 1:
                    if (magicBishopAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing+(int)(!side)]) {return true;}
                    break;
                case _nKnights >> 1:
                    if (knightAttacks(1ull << finishSquare) & pieces[_nKing+(int)(!side)]) {return true;}
                    break;
                case _nPawns >> 1:
                    if (pawnAttacks(1ull << finishSquare, side) & pieces[_nKing+(int)(!side)]) {return true;}
                    break;
            }

            U32 startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
            int kingPos = __builtin_ctzll(pieces[_nKing+(int)(!side)]);

            //regular discovered check (rook/bishop rays).
            if ((magicRookAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
                (magicRookAttacks((occupied[0] | occupied[1]) ^ (1ull << startSquare), kingPos) & (pieces[_nRooks+(int)(side)] | pieces[_nQueens+(int)(side)])))
            {
                return true;
            }
            if ((magicBishopAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
                (magicBishopAttacks((occupied[0] | occupied[1]) ^ (1ull << startSquare), kingPos) & (pieces[_nBishops+(int)(side)] | pieces[_nQueens+(int)(side)])))
            {
                return true;
            }

            //enpassant discovered check.
            if (chessMove & MOVEINFO_ENPASSANT_MASK)
            {
                U32 enPassantSquare = finishSquare - 8 + 16*side;
                U64 after = (occupied[0] | occupied[1]) ^ (1ull << startSquare) ^ (1ull << finishSquare) ^ (1ull << enPassantSquare);
                if ((magicRookAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
                    (magicRookAttacks(after, kingPos) & (pieces[_nRooks+(int)(side)] | pieces[_nQueens+(int)(side)])))
                {
                    return true;
                }
                if ((magicBishopAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
                    (magicBishopAttacks(after, kingPos) & (pieces[_nBishops+(int)(side)] | pieces[_nQueens+(int)(side)])))
                {
                    return true;
                }
            }

            //castles discovered check.
            if (pieceType >> 1 == _nKing >> 1 && abs((int)(finishSquare) - (int)(startSquare)) == 2)
            {
                if (finishSquare > startSquare)
                {
                    if (magicRookAttacks((occupied[0] | occupied[1]) ^ startSquare, KING_ROOK_SQUARE[side]-2) & pieces[_nKing+(int)(!side)]) {return true;}
                }
                else
                {
                    if (magicRookAttacks((occupied[0] | occupied[1]) ^ startSquare, QUEEN_ROOK_SQUARE[side]+3) & pieces[_nKing+(int)(!side)]) {return true;}
                }
            }

            return false;
        }

        void appendPawnCapture(U32 pieceType, U32 startSquare, U32 finishSquare, bool enPassant, bool shouldCheck)
        {
            //pawn captures, promotion and enPassant.
            bool side = pieceType & 1;
            bool promotion = false;
            U32 capturedPieceType = 15;

            if (enPassant)
            {
                //enPassant.
                capturedPieceType = _nPawns + (U32)(!side);
            }
            else if ((finishSquare >> 3) == (U32)(7-7*(side)))
            {
                //promotion.
                promotion = true;
                if (((1ull << finishSquare) & occupied[(int)(!side)]) != 0)
                {
                    //check for captures on promotion.
                    U64 x = 1ull << finishSquare;
                    for (U32 i=_nQueens+(!side);i<12;i+=2)
                    {
                        if ((x & pieces[i]) != 0) {capturedPieceType = i; break;}
                    }
                }
            }
            else
            {
                //regular pawn capture.
                U64 x = 1ull << finishSquare;
                for (U32 i=_nQueens+(!side);i<12;i+=2)
                {
                    if ((x & pieces[i]) != 0) {capturedPieceType = i; break;}
                }
            }

            if (shouldCheck)
            {
                //check if move is legal (does not leave king in check).
                //move pieces.
                U64 start = 1ull << startSquare;
                U64 finish = 1ull << finishSquare;
                pieces[pieceType] -= start;
                pieces[pieceType] += finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                if (capturedPieceType != 15)
                {
                    pieces[capturedPieceType] -= 1ull << (finishSquare+(int)(enPassant)*(-8+16*(side)));
                    occupied[(int)(!side)] -= 1ull << (finishSquare+(int)(enPassant)*(-8+16*(side)));
                }
                bool isBad = isInCheck(side);

                //unmove pieces.
                pieces[pieceType] += start;
                pieces[pieceType] -= finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;
                if (capturedPieceType != 15)
                {
                    pieces[capturedPieceType] += 1ull << (finishSquare+(int)(enPassant)*(-8+16*(side)));
                    occupied[(int)(!side)] += 1ull << (finishSquare+(int)(enPassant)*(-8+16*(side)));
                }
                if (isBad) {return;}
            }

            newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (enPassant << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

            if (promotion)
            {
                //promotion.
                for (U32 i=_nQueens+(side);i<_nPawns;i+=2)
                {
                    newMove &= ~MOVEINFO_FINISHPIECETYPE_MASK;
                    newMove |= i << MOVEINFO_FINISHPIECETYPE_OFFSET;
                    moveBuffer.push_back(newMove);
                }
            }
            else
            {
                //append normally.
                moveBuffer.push_back(newMove);
            }
        }

        void appendCapture(U32 pieceType, U32 startSquare, U32 finishSquare, bool shouldCheck)
        {
            bool side = pieceType & 1;
            U32 capturedPieceType = 15;
            //regular capture, loop through to find victim.
            U64 x = 1ull << finishSquare;
            for (U32 i=_nQueens+(!side);i<12;i+=2)
            {
                if ((x & pieces[i]) != 0) {capturedPieceType = i; break;}
            }

            if (shouldCheck)
            {
                //check if move is legal (does not leave king in check).
                //move pieces.
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

                if (isBad) {return;}
            }

            newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (false << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

            moveBuffer.push_back(newMove);
        }

        void appendQuiet(U32 pieceType, U32 startSquare, U32 finishSquare, bool shouldCheck)
        {
            //never check castles.
            if (shouldCheck)
            {
                //check if move is legal (does not leave king in check).
                //move pieces.
                bool side = pieceType & 1;
                U64 start = 1ull << startSquare;
                U64 finish = 1ull << finishSquare;
                pieces[pieceType] -= start;
                pieces[pieceType] += finish;
                occupied[(int)(side)] -= start;
                occupied[(int)(side)] += finish;
                bool isBad = isInCheck(pieceType & 1);

                //unmove pieces.
                pieces[pieceType] += start;
                pieces[pieceType] -= finish;
                occupied[(int)(side)] += start;
                occupied[(int)(side)] -= finish;

                if (isBad) {return;}
            }

            newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (false << MOVEINFO_ENPASSANT_OFFSET) |
            (15u << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

            moveBuffer.push_back(newMove);
        }

        bool checkStalemateMove(U32 pieceType, U32 startSquare, U32 finishSquare)
        {
            int capturedPieceType = 15; bool enPassant = false;

            //check for captured pieces (including en passant).
            if (((1ull << finishSquare) & occupied[(pieceType+1) & 1])!=0)
            {
                //check for captures.
                U64 x = 1ull << finishSquare;
                for (U32 i=_nQueens+((pieceType+1) & 1);i<12;i+=2)
                {
                    if ((x & pieces[i]) != 0)
                    {
                        capturedPieceType = i;
                        break;
                    }
                }
            }
            else if ((pieceType >> 1 == _nPawns >> 1) && (finishSquare >> 3 == 5u-3u*(pieceType & 1u)) && ((finishSquare & 7) != (startSquare & 7)))
            {
                //en-passant capture.
                enPassant = true;

                //check for captured piece.
                capturedPieceType = (_nPawns+((pieceType+1u) & 1u));
            }

            //check if move is legal (does not leave king in check).
            //move pieces.
            U64 start = 1ull << startSquare;
            U64 finish = 1ull << finishSquare;
            pieces[pieceType] -= start;
            pieces[pieceType] += finish;
            occupied[pieceType & 1] -= start;
            occupied[pieceType & 1] += finish;
            if (capturedPieceType != 15)
            {
                U64 captured = 1ull << (finishSquare+(int)(enPassant)*(-8+16*(pieceType & 1)));
                pieces[capturedPieceType] -= captured;
                occupied[capturedPieceType & 1] -= captured;
            }
            bool isBad = isInCheck(pieceType & 1);

            //unmove pieces.
            pieces[pieceType] += start;
            pieces[pieceType] -= finish;
            occupied[pieceType & 1] += start;
            occupied[pieceType & 1] -= finish;
            if (capturedPieceType != 15)
            {
                U64 captured = 1ull << (finishSquare+(int)(enPassant)*(-8+16*(pieceType & 1)));
                pieces[capturedPieceType] += captured;
                occupied[capturedPieceType & 1] += captured;
            }

            if (isBad) {return false;}

            return true;
        }

        void updateOccupied()
        {
            occupied[0] = pieces[_nKing] | pieces[_nQueens] | pieces[_nRooks] | pieces[_nBishops] | pieces[_nKnights] | pieces[_nPawns];
            occupied[1] = pieces[_nKing+1] | pieces[_nQueens+1] | pieces[_nRooks+1] | pieces[_nBishops+1] | pieces[_nKnights+1] | pieces[_nPawns+1];
        }

        void updateAttacked(bool side)
        {
            //king.
            attacked[(int)(side)] = kingAttacks(pieces[_nKing+(int)(side)]);

            //queen.
            U64 b = occupied[0] | occupied[1];
            U64 temp = pieces[_nQueens+(int)(side)];
            while (temp)
            {
                attacked[(int)(side)] |= magicQueenAttacks(b,popLSB(temp));
            }

            //rooks.
            temp = pieces[_nRooks+(int)(side)];
            while (temp)
            {
                attacked[(int)(side)] |= magicRookAttacks(b,popLSB(temp));
            }

            //bishops.
            temp = pieces[_nBishops+(int)(side)];
            while (temp)
            {
                attacked[(int)(side)] |= magicBishopAttacks(b,popLSB(temp));
            }

            //knights.
            attacked[(int)(side)] |= knightAttacks(pieces[_nKnights+(int)(side)]);

            //pawns.
            attacked[(int)(side)] |= pawnAttacks(pieces[_nPawns+(int)(side)],side);
        }

        bool isInCheck(bool side)
        {
            //check if the king's square is attacked.
            int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
            U64 b = occupied[0] | occupied[1];

            return
                (bool)(magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)])) ||
                (bool)(magicBishopAttacks(b,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)])) ||
                (bool)(knightAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKnights+(int)(!side)]) ||
                (bool)(pawnAttacks(pieces[_nKing+(int)(side)],side) & pieces[_nPawns+(int)(!side)]);
        }

        U32 isInCheckDetailed(bool side)
        {
            //check if the king's square is attacked.
            int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
            U64 b = occupied[0] | occupied[1];

            U64 inCheck = magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
            inCheck |= magicBishopAttacks(b,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
            inCheck |= knightAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKnights+(int)(!side)];
            inCheck |= pawnAttacks(pieces[_nKing+(int)(side)],side) & pieces[_nPawns+(int)(!side)];

            return __builtin_popcountll(inCheck);
        }

        U64 getCheckPiece(bool side, U32 square)
        {
            //assumes a single piece is giving check.
            U64 b = occupied[0] | occupied[1];

            if (U64 bishop = magicBishopAttacks(b,square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)])) {return bishop;}
            else if (U64 rook = magicRookAttacks(b,square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)])) {return rook;}
            else if (U64 knight = knightAttacks(1ull << square) & pieces[_nKnights+(int)(!side)]) {return knight;}
            else {return pawnAttacks(1ull << square,side) & pieces[_nPawns+(int)(!side)];}
        }

        U64 getBlockSquares(bool side, U32 square)
        {
            //assumes a single piece is giving check.
            U64 b = occupied[0] | occupied[1];

            if (U64 bishop = magicBishopAttacks(b, square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
            {
                return magicBishopAttacks(b, square) & magicBishopAttacks(b, __builtin_ctzll(bishop));
            }
            if (U64 rook = magicRookAttacks(b, square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
            {
                return magicRookAttacks(b, square) & magicRookAttacks(b, __builtin_ctzll(rook));
            }
            return 0;
        }

        U64 getPinnedPieces(bool side)
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

        void display()
        {
            //display the current position in console.

            const char symbols[12]={'K','k','Q','q','R','r','B','b','N','n','P','p'};

            std::vector<std::vector<std::string> > grid(
                8,
                std::vector<std::string>(8,"[ ]")
            );

            U64 x;
            for (int i=0;i<12;i++)
            {
                x = pieces[i];
                for (int j=0;j<64;j++)
                {
                    if (x & 1) {grid[j/8][j%8][1]=symbols[i];}
                    x = x >> 1;
                }
            }

            for (int i=7;i>=0;i--)
            {
                for (int j=0;j<8;j++) {std::cout << grid[i][j];}
                std::cout << " " << i+1 << std::endl;
            }
            std::cout << " A  B  C  D  E  F  G  H" << std::endl;
        }

        void resetCaptureCounter(int side, int numChecks)
        {
            cc.side = side;
            cc.numChecks = numChecks;
            cc.pinned = getPinnedPieces(side);
            cc.enPassant = false;
            cc.victimPieceType = _nKing + (int)(!side);
            cc.victimsBB = 0;
            cc.attackerPieceType = _nKing + (int)(side);
            cc.attackerBB = 0;
            updateCaptureCounter();
        }

        void updateCaptureCounter()
        {
            //check if we update the victim.
            if (cc.attackerPieceType >> 1 == cc.victimPieceType >> 1)
            {
                //check if we update victim piece type.
                if (!cc.victimsBB)
                {
                    //check if all captures exhausted.
                    if (cc.victimPieceType == _nPawns + (int)(!cc.side))
                    {
                        if (cc.enPassant) {cc.attackerBB = 0; return;}

                        cc.enPassant = true;
                        //test for enpassant.
                        if (current.enPassantSquare != -1)
                        {
                            cc.currentVictimSquare = current.enPassantSquare;
                            cc.attackerBB = pawnAttacks(1ull << current.enPassantSquare, !cc.side) & pieces[cc.attackerPieceType];
                        }
                        return;
                    }

                    //update victim piece type.
                    while (!cc.victimsBB && cc.victimPieceType < _nPawns + 2)
                    {
                        cc.victimPieceType += 2;
                        cc.victimsBB = pieces[cc.victimPieceType];
                    }
                    if (!cc.victimsBB) {return;}
                }

                //update victim.
                cc.currentVictimSquare = popLSB(cc.victimsBB);
                cc.attackerPieceType = _nPawns + (int)(cc.side) + 2;
            }

            //update attacker.
            cc.attackerPieceType -= 2;

            switch(cc.attackerPieceType >> 1)
            {
                case _nPawns:
                    cc.attackerBB = pawnAttacks(1ull << cc.currentVictimSquare, !cc.side);
                    break;
                case _nKnights:
                    cc.attackerBB = knightAttacks(1ull << cc.currentVictimSquare);
                    break;
                case _nBishops:
                    cc.attackerBB = magicBishopAttacks(occupied[0] | occupied[1], cc.currentVictimSquare);
                    break;
                case _nRooks:
                    cc.attackerBB = magicRookAttacks(occupied[0] | occupied[1], cc.currentVictimSquare);
                    break;
                case _nQueens:
                    cc.attackerBB = magicQueenAttacks(occupied[0] | occupied[1], cc.currentVictimSquare);
                    break;
                case _nKing:
                    cc.attackerBB = kingAttacks(1ull << cc.currentVictimSquare) & ~kingAttacks(pieces[_nKing + (int)(!cc.side)]);
                    break;
            }

            //intersection with attackers.
            cc.attackerBB &= pieces[cc.attackerPieceType];

            //if no valid attacks, iterate again.
            if (!cc.attackerBB) {updateCaptureCounter();}
        }

        U32 generateNextGoodCapture()
        {
            if (!cc.attackerBB)
            {
                updateCaptureCounter();
                if (!cc.attackerBB) {return 0;}
            }

            int finishSquare = cc.currentVictimSquare;
            int startSquare = popLSB(cc.attackerBB);
            int pieceType = cc.attackerPieceType;
            int finishPieceType = cc.attackerPieceType;
            //check for promotion. only consider queen promotion for qsearch.
            if ((pieceType == _nPawns + (int)(cc.side)) &&
                ((cc.side && ((1ull << startSquare) & FILE_2)) ||
                 (!cc.side && ((1ull << startSquare) & FILE_7))))
            {
                finishPieceType = _nQueens + (int)(cc.side);
            }
            int capturedPieceType = cc.victimPieceType;
            bool enPassant = cc.enPassant;

            //check if the piece is pinned.
            if ((cc.pinned & (1ull << startSquare)) || enPassant || pieceType >> 1 == _nKing >> 1)
            {
                U64 start = 1ull << startSquare;
                U64 finish = 1ull << finishSquare;
                pieces[pieceType] -= start;
                pieces[pieceType] += finish;
                pieces[capturedPieceType] -= finish;
                occupied[(int)(cc.side)] -= start;
                occupied[(int)(cc.side)] += finish;
                occupied[(int)(!cc.side)] -= finish;
                bool isBad = isInCheck(cc.side);

                //unmove pieces.
                pieces[pieceType] += start;
                pieces[pieceType] -= finish;
                pieces[capturedPieceType] += finish;
                occupied[(int)(cc.side)] += start;
                occupied[(int)(cc.side)] -= finish;
                occupied[(int)(!cc.side)] += finish;

                if (isBad) {return generateNextGoodCapture();}
            }

            //return the move.
            return (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (enPassant << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);
        }

        void generateCaptures(bool side, int numChecks = 0)
        {
            if (numChecks == 0)
            {
                //regular captures.
                U32 pos; U64 x; U64 temp;
                U64 pinned = getPinnedPieces(side);
                U64 p = (occupied[0] | occupied[1]);

                //pawns.
                temp = pieces[_nPawns+(int)(side)];
                U64 pawnPosBoard;
                while (temp)
                {
                    pos = popLSB(temp);
                    pawnPosBoard = 1ull << pos;
                    x = pawnAttacks(pawnPosBoard,side) & occupied[(int)(!side)];

                    //promotion by moving forward.
                    if (!side) {x |= ((pawnPosBoard & RANK_7) << 8) & (~p);}
                    else {x |= ((pawnPosBoard & RANK_2) >> 8) & (~p);}

                    while (x) {appendPawnCapture(_nPawns+(int)(side), pos, popLSB(x), false, (pawnPosBoard & pinned)!=0);}
                }

                //enPassant.
                if (current.enPassantSquare != -1)
                {
                    temp = pawnAttacks(1ull << current.enPassantSquare, !side) & pieces[_nPawns+(int)(side)];
                    while (temp)
                    {
                        pos = popLSB(temp);
                        appendPawnCapture(_nPawns+(int)(side), pos, current.enPassantSquare, true, true);
                    }
                }

                //knights.
                temp = pieces[_nKnights+(int)(side)] & ~pinned;
                while (temp)
                {
                    pos = popLSB(temp);
                    x = knightAttacks(1ull << pos) & occupied[(int)(!side)];
                    while (x) {appendCapture(_nKnights+(int)(side), pos, popLSB(x), false);}
                }

                //bishops.
                temp = pieces[_nBishops+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicBishopAttacks(p,pos) & occupied[(int)(!side)];
                    while (x) {appendCapture(_nBishops+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //rook.
                temp = pieces[_nRooks+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicRookAttacks(p,pos) & occupied[(int)(!side)];
                    while (x) {appendCapture(_nRooks+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //queen.
                temp = pieces[_nQueens+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicQueenAttacks(p,pos) & occupied[(int)(!side)];
                    while (x) {appendCapture(_nQueens+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //king.
                pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & occupied[(int)(!side)];
                while (x) {appendCapture(_nKing+(int)(side), pos, popLSB(x), true);}
            }
            else if (numChecks == 1)
            {
                //single check.
                U32 pos; U64 x; U64 temp;
                U64 p = (occupied[0] | occupied[1]);

                pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 target = getCheckPiece(side, pos);

                //king.
                x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & occupied[(int)(!side)];
                while (x) {appendCapture(_nKing+(int)(side), pos, popLSB(x), true);}

                //pawns.
                temp = pieces[_nPawns+(int)(side)];
                U64 pawnPosBoard;
                while (temp)
                {
                    pos = popLSB(temp);
                    pawnPosBoard = 1ull << pos;
                    x = pawnAttacks(pawnPosBoard,side) & target;

                    //promotion by moving forward.
                    if (!side) {x |= ((pawnPosBoard & RANK_7) << 8) & (~p);}
                    else {x |= ((pawnPosBoard & RANK_2) >> 8) & (~p);}

                    while (x) {appendPawnCapture(_nPawns+(int)(side), pos, popLSB(x), false, true);}
                }

                //enPassant.
                if (current.enPassantSquare != -1)
                {
                    temp = pawnAttacks(1ull << current.enPassantSquare, !side) & pieces[_nPawns+(int)(side)];
                    while (temp)
                    {
                        pos = popLSB(temp);
                        appendPawnCapture(_nPawns+(int)(side), pos, current.enPassantSquare, true, true);
                    }
                }

                //knights.
                temp = pieces[_nKnights+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = knightAttacks(1ull << pos) & target;
                    while (x) {appendCapture(_nKnights+(int)(side), pos, popLSB(x), true);}
                }

                //bishops.
                temp = pieces[_nBishops+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicBishopAttacks(p,pos) & target;
                    while (x) {appendCapture(_nBishops+(int)(side), pos, popLSB(x), true);}
                }

                //rook.
                temp = pieces[_nRooks+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicRookAttacks(p,pos) & target;
                    while (x) {appendCapture(_nRooks+(int)(side), pos, popLSB(x), true);}
                }

                //queen.
                temp = pieces[_nQueens+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicQueenAttacks(p,pos) & target;
                    while (x) {appendCapture(_nQueens+(int)(side), pos, popLSB(x), true);}
                }
            }
            else
            {
                //multiple check. only king moves allowed.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & occupied[(int)(!side)];
                while (x) {appendCapture(_nKing+(int)(side), pos, popLSB(x), true);}
            }
        }

        void generateQuiets(bool side, int numChecks = 0)
        {
            U64 p = (occupied[0] | occupied[1]);
            if (numChecks == 0)
            {
                //regular moves.
                U32 pos; U64 x; U64 temp;

                //castling.

                if (current.canKingCastle[(int)(side)] || current.canQueenCastle[(int)(side)])
                {
                    updateAttacked(!side);
                    if (current.canKingCastle[(int)(side)] &&
                        !(bool)(KING_CASTLE_OCCUPIED[(int)(side)] & p) &&
                        !(bool)(KING_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                    {
                        //kingside castle.
                        pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                        appendQuiet(_nKing+(int)(side), pos, pos+2, false);
                    }
                    if (current.canQueenCastle[(int)(side)] &&
                        !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & p) &&
                        !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                    {
                        //queenside castle.
                        pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                        appendQuiet(_nKing+(int)(side), pos, pos-2, false);
                    }
                }

                U64 pinned = getPinnedPieces(side);

                //knights.
                temp = pieces[_nKnights+(int)(side)] & ~pinned;
                while (temp)
                {
                    pos = popLSB(temp);
                    x = knightAttacks(1ull << pos) & ~p;
                    while (x) {appendQuiet(_nKnights+(int)(side), pos, popLSB(x), false);}
                }

                //bishops.
                temp = pieces[_nBishops+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicBishopAttacks(p,pos) & ~p;
                    while (x) {appendQuiet(_nBishops+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //pawns.
                temp = pieces[_nPawns+(int)(side)];
                U64 pawnPosBoard;
                while (temp)
                {
                    pos = popLSB(temp);
                    pawnPosBoard = 1ull << pos;
                    x = 0;

                    //move forward (exclude promotion).
                    if (side==0)
                    {
                        x |= ((pawnPosBoard & (~RANK_7)) << 8) & (~p);
                        x |= ((((pawnPosBoard & RANK_2) << 8) & (~p)) << 8) & (~p);
                    }
                    else
                    {
                        x |= ((pawnPosBoard & (~RANK_2)) >> 8 & (~p));
                        x |= ((((pawnPosBoard & RANK_7) >> 8) & (~p)) >> 8) & (~p);
                    }

                    while (x) {appendQuiet(_nPawns+(int)(side), pos, popLSB(x), (pawnPosBoard & pinned)!=0);}
                }

                //rook.
                temp = pieces[_nRooks+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicRookAttacks(p,pos) & ~p;
                    while (x) {appendQuiet(_nRooks+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //queen.
                temp = pieces[_nQueens+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicQueenAttacks(p,pos) & ~p;
                    while (x) {appendQuiet(_nQueens+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //king.
                pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & ~p;
                while (x) {appendQuiet(_nKing+(int)(side), pos, popLSB(x), true);}
            }
            else if (numChecks == 1)
            {
                //single check.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & ~p;
                while (x) {appendQuiet(_nKing+(int)(side), pos, popLSB(x), true);}

                U64 blockBB = getBlockSquares(side, pos);
                if (!blockBB) {return;}

                U64 temp;

                //pawns.
                temp = pieces[_nPawns+(int)(side)];
                U64 pawnPosBoard;
                while (temp)
                {
                    pos = popLSB(temp);
                    pawnPosBoard = 1ull << pos;
                    x = 0;

                    //move forward (exclude promotion).
                    if (side==0)
                    {
                        x |= ((pawnPosBoard & (~RANK_7)) << 8) & blockBB;
                        x |= ((((pawnPosBoard & RANK_2) << 8) & (~p)) << 8) & blockBB;
                    }
                    else
                    {
                        x |= ((pawnPosBoard & (~RANK_2)) >> 8) & blockBB;
                        x |= ((((pawnPosBoard & RANK_7) >> 8) & (~p)) >> 8) & blockBB;
                    }

                    while (x) {appendQuiet(_nPawns+(int)(side), pos, popLSB(x), true);}
                }

                //bishops.
                temp = pieces[_nBishops+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicBishopAttacks(p,pos) & blockBB;
                    while (x) {appendQuiet(_nBishops+(int)(side), pos, popLSB(x), true);}
                }

                //knights.
                temp = pieces[_nKnights+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = knightAttacks(1ull << pos) & blockBB;
                    while (x) {appendQuiet(_nKnights+(int)(side), pos, popLSB(x), true);}
                }

                //rook.
                temp = pieces[_nRooks+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicRookAttacks(p,pos) & blockBB;
                    while (x) {appendQuiet(_nRooks+(int)(side), pos, popLSB(x), true);}
                }

                //queen.
                temp = pieces[_nQueens+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicQueenAttacks(p,pos) & blockBB;
                    while (x) {appendQuiet(_nQueens+(int)(side), pos, popLSB(x), true);}
                }
            }
            else
            {
                //multiple check. only king moves allowed.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & ~p;
                while (x) {appendQuiet(_nKing+(int)(side), pos, popLSB(x), true);}
            }
        }

        bool generatePseudoMoves(bool side)
        {
            moveBuffer.clear();
            bool inCheck = isInCheck(side);
            U32 numChecks = 0;
            if (inCheck) {numChecks = isInCheckDetailed(side);}
            generateCaptures(side, numChecks);
            generateQuiets(side, numChecks);
            return inCheck;
        }

        bool stalemateCheck(bool side)
        {
            //we assume we are not in check.
            U64 p = (occupied[0] | occupied[1]);
            U64 pinned = getPinnedPieces(side);

            //check for ordinary moves.
            U32 pos; U64 x; U64 temp;

            //knight - not pinned.
            temp = pieces[_nKnights+(int)(side)] & ~pinned;
            x = knightAttacks(temp) & ~occupied[(int)(side)];
            if (x) {return false;}

            //bishop - not pinned.
            temp = pieces[_nBishops+(int)(side)] & ~pinned;
            while (temp)
            {
                pos = popLSB(temp);
                x = magicBishopAttacks(p, pos) & ~occupied[(int)(side)];
                if (x) {return false;}
            }

            //rook - not pinned.
            temp = pieces[_nRooks+(int)(side)] & ~pinned;
            while (temp)
            {
                pos = popLSB(temp);
                x = magicRookAttacks(p, pos) & ~occupied[(int)(side)];
                if (x) {return false;}
            }

            //queen - not pinned.
            temp = pieces[_nQueens+(int)(side)] & ~pinned;
            while (temp)
            {
                pos = popLSB(temp);
                x = magicQueenAttacks(p, pos) & ~occupied[(int)(side)];
                if (x) {return false;}
            }

            //pawn - not pinned.
            temp = pieces[_nPawns+(int)(side)] & ~pinned;
            x = 0;
            //move forward.
            if (side==0)
            {
                x |= (temp << 8) & ~p;
                x |= ((((temp & RANK_2) << 8) & (~p)) << 8) & (~p);
            }
            else
            {
                x |= (temp >> 8) & ~p;
                x |= ((((temp & RANK_7) >> 8) & (~p)) >> 8) & (~p);
            }
            if (x) {return false;}
            //capture.
            x = pawnAttacks(temp, side) & occupied[(int)(!side)];
            if (x) {return false;}

            //check for special moves (en-passant, king, pinned).

            //bishop - pinned.
            temp = pieces[_nBishops+(int)(side)] & pinned;
            while (temp)
            {
                pos = popLSB(temp);
                x = magicBishopAttacks(p, pos) & ~occupied[(int)(side)];
                while (x)
                {
                    if (checkStalemateMove(_nBishops+(int)(side), pos, popLSB(x))) {return false;}
                }
            }

            //rook - pinned.
            temp = pieces[_nRooks+(int)(side)] & pinned;
            while (temp)
            {
                pos = popLSB(temp);
                x = magicRookAttacks(p, pos) & ~occupied[(int)(side)];
                while (x)
                {
                    if (checkStalemateMove(_nRooks+(int)(side), pos, popLSB(x))) {return false;}
                }
            }

            //queen - pinned.
            temp = pieces[_nQueens+(int)(side)] & pinned;
            while (temp)
            {
                pos = popLSB(temp);
                x = magicQueenAttacks(p, pos) & ~occupied[(int)(side)];
                while (x)
                {
                    if (checkStalemateMove(_nQueens+(int)(side), pos, popLSB(x))) {return false;}
                }
            }

            //pawn - pinned.
            temp = pieces[_nPawns+(int)(side)] & pinned;
            U64 pawnPosBoard;
            while (temp)
            {
                pos = popLSB(temp);
                pawnPosBoard = 1ull << pos;
                
                x = pawnAttacks(pawnPosBoard, side) & occupied[(int)(!side)];
                //move forward.
                if (side == 0)
                {
                    x |= (pawnPosBoard << 8) & (~p);
                    x |= ((((pawnPosBoard & RANK_2) << 8) & (~p)) << 8) & (~p);
                }
                else
                {
                    x |= (pawnPosBoard >> 8 & (~p));
                    x |= ((((pawnPosBoard & RANK_7) >> 8) & (~p)) >> 8) & (~p);
                }

                while (x)
                {
                    if (checkStalemateMove(_nPawns+(int)(side), pos, popLSB(x))) {return false;}
                }
            }

            //pawn - en-passant.
            if (current.enPassantSquare != -1)
            {
                temp = pieces[_nPawns+(int)(side)];
                x = pawnAttacks(1ull << current.enPassantSquare, !side) & temp;
                while (x)
                {
                    if (checkStalemateMove(_nPawns+(int)(side), popLSB(x), current.enPassantSquare)) {return false;}
                }
            }

            //king moves.
            pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
            x = kingAttacks(pieces[_nKing+(int)(side)]) & ~kingAttacks(pieces[_nKing+(int)(!side)]) & ~occupied[(int)(side)];
            while (x)
            {
                if (checkStalemateMove(_nKing+(int)(side), pos, popLSB(x))) {return false;}
            }

            //castling.
            if (current.canKingCastle[(int)(side)] || current.canQueenCastle[(int)(side)])
            {
                updateAttacked(!side);
                if (current.canKingCastle[(int)(side)] &&
                    !(bool)(KING_CASTLE_OCCUPIED[(int)(side)] & p) &&
                    !(bool)(KING_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                {
                    //kingside castle.
                    return false;
                }
                if (current.canQueenCastle[(int)(side)] &&
                    !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & p) &&
                    !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                {
                    //queenside castle.
                    return false;
                }
            }

            //no legal moves found - stalemate.
            return true;
        }

        void movePieces()
        {
            //remove piece from start square;
            pieces[currentMove.pieceType] -= 1ull << (currentMove.startSquare);
            zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];

            //add piece to end square, accounting for promotion.
            pieces[currentMove.finishPieceType] += 1ull << (currentMove.finishSquare);
            zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];

            //update nnue.
            nnue.zeroInput(64 * currentMove.pieceType + currentMove.startSquare);
            nnue.oneInput(64 * currentMove.finishPieceType + currentMove.finishSquare);

            //update phase on promotion.
            if (currentMove.pieceType != currentMove.finishPieceType)
            {
                phase += piecePhases[currentMove.finishPieceType >> 1];
            }

            //remove any captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                int capturedSquare = currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1));
                pieces[currentMove.capturedPieceType] -= 1ull << capturedSquare;
                zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + capturedSquare];

                //update the game phase.
                phase -= piecePhases[currentMove.capturedPieceType >> 1];

                //update nnue.
                nnue.zeroInput(64 * currentMove.capturedPieceType + capturedSquare);
            }

            //if castles, then move the rook too.
            if (currentMove.pieceType >> 1 == _nKing >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare))==2)
            {
                if (currentMove.finishSquare-currentMove.startSquare==2)
                {
                    //kingside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= KING_ROOK_POS[currentMove.pieceType & 1];
                    pieces[_nRooks+(currentMove.pieceType & 1)] += KING_ROOK_POS[currentMove.pieceType & 1] >> 2;

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2];

                    //update nnue.
                    nnue.zeroInput(64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1]);
                    nnue.oneInput(64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2);
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1];
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];

                    //update nnue.
                    nnue.zeroInput(64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]);
                    nnue.oneInput(64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3);
                }
            }

            updateOccupied();
        }

        void unMovePieces()
        {
            //remove piece from destination square.
            pieces[currentMove.finishPieceType] -= 1ull << (currentMove.finishSquare);
            zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];
            
            //add piece to start square.
            pieces[currentMove.pieceType] += 1ull << (currentMove.startSquare);
            zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];

            //update nnue.
            nnue.oneInput(64 * currentMove.pieceType + currentMove.startSquare);
            nnue.zeroInput(64 * currentMove.finishPieceType + currentMove.finishSquare);

            //update phase on promotion.
            if (currentMove.pieceType != currentMove.finishPieceType)
            {
                phase -= piecePhases[currentMove.finishPieceType >> 1];
            }

            //add back captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                int capturedSquare = currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1));
                pieces[currentMove.capturedPieceType] += 1ull << capturedSquare;
                zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + capturedSquare];

                //update the game phase.
                phase += piecePhases[currentMove.capturedPieceType >> 1];

                //update nnue.
                nnue.oneInput(64 * currentMove.capturedPieceType + capturedSquare);
            }

            //if castles move the rook back.
            if (currentMove.pieceType >> 1 == _nKing >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare))==2)
            {
                if (currentMove.finishSquare-currentMove.startSquare==2)
                {
                    //kingside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= KING_ROOK_POS[currentMove.pieceType & 1] >> 2;
                    pieces[_nRooks+(currentMove.pieceType & 1)] += KING_ROOK_POS[currentMove.pieceType & 1];

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2];

                    //update nnue.
                    nnue.oneInput(64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1]);
                    nnue.zeroInput(64 * (_nRooks+(currentMove.pieceType & 1)) + KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2);
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1];

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];

                    //update nnue.
                    nnue.oneInput(64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]);
                    nnue.zeroInput(64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3);
                }
            }

            updateOccupied();
        }

        void unpackMove(U32 chessMove)
        {
            currentMove.pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK);
            currentMove.startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
            currentMove.finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            currentMove.enPassant = (chessMove & MOVEINFO_ENPASSANT_MASK);
            currentMove.capturedPieceType = (chessMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
            currentMove.finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
        }

        void makeMove(U32 chessMove)
        {
            unpackMove(chessMove);

            //save zHash.
            hashHistory.push_back(zHashPieces ^ zHashState);

            //move pieces.
            movePieces();

            //update history.
            stateHistory.push_back(current);
            moveHistory.push_back(chessMove);

            //turn increment can be done in zHashPieces.
            zHashPieces ^= randomNums[ZHASH_TURN];
            zHashState = 0;

            //irrev move.
            if (currentMove.pieceType >> 1 == _nPawns >> 1 || currentMove.capturedPieceType != 15 ||
                (currentMove.pieceType >> 1 == _nKing >> 1 && abs((int)currentMove.finishSquare - (int)currentMove.startSquare) == 2))
            {
                irrevMoveInd.push_back(moveHistory.size() - 1);
            }

            //if double-pawn push, set en-passant square.
            //otherwise, set en-passant square to -1.
            if (currentMove.pieceType >> 1 == _nPawns >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare)) == 16)
            {
                current.enPassantSquare = currentMove.finishSquare-8+16*(currentMove.pieceType & 1);
                zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
            }
            else
            {
                current.enPassantSquare = -1;
            }

            if (currentMove.pieceType >> 1 == _nRooks >> 1)
            {
                if (currentMove.startSquare == (U32)KING_ROOK_SQUARE[currentMove.pieceType & 1])
                {
                    current.canKingCastle[currentMove.pieceType & 1] = false;
                }
                else if (currentMove.startSquare == (U32)QUEEN_ROOK_SQUARE[currentMove.pieceType & 1])
                {
                    current.canQueenCastle[currentMove.pieceType & 1] = false;
                }
            }
            else if (currentMove.pieceType >> 1 == _nKing >> 1)
            {
                current.canKingCastle[currentMove.pieceType & 1] = false;
                current.canQueenCastle[currentMove.pieceType & 1] = false;
            }

            if (currentMove.capturedPieceType >> 1 == _nRooks >> 1)
            {
                if (currentMove.finishSquare == (U32)KING_ROOK_SQUARE[currentMove.capturedPieceType & 1])
                {
                    current.canKingCastle[currentMove.capturedPieceType & 1] = false;
                }
                else if (currentMove.finishSquare == (U32)QUEEN_ROOK_SQUARE[currentMove.capturedPieceType & 1])
                {
                    current.canQueenCastle[currentMove.capturedPieceType & 1] = false;
                }
            }

            //update castling rights for zHash.
            if (current.canKingCastle[0]) {zHashState ^= randomNums[ZHASH_CASTLES[0]];}
            if (current.canKingCastle[1]) {zHashState ^= randomNums[ZHASH_CASTLES[1]];}
            if (current.canQueenCastle[0]){zHashState ^= randomNums[ZHASH_CASTLES[2]];}
            if (current.canQueenCastle[1]){zHashState ^= randomNums[ZHASH_CASTLES[3]];}
        }

        void unmakeMove()
        {
            //unmake most recent move and update gameState.
            current = stateHistory.back();
            unpackMove(moveHistory.back());
            unMovePieces();


            //revert zhash for gamestate.
            zHashPieces ^= randomNums[ZHASH_TURN];
            zHashState = 0;

            if (current.enPassantSquare != -1)
            {
                zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
            }

            if (current.canKingCastle[0]) {zHashState ^= randomNums[ZHASH_CASTLES[0]];}
            if (current.canKingCastle[1]) {zHashState ^= randomNums[ZHASH_CASTLES[1]];}
            if (current.canQueenCastle[0]){zHashState ^= randomNums[ZHASH_CASTLES[2]];}
            if (current.canQueenCastle[1]){zHashState ^= randomNums[ZHASH_CASTLES[3]];}

            stateHistory.pop_back();
            moveHistory.pop_back();
            hashHistory.pop_back();

            if (irrevMoveInd.size() && irrevMoveInd.back() >= (int)moveHistory.size())
            {
                irrevMoveInd.pop_back();
            }
        }

        void makeNullMove()
        {
            stateHistory.push_back(current);
            moveHistory.push_back(0);
            hashHistory.push_back(zHashPieces ^ zHashState);

            zHashPieces ^= randomNums[ZHASH_TURN];
            zHashState = 0;

            irrevMoveInd.push_back(moveHistory.size() - 1);

            current.enPassantSquare = -1;

            if (current.canKingCastle[0]) {zHashState ^= randomNums[ZHASH_CASTLES[0]];}
            if (current.canKingCastle[1]) {zHashState ^= randomNums[ZHASH_CASTLES[1]];}
            if (current.canQueenCastle[0]){zHashState ^= randomNums[ZHASH_CASTLES[2]];}
            if (current.canQueenCastle[1]){zHashState ^= randomNums[ZHASH_CASTLES[3]];}
        }

        void unmakeNullMove()
        {
            current = stateHistory.back();

            zHashPieces ^= randomNums[ZHASH_TURN];
            zHashState = 0;

            irrevMoveInd.pop_back();

            if (current.enPassantSquare != -1)
            {
                zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
            }

            if (current.canKingCastle[0]) {zHashState ^= randomNums[ZHASH_CASTLES[0]];}
            if (current.canKingCastle[1]) {zHashState ^= randomNums[ZHASH_CASTLES[1]];}
            if (current.canQueenCastle[0]){zHashState ^= randomNums[ZHASH_CASTLES[2]];}
            if (current.canQueenCastle[1]){zHashState ^= randomNums[ZHASH_CASTLES[3]];}

            stateHistory.pop_back();
            moveHistory.pop_back();
            hashHistory.pop_back();
        }

        int regularEval()
        {
            return nnue.forward() * (1-2*(int)(moveHistory.size() & 1));
        }

        int evaluateBoard()
        {
            //assume we are not in check.
            bool side = moveHistory.size() & 1;
            bool stalemate = stalemateCheck(side);

            return stalemate ? 0 : regularEval();
        }

        std::vector<std::pair<U32,int> > orderCaptures()
        {
            //order captures/promotions.
            scoredMoves.clear();

            for (const auto &move: moveBuffer)
            {
                U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                int score = 16 * (15 - capturedPieceType) + pieceType;

                if (pieceType < capturedPieceType && pieceType >= _nQueens)
                {
                    int seeCheck = seeCaptures(move, pieces, occupied);
                    if (seeCheck < 0) {score = seeCheck;}
                }

                scoredMoves.push_back(std::pair<U32, int>(move, score));
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }

        std::vector<std::pair<U32,int> > orderQuiets()
        {
            //order quiet moves by history.
            scoredMoves.clear();

            for (const auto &move: moveBuffer)
            {
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 startSquare = (move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
                U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                int moveScore = history.scores[pieceType][finishSquare];
                if (pieceType & 1)
                {
                    moveScore += PIECE_TABLES_START[pieceType >> 1][finishSquare] - PIECE_TABLES_START[pieceType >> 1][startSquare];
                }
                else
                {
                    moveScore += PIECE_TABLES_START[pieceType >> 1][finishSquare ^ 56] - PIECE_TABLES_START[pieceType >> 1][startSquare ^ 56];
                }
                scoredMoves.push_back(std::pair<U32,int>(move, moveScore));
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }
        
        std::vector<std::pair<U32, int> > orderQMoves()
        {
            scoredMoves.clear();

            for (const auto &move: moveBuffer)
            {
                U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;

                if (pieceType >= capturedPieceType || pieceType < _nQueens || seeCaptures(move, pieces, occupied) >= 0)
                {
                    int score = 16 * (15 - capturedPieceType) + pieceType;
                    scoredMoves.push_back(std::pair<U32, int>(move, score));
                }
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }

        std::vector<std::pair<U32,int> > orderQMovesInCheck()
        {
            scoredMoves.clear();

            for (const auto &move: moveBuffer)
            {
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
                U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                if (capturedPieceType != 15 || pieceType != finishPieceType)
                {
                    int score = seeCaptures(move, pieces, occupied);
                    scoredMoves.push_back(std::pair<U32,int>(move, score));
                }
                else
                {
                    //non-capture moves.
                    scoredMoves.push_back(std::pair<U32,int>(move, 0));
                }
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }
};

#endif // BOARD_H_INCLUDED
