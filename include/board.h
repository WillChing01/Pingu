#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <vector>
#include <algorithm>
#include <iostream>

#include "constants.h"
#include "bitboard.h"

#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "magic.h"

#include "evaluation.h"

#include "transposition.h"

struct gameState
{
    bool canKingCastle[2];
    bool canQueenCastle[2];
    int enPassantSquare;
};

struct moveInfo
{
    U32 pieceType;
    U32 startSquare;
    U32 finishSquare;
    bool enPassant;
    U32 capturedPieceType;
    U32 finishPieceType;
};

static const std::array<int,6> seeValues = 
{{
    20000,
    1000,
    525,
    350,
    350,
    100,
}};

class Board {
    public:
        U64 pieces[12]={};

        U64 occupied[2]={0,0};
        U64 attacked[2]={0,0};

        std::vector<gameState> stateHistory;
        std::vector<U32> moveHistory;
        std::vector<U32> hashHistory;

        std::vector<U32> captureBuffer;
        std::vector<U32> quietBuffer;
        std::vector<U32> moveBuffer;
        std::vector<std::pair<U32,int> > scoredMoves;
        U32 killerMoves[128][2] = {};

        gameState current = {
            .canKingCastle = {true,true},
            .canQueenCastle = {true,true},
            .enPassantSquare = -1,
        };

        moveInfo currentMove = {};

        static const U32 _nKing=0;
        static const U32 _nQueens=2;
        static const U32 _nRooks=4;
        static const U32 _nBishops=6;
        static const U32 _nKnights=8;
        static const U32 _nPawns=10;

        const int piecePhases[6] = {0,4,2,1,1,0};

        int phase = 24;
        int shiftedPhase = (64 * phase + 3)/6;

        int materialStart = 0;
        int materialEnd = 0;

        int pstStart = 0;
        int pstEnd = 0;

        //overall zHash is XOR of these two.
        U64 zHashPieces = 0;
        U64 zHashState = 0;

        //SEE.
        int gain[32]={};

        //history table, history[pieceType][to_square]
        int history[12][64] = {};

        //temp variable for move appending.
        U32 newMove;

        Board()
        {
            //default constructor for regular games.
            pieces[_nKing]=WHITE_KING;
            pieces[_nKing+1]=BLACK_KING;

            pieces[_nQueens]=WHITE_QUEENS;
            pieces[_nQueens+1]=BLACK_QUEENS;

            pieces[_nRooks]=WHITE_ROOKS;
            pieces[_nRooks+1]=BLACK_ROOKS;

            pieces[_nBishops]=WHITE_BISHOPS;
            pieces[_nBishops+1]=BLACK_BISHOPS;

            pieces[_nKnights]=WHITE_KNIGHTS;
            pieces[_nKnights+1]=BLACK_KNIGHTS;

            pieces[_nPawns]=WHITE_PAWNS;
            pieces[_nPawns+1]=BLACK_PAWNS;

            updateOccupied();
            updateAttacked(0); updateAttacked(1);

            zHashHardUpdate();
            phaseHardUpdate();
            evalHardUpdate();
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
            shiftedPhase = (64 * phase + 3)/6;
        }

        void evalHardUpdate()
        {
            materialStart = 0;
            materialEnd = 0;
            pstStart = 0;
            pstEnd = 0;
            for (int i=0;i<12;i+=2)
            {
                materialStart += (__builtin_popcountll(pieces[i]) - __builtin_popcountll(pieces[i+1])) * PIECE_VALUES_START[i >> 1];
                materialEnd += (__builtin_popcountll(pieces[i]) - __builtin_popcountll(pieces[i+1])) * PIECE_VALUES_END[i >> 1];
                U64 white = pieces[i];
                U64 black = pieces[i+1];
                U64 x;
                while (white)
                {
                    x = popLSB(white);
                    pstStart += PIECE_TABLES_START[i >> 1][x ^ 56];
                    pstEnd += PIECE_TABLES_END[i >> 1][x ^ 56];
                }
                while (black)
                {
                    x = popLSB(black);
                    pstStart -= PIECE_TABLES_START[i >> 1][x];
                    pstEnd -= PIECE_TABLES_END[i >> 1][x];
                }
            }
        }

        void setPositionFen(const std::string &fen)
        {
            //reset history.
            stateHistory.clear();
            moveHistory.clear();
            hashHistory.clear();

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
            moveHistory.clear();
            if (temp[1] == "b") {moveHistory.push_back(0);}

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
            evalHardUpdate();
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
            updateOccupied();
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

        void appendMove(U32 pieceType, U32 startSquare, U32 finishSquare, bool shouldCheck=false)
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

                //always check en-passant.
                shouldCheck = true;

                //check for captured piece.
                capturedPieceType = (_nPawns+((pieceType+1u) & 1u));
            }

            if (shouldCheck)
            {
                //check if move is legal (does not leave king in check).
                //move pieces.
                pieces[pieceType] -= 1ull << (startSquare);
                pieces[pieceType] += 1ull << (finishSquare);
                if (capturedPieceType != 15)
                {
                    pieces[capturedPieceType] -= 1ull << (finishSquare+(int)(enPassant)*(-8+16*(pieceType & 1)));
                }
                updateOccupied();
                bool isBad = isInCheck(pieceType & 1);

                //unmove pieces.
                pieces[pieceType] += 1ull << (startSquare);
                pieces[pieceType] -= 1ull << (finishSquare);
                if (capturedPieceType != 15)
                {
                    pieces[capturedPieceType] += 1ull << (finishSquare+(int)(enPassant)*(-8+16*(pieceType & 1)));
                }
                updateOccupied();

                if (isBad) {return;}
            }

            newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            (enPassant << MOVEINFO_ENPASSANT_OFFSET) |
            (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

            //check for pawn promotion.
            if (pieceType >> 1 == _nPawns >> 1 && finishSquare >> 3 == 7-7*(pieceType & 1))
            {
                //promotion.
                for (U32 i=_nQueens+(pieceType & 1);i<_nPawns;i+=2)
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

//        U32 isSquareAttacked(bool side, U32 square)
//        {
//            U64 b = occupied[0] | occupied[1];
//
//            U64 isAttacked = kingAttacks(1ull << square) & pieces[_nKing+(int)(!side)];
//            isAttacked |= magicRookAttacks(b,square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
//            isAttacked |= magicBishopAttacks(b,square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
//            isAttacked |= knightAttacks(1ull << square) & pieces[_nKnights+(int)(!side)];
//            isAttacked |= pawnAttacks(1ull << square,(int)(side)) & pieces[_nPawns+(int)(!side)];
//
//            return __builtin_popcountll(isAttacked);
//        }

        U64 getCheckPiece(bool side, U32 square)
        {
            //assumes a single piece is giving check.
            U64 b = occupied[0] | occupied[1];

            if (U64 bishop = magicBishopAttacks(b,square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
            {
                //diagonal attack.
                return bishop;
            }
            else if (U64 rook = magicRookAttacks(b,square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
            {
                //rook-like attack.
                return rook;
            }
            else if (U64 knight = knightAttacks(1ull << square) & pieces[_nKnights+(int)(!side)])
            {
                //knight attack.
                return knight;
            }
            else
            {
                //pawn.
                return pawnAttacks(1ull << square,side) & pieces[_nPawns+(int)(!side)];
            }
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

        void generateCaptures(bool side, int numChecks = 0)
        {
            updateOccupied();
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
                    if (!side) {x |= ((pawnPosBoard & FILE_7) << 8) & (~p);}
                    else {x |= ((pawnPosBoard & FILE_2) >> 8) & (~p);}

                    while (x) {appendPawnCapture(_nPawns+(int)(side), pos,popLSB(x), false, (pawnPosBoard & pinned)!=0);}
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
                    if (!side) {x |= ((pawnPosBoard & FILE_7) << 8) & (~p);}
                    else {x |= ((pawnPosBoard & FILE_2) >> 8) & (~p);}

                    while (x) {appendPawnCapture(_nPawns+(int)(side), pos,popLSB(x), false, true);}
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
            updateOccupied();
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
                        x |= ((pawnPosBoard & (~FILE_7)) << 8) & (~p);
                        x |= ((((pawnPosBoard & FILE_2) << 8) & (~p)) << 8) & (~p);
                    }
                    else
                    {
                        x |= ((pawnPosBoard & (~FILE_2)) >> 8 & (~p));
                        x |= ((((pawnPosBoard & FILE_7) >> 8) & (~p)) >> 8) & (~p);
                    }

                    while (x) {appendQuiet(_nPawns+(int)(side), pos,popLSB(x), (pawnPosBoard & pinned)!=0);}
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
                        x |= ((pawnPosBoard & (~FILE_7)) << 8) & (~p);
                        x |= ((((pawnPosBoard & FILE_2) << 8) & (~p)) << 8) & (~p);
                    }
                    else
                    {
                        x |= ((pawnPosBoard & (~FILE_2)) >> 8 & (~p));
                        x |= ((((pawnPosBoard & FILE_7) >> 8) & (~p)) >> 8) & (~p);
                    }

                    while (x) {appendQuiet(_nPawns+(int)(side), pos,popLSB(x), true);}
                }

                //bishops.
                temp = pieces[_nBishops+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicBishopAttacks(p,pos) & ~p;
                    while (x) {appendQuiet(_nBishops+(int)(side), pos, popLSB(x), true);}
                }

                //knights.
                temp = pieces[_nKnights+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = knightAttacks(1ull << pos) & ~p;
                    while (x) {appendQuiet(_nKnights+(int)(side), pos, popLSB(x), true);}
                }

                //rook.
                temp = pieces[_nRooks+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicRookAttacks(p,pos) & ~p;
                    while (x) {appendQuiet(_nRooks+(int)(side), pos, popLSB(x), true);}
                }

                //queen.
                temp = pieces[_nQueens+(int)(side)];
                while (temp)
                {
                    pos = popLSB(temp);
                    x = magicQueenAttacks(p,pos) & ~p;
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
            updateOccupied();
            bool inCheck = isInCheck(side);
            U32 numChecks = 0;
            if (inCheck) {numChecks = isInCheckDetailed(side);}
            generateCaptures(side, numChecks);
            generateQuiets(side, numChecks);
            return inCheck;
        }

        bool generateEvalMoves(bool side)
        {
            //only need one legal move to exit early.
            moveBuffer.clear();
            //generate all pseudo-legal moves.
            updateOccupied();
            updateAttacked(!side);

            if (!(bool)(pieces[_nKing+(int)(side)] & attacked[(int)(!side)]))
            {
                //regular moves.
                U32 pos; U64 x;

                //castling.
                if (current.canKingCastle[(int)(side)] &&
                    !(bool)(KING_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1])) &&
                    !(bool)(KING_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                {
                    //kingside castle.
                    pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                    appendMove(_nKing+(int)(side), pos, pos+2, false);
                    return false;
                }
                if (current.canQueenCastle[(int)(side)] &&
                    !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1])) &&
                    !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                {
                    //queenside castle.
                    pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                    appendMove(_nKing+(int)(side), pos, pos-2, false);
                    return false;
                }

                U64 pinned = getPinnedPieces(side);
                U64 p = (occupied[0] | occupied[1]);

                //knights.
                U64 knights = pieces[_nKnights+(int)(side)] & ~pinned;
                while (knights)
                {
                    pos = popLSB(knights);
                    x = knightAttacks(1ull << pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nKnights+(int)(side), pos, popLSB(x), false);
                        if (moveBuffer.size() > 0) {return false;}
                    }
                }

                //bishops.
                U64 bishops = pieces[_nBishops+(int)(side)];
                while (bishops)
                {
                    pos = popLSB(bishops);
                    x = magicBishopAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nBishops+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);
                        if (moveBuffer.size() > 0) {return false;}
                    }
                }

                //pawns.
                U64 pawns = pieces[_nPawns+(int)(side)];
                U64 pawnPosBoard;
                while (pawns)
                {
                    pos = popLSB(pawns);
                    pawnPosBoard = 1ull << pos;
                    //en passant square included.
                    x = pawnAttacks(pawnPosBoard,side) & ~occupied[(int)(side)];

                    if (current.enPassantSquare != -1) {x &= (occupied[(int)(!side)] | (1ull << current.enPassantSquare));}
                    else {x &= occupied[(int)(!side)];}

                    //move forward.
                    if (side==0)
                    {
                        x |= (pawnPosBoard << 8) & (~p);
                        x |= ((((pawnPosBoard & FILE_2) << 8) & (~p)) << 8) & (~p);
                    }
                    else
                    {
                        x |= (pawnPosBoard >> 8 & (~p));
                        x |= ((((pawnPosBoard & FILE_7) >> 8) & (~p)) >> 8) & (~p);
                    }

                    while (x)
                    {
                        appendMove(_nPawns+(int)(side), pos,popLSB(x), (pawnPosBoard & pinned)!=0);
                        if (moveBuffer.size() > 0) {return false;}
                    }
                }

                //rook.
                U64 rooks = pieces[_nRooks+(int)(side)];
                while (rooks)
                {
                    pos = popLSB(rooks);
                    x = magicRookAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nRooks+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);
                        if (moveBuffer.size() > 0) {return false;}
                    }
                }

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nQueens+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);
                        if (moveBuffer.size() > 0) {return false;}
                    }
                }

                //king.
                pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x)
                {
                    appendMove(_nKing+(int)(side), pos, popLSB(x), false);
                    if (moveBuffer.size() > 0) {return false;}
                }

                return false;
            }
            else if (isInCheckDetailed(side) == 1)
            {
                //single check.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x)
                {
                    appendMove(_nKing+(int)(side), pos, popLSB(x),true);
                    if (moveBuffer.size() > 0) {return true;}
                }

                U64 p = (occupied[0] | occupied[1]);

                //pawns.
                U64 pawns = pieces[_nPawns+(int)(side)];
                U64 pawnPosBoard;
                while (pawns)
                {
                    pos = popLSB(pawns);
                    pawnPosBoard = 1ull << pos;

                    //en passant square included.
                    x = pawnAttacks(pawnPosBoard,side) & ~occupied[(int)(side)];

                    if (current.enPassantSquare != -1) {x &= (occupied[(int)(!side)] | (1ull << current.enPassantSquare));}
                    else {x &= occupied[(int)(!side)];}

                    //move forward.
                    if (side==0)
                    {
                        x |= (pawnPosBoard << 8) & (~p);
                        x |= ((((pawnPosBoard & FILE_2) << 8) & (~p)) << 8) & (~p);
                    }
                    else
                    {
                        x |= (pawnPosBoard >> 8 & (~p));
                        x |= ((((pawnPosBoard & FILE_7) >> 8) & (~p)) >> 8) & (~p);
                    }

                    while (x)
                    {
                        appendMove(_nPawns+(int)(side), pos,popLSB(x), true);
                        if (moveBuffer.size() > 0) {return true;}
                    }
                }

                //bishops.
                U64 bishops = pieces[_nBishops+(int)(side)];
                while (bishops)
                {
                    pos = popLSB(bishops);
                    x = magicBishopAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nBishops+(int)(side), pos, popLSB(x), true);
                        if (moveBuffer.size() > 0) {return true;}
                    }
                }

                //knights.
                U64 knights = pieces[_nKnights+(int)(side)];
                while (knights)
                {
                    pos = popLSB(knights);
                    x = knightAttacks(1ull << pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nKnights+(int)(side), pos, popLSB(x), true);
                        if (moveBuffer.size() > 0) {return true;}
                    }
                }

                //rook.
                U64 rooks = pieces[_nRooks+(int)(side)];
                while (rooks)
                {
                    pos = popLSB(rooks);
                    x = magicRookAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nRooks+(int)(side), pos, popLSB(x), true);
                        if (moveBuffer.size() > 0) {return true;}
                    }
                }

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nQueens+(int)(side), pos, popLSB(x), true);
                        if (moveBuffer.size() > 0) {return true;}
                    }
                }

                return true;
            }
            else
            {
                //multiple check. only king moves allowed.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x)
                {
                    appendMove(_nKing+(int)(side), pos, popLSB(x), true);
                    if (moveBuffer.size() > 0) {return true;}
                }

                return true;
            }
        }

        void updatePST(int pieceType, int finishPieceType, int fromSquare, int toSquare)
        {
            if (pieceType & 1)
            {
                pstStart -= PIECE_TABLES_START[finishPieceType >> 1][toSquare] - PIECE_TABLES_START[pieceType >> 1][fromSquare];
                pstEnd -= PIECE_TABLES_END[finishPieceType >> 1][toSquare] - PIECE_TABLES_END[pieceType >> 1][fromSquare];
            }
            else
            {
                pstStart += PIECE_TABLES_START[finishPieceType >> 1][toSquare ^ 56] - PIECE_TABLES_START[pieceType >> 1][fromSquare ^ 56];
                pstEnd += PIECE_TABLES_END[finishPieceType >> 1][toSquare ^ 56] - PIECE_TABLES_END[pieceType >> 1][fromSquare ^ 56];
            }
        }

        void updateCapturePST(int capturedPieceType, int capturedSquare, bool reverse)
        {
            if (capturedPieceType & 1)
            {
                pstStart += PIECE_TABLES_START[capturedPieceType >> 1][capturedSquare] * (1-2*(int)(reverse));
                pstEnd += PIECE_TABLES_END[capturedPieceType >> 1][capturedSquare] * (1-2*(int)(reverse));
            }
            else
            {
                pstStart -= PIECE_TABLES_START[capturedPieceType >> 1][capturedSquare ^ 56] * (1-2*(int)(reverse));
                pstEnd -= PIECE_TABLES_END[capturedPieceType >> 1][capturedSquare ^ 56] * (1-2*(int)(reverse));
            }
        }

        void movePieces()
        {
            //remove piece from start square;
            pieces[currentMove.pieceType] -= 1ull << (currentMove.startSquare);
            zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];

            //add piece to end square, accounting for promotion.
            pieces[currentMove.finishPieceType] += 1ull << (currentMove.finishSquare);
            zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];

            //update pst.
            updatePST(currentMove.pieceType, currentMove.finishPieceType, currentMove.startSquare, currentMove.finishSquare);

            //update material+phase on promotion.
            if (currentMove.pieceType != currentMove.finishPieceType)
            {
                phase += piecePhases[currentMove.finishPieceType >> 1];
                shiftedPhase = (64 * phase + 3) / 6;

                materialStart += (PIECE_VALUES_START[currentMove.finishPieceType >> 1] - PIECE_VALUES_START[currentMove.pieceType >> 1]) * (1-2*(int)(currentMove.pieceType & 1));
                materialEnd += (PIECE_VALUES_END[currentMove.finishPieceType >> 1] - PIECE_VALUES_END[currentMove.pieceType >> 1]) * (1-2*(int)(currentMove.pieceType & 1));
            }

            //remove any captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                int capturedSquare = currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1));
                pieces[currentMove.capturedPieceType] -= 1ull << capturedSquare;
                zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + capturedSquare];

                //update the game phase.
                phase -= piecePhases[currentMove.capturedPieceType >> 1];
                shiftedPhase = (64 * phase + 3) / 6;

                //update material.
                materialStart -= PIECE_VALUES_START[currentMove.capturedPieceType >> 1] * (1-2*(int)(currentMove.capturedPieceType & 1));
                materialEnd -= PIECE_VALUES_END[currentMove.capturedPieceType >> 1] * (1-2*(int)(currentMove.capturedPieceType & 1));

                //update pst.
                updateCapturePST(currentMove.capturedPieceType, capturedSquare, false);
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

                    //update pst.
                    updatePST(_nRooks+(currentMove.pieceType & 1), _nRooks+(currentMove.pieceType & 1), KING_ROOK_SQUARE[currentMove.pieceType & 1], KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2);
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1];
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];

                    //update pst.
                    updatePST(_nRooks+(currentMove.pieceType & 1), _nRooks+(currentMove.pieceType & 1), QUEEN_ROOK_SQUARE[currentMove.pieceType & 1], QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3);
                }
            }
        }

        void unMovePieces()
        {
            //remove piece from destination square.
            pieces[currentMove.finishPieceType] -= 1ull << (currentMove.finishSquare);
            zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];
            
            //add piece to start square.
            pieces[currentMove.pieceType] += 1ull << (currentMove.startSquare);
            zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];

            //update pst.
            updatePST(currentMove.finishPieceType, currentMove.pieceType, currentMove.finishSquare, currentMove.startSquare);

            //update material+phase on promotion.
            if (currentMove.pieceType != currentMove.finishPieceType)
            {
                phase -= piecePhases[currentMove.finishPieceType >> 1];
                shiftedPhase = (64 * phase + 3) / 6;

                materialStart -= (PIECE_VALUES_START[currentMove.finishPieceType >> 1] - PIECE_VALUES_START[currentMove.pieceType >> 1]) * (1-2*(int)(currentMove.pieceType & 1));
                materialEnd -= (PIECE_VALUES_END[currentMove.finishPieceType >> 1] - PIECE_VALUES_END[currentMove.pieceType >> 1]) * (1-2*(int)(currentMove.pieceType & 1));
            }

            //add back captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                int capturedSquare = currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1));
                pieces[currentMove.capturedPieceType] += 1ull << capturedSquare;
                zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + capturedSquare];

                //update the game phase.
                phase += piecePhases[currentMove.capturedPieceType >> 1];
                shiftedPhase = (64 * phase + 3) / 6;

                //update material.
                materialStart += PIECE_VALUES_START[currentMove.capturedPieceType >> 1] * (1-2*(int)(currentMove.capturedPieceType & 1));
                materialEnd += PIECE_VALUES_END[currentMove.capturedPieceType >> 1] * (1-2*(int)(currentMove.capturedPieceType & 1));

                //update pst.
                updateCapturePST(currentMove.capturedPieceType, capturedSquare, true);
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

                    //update pst.
                    updatePST(_nRooks+(currentMove.pieceType & 1), _nRooks+(currentMove.pieceType & 1), KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2, KING_ROOK_SQUARE[currentMove.pieceType & 1]);
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1];

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];

                    //update pst.
                    updatePST(_nRooks+(currentMove.pieceType & 1), _nRooks+(currentMove.pieceType & 1), QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3, QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]);
                }
            }
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
        }

        void makeNullMove()
        {
            stateHistory.push_back(current);
            moveHistory.push_back(0);
            hashHistory.push_back(zHashPieces ^ zHashState);

            zHashPieces ^= randomNums[ZHASH_TURN];
            zHashState = 0;

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

        int seeCaptures(U32 chessMove)
        {
            //perform static evaluation exchange (SEE).
            U32 startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
            U32 finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            U32 attackingPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
            U32 capturedPieceType = (chessMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;

            bool side = attackingPieceType & 1;
            int d = 0;
            gain[0] = (capturedPieceType != 15 ? seeValues[capturedPieceType >> 1] : 0)
            + (attackingPieceType == ((chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET) ? 0 : seeValues[attackingPieceType >> 1] - seeValues[_nPawns >> 1]);
            U64 attackingPieceBB = 1ull << startSquare;
            U64 occ = occupied[0] | occupied[1];
            if (chessMove & MOVEINFO_ENPASSANT_MASK) {occ ^= 1ull << (finishSquare - 8 + side * 16);}

            U64 attackersBB = 0;
            attackersBB ^= kingAttacks(1ull << finishSquare) & (pieces[_nKing] | pieces[_nKing+1]);
            attackersBB ^= pawnAttacks(1ull << finishSquare, 0) & pieces[_nPawns+1];
            attackersBB ^= pawnAttacks(1ull << finishSquare, 1) & pieces[_nPawns];
            attackersBB ^= knightAttacks(1ull << finishSquare) & (pieces[_nKnights] | pieces[_nKnights+1]);
            attackersBB ^= magicRookAttacks(occ, finishSquare) & (pieces[_nRooks] | pieces[_nRooks+1] | pieces[_nQueens] | pieces[_nQueens+1]);
            attackersBB ^= magicBishopAttacks(occ, finishSquare) & (pieces[_nBishops] | pieces[_nBishops+1] | pieces[_nQueens] | pieces[_nQueens+1]);

            attackingPieceType = attackingPieceType >> 1;

            do
            {
                d++; side = !side;
                gain[d] = -gain[d-1] + seeValues[attackingPieceType];
                if (std::max(-gain[d-1],gain[d]) < 0) {break;}
                attackersBB ^= attackingPieceBB;
                occ ^= attackingPieceBB;

                //update possible x-ray attacks.
                if (attackingPieceType == (_nRooks >> 1) || attackingPieceType == (_nQueens >> 1))
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
            } while (attackingPieceBB);
            while (--d) {gain[d-1] = -std::max(-gain[d-1], gain[d]);}

            return gain[0];
        }

        int regularEval()
        {
            int startTotal = materialStart + pstStart;
            int endTotal = materialEnd + pstEnd;

            U64 x;
            U64 b = occupied[0] | occupied[1];

            //mobility.
            x = pieces[_nRooks];
            while (x)
            {
                int mob = magicRookMob(b, popLSB(x));
                startTotal += MOB_ROOK_START * mob;
                endTotal += MOB_ROOK_END * mob;
            }

            x = pieces[_nRooks + 1];
            while (x)
            {
                int mob = magicRookMob(b, popLSB(x));
                startTotal -= MOB_ROOK_START * mob;
                endTotal -= MOB_ROOK_END * mob;
            }

            x = pieces[_nBishops];
            while (x)
            {
                int mob = magicBishopMob(b, popLSB(x));
                startTotal += MOB_BISHOP_START * mob;
                endTotal += MOB_BISHOP_END * mob;
            }

            x = pieces[_nBishops + 1];
            while (x)
            {
                int mob = magicBishopMob(b, popLSB(x));
                startTotal -= MOB_BISHOP_START * mob;
                endTotal -= MOB_BISHOP_END * mob;
            }

            return (((startTotal * shiftedPhase) + (endTotal * (256 - shiftedPhase))) / 256) * (1-2*(int)(moveHistory.size() & 1));
        }

        int evaluateBoard()
        {
            bool turn = moveHistory.size() & 1;
            bool inCheck = generateEvalMoves(turn);

            if (moveBuffer.size() > 0) {return regularEval();}
            return inCheck ? -MATE_SCORE : 0;
        }

        std::vector<std::pair<U32,int> > orderCaptures()
        {
            //order captures/promotions.
            updateOccupied();
            scoredMoves.clear();

            for (const auto &move: moveBuffer)
            {
                scoredMoves.push_back(std::pair<U32,int>(move,seeCaptures(move)));
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
                scoredMoves.push_back(
                    std::pair<U32,int>(
                        move,
                        history[(move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET][(move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET]
                    )
                );
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }

        std::vector<std::pair<U32,int> > orderQMoves(const int threshhold = 0)
        {
            //assumes that updateOccupied() has been called immediately before.
            scoredMoves.clear();

            for (const auto &move: moveBuffer)
            {
                U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
                U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                if (capturedPieceType != 15 || pieceType != finishPieceType)
                {
                    int score = seeCaptures(move);
                    if (score >= threshhold) {scoredMoves.push_back(std::pair<U32,int>(move, score));}
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

        void ageHistory(const int factor = 16)
        {
            for (int i=0;i<12;i++)
            {
                for (int j=0;j<64;j++) {history[i][j] /= factor;}
            }
        }

        void clearHistory()
        {
            for (int i=0;i<12;i++)
            {
                for (int j=0;j<64;j++) {history[i][j] = 0;}
            }
        }
};

#endif // BOARD_H_INCLUDED
