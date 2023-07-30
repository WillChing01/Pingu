#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <vector>
#include <bitset>
#include <windows.h>
#include <cctype>
#include <algorithm>
#include <iostream>

#include "constants.h"
#include "bitboard.h"

#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "slider.h"
#include "magic.h"

#include "evaluation.h"

#include "transposition.h"

using namespace std;

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

class Board {
    public:
        U64 pieces[12]={};

        U64 occupied[2]={0,0};
        U64 attacked[2]={0,0};

        vector<gameState> stateHistory;
        vector<U32> moveHistory;

        vector<U32> captureBuffer;
        vector<U32> nonCaptureBuffer;
        vector<U32> moveBuffer;
        vector<pair<U32,int> > scoredMoves;
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

        //overall zHash is XOR of these two.
        U64 zHashPieces = 0;
        U64 zHashState = 0;

        //draw hash table.
        U64 drawHash[128] = {};

        //SEE.
        int gain[32]={};

        //pawn hash tables.
        static const U64 pawnHashMask = 1023;
        pair<U64,int> pawnHash[pawnHashMask + 1] = {};
        U64 zHashPawns = 0;

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
        };

        void zHashHardUpdate()
        {
            zHashPieces = 0;
            zHashState = 0;

            zHashPawns = 0;

            U64 x = pieces[_nPawns];
            while (x) {zHashPawns ^= randomNums[ZHASH_PIECES[_nPawns] + popLSB(x)];}

            x = pieces[_nPawns+1];
            while (x) {zHashPawns ^= randomNums[ZHASH_PIECES[_nPawns+1] + popLSB(x)];}

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

        void setPositionFen(string fen)
        {
            vector<string> temp; temp.push_back("");

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
                if (pieceType >> 1 == _nKing >> 1 && abs((int)(finishSquare)-(int)(startSquare))==2)
                {
                    if (finishSquare-startSquare==2)
                    {
                        //kingside.
                        pieces[_nRooks+(pieceType & 1)] -= KING_ROOK_POS[pieceType & 1];
                        pieces[_nRooks+(pieceType & 1)] += KING_ROOK_POS[pieceType & 1] >> 2;
                    }
                    else
                    {
                        //queenside.
                        pieces[_nRooks+(pieceType & 1)] -= QUEEN_ROOK_POS[pieceType & 1];
                        pieces[_nRooks+(pieceType & 1)] += QUEEN_ROOK_POS[pieceType & 1] << 3;
                    }
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
                if (pieceType >> 1 == _nKing >> 1 && abs((int)(finishSquare)-(int)(startSquare))==2)
                {
                    if (finishSquare-startSquare==2)
                    {
                        //kingside.
                        pieces[_nRooks+(pieceType & 1)] += KING_ROOK_POS[pieceType & 1];
                        pieces[_nRooks+(pieceType & 1)] -= KING_ROOK_POS[pieceType & 1] >> 2;
                    }
                    else
                    {
                        //queenside.
                        pieces[_nRooks+(pieceType & 1)] += QUEEN_ROOK_POS[pieceType & 1];
                        pieces[_nRooks+(pieceType & 1)] -= QUEEN_ROOK_POS[pieceType & 1] << 3;
                    }
                }
                updateOccupied();

                if (isBad) {return;}
            }

            U32 newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
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

        std::pair<U32,U32> getCheckPiece(bool side, U32 square)
        {
            //assumes a single piece is giving check.
            U64 b = occupied[0] | occupied[1];

            if (U64 bishop = magicBishopAttacks(b,square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
            {
                //diagonal attack.
                return pair<U32,U32>(_nBishops,__builtin_ctzll(bishop));
            }
            else if (U64 rook = magicRookAttacks(b,square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]))
            {
                //rook-like attack.
                return pair<U32,U32>(_nRooks,__builtin_ctzll(rook));
            }
            else if (U64 knight = knightAttacks(1ull << square) & pieces[_nKnights+(int)(!side)])
            {
                //knight attack.
                return pair<U32,U32>(_nKnights,__builtin_ctzll(knight));
            }
            else
            {
                //pawn.
                return pair<U32,U32>(_nPawns,__builtin_ctzll(pawnAttacks(1ull << square,side) & pieces[_nPawns+(int)(!side)]));
            }
        }

        U64 getPinnedPieces(bool side)
        {
            //generate attacks to the king.
            int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
            U64 b = occupied[0] | occupied[1];

            U64 pinned = 0;

            //for each potentially pinned piece, loop through and recalculate the attacks.
            U64 potentialPinned = magicRookAttacks(b,kingPos) & occupied[(int)(side)];
            while (potentialPinned)
            {
                U64 x = 1ull << popLSB(potentialPinned);
                if ((magicRookAttacks(b & ~x,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)])) != 0) {pinned |= x;}
            }

            potentialPinned = magicBishopAttacks(b,kingPos) & occupied[(int)(side)];
            while (potentialPinned)
            {
                U64 x = 1ull << popLSB(potentialPinned);
                if ((magicBishopAttacks(b & ~x,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)])) != 0) {pinned |= x;}
            }

            return pinned;
        }

        void display()
        {
            //display the current position in console.

            const char symbols[12]={'K','k','Q','q','R','r','B','b','N','n','P','p'};

            HANDLE hConsole=GetStdHandle(STD_OUTPUT_HANDLE);

            const int colours[2]={7,8};
            const int standardColour=15;

            vector<vector<string> > grid;
            for (int i=0;i<8;i++)
            {
                grid.push_back(vector<string>());
                for (int j=0;j<8;j++)
                {
                    grid.back().push_back("[ ]");
                }
            }

            string temp=bitset<64>(0).to_string();
            for (int i=0;i<12;i++)
            {
                temp=bitset<64>(pieces[i]).to_string();
                for (int j=0;j<64;j++)
                {
                    if (temp[j]=='0') {continue;}

                    grid[(63-j)/8][(63-j)%8][1]=symbols[i];
                }
            }

            for (int i=7;i>=0;i--)
            {
                for (int j=0;j<8;j++)
                {
                    SetConsoleTextAttribute(hConsole,colours[(i+j)%2]);
                    cout << grid[i][j][0];
                    SetConsoleTextAttribute(hConsole,standardColour);

                    if (grid[i][j][1]!=' ')
                    {
                        if (isupper(grid[i][j][1])==true)
                        {
                            SetConsoleTextAttribute(hConsole,colours[0]);
                        }
                        else
                        {
                            SetConsoleTextAttribute(hConsole,colours[1]);
                        }
                        cout << char(toupper(grid[i][j][1]));

                        SetConsoleTextAttribute(hConsole,standardColour);
                    }
                    else {cout << ' ';}

                    SetConsoleTextAttribute(hConsole,colours[(i+j)%2]);
                    cout << grid[i][j][2];
                    SetConsoleTextAttribute(hConsole,standardColour);
                } cout << " " << i+1 << endl;
            }
            cout << " A  B  C  D  E  F  G  H" << endl;
        }

        bool generatePseudoQMoves(bool side)
        {
            moveBuffer.clear();
            //generate captures/ check evasions for a quiescence search.
            updateOccupied();
            updateAttacked(!side);

            if (!(bool)(pieces[_nKing+(int)(side)] & attacked[(int)(!side)]))
            {
                //captures only.
                U32 pos; U64 x;

                U64 pinned = getPinnedPieces(side);
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

                    //promotion by moving forward.
                    if (!side) {x |= ((pawnPosBoard & FILE_7) << 8) & (~p);}
                    else {x |= ((pawnPosBoard & FILE_2) >> 8) & (~p);}

                    while (x) {appendMove(_nPawns+(int)(side), pos,popLSB(x), (pawnPosBoard & pinned)!=0);}
                }

                //knights.
                U64 knights = pieces[_nKnights+(int)(side)] & ~pinned;
                while (knights)
                {
                    pos = popLSB(knights);
                    x = knightAttacks(1ull << pos) & ~occupied[(int)(side)] & occupied[(int)(!side)];
                    while (x) {appendMove(_nKnights+(int)(side), pos, popLSB(x), false);}
                }

                //bishops.
                U64 bishops = pieces[_nBishops+(int)(side)];
                while (bishops)
                {
                    pos = popLSB(bishops);
                    x = magicBishopAttacks(p,pos) & ~occupied[(int)(side)] & occupied[(int)(!side)];
                    while (x) {appendMove(_nBishops+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //rook.
                U64 rooks = pieces[_nRooks+(int)(side)];
                while (rooks)
                {
                    pos = popLSB(rooks);
                    x = magicRookAttacks(p,pos) & ~occupied[(int)(side)] & occupied[(int)(!side)];
                    while (x) {appendMove(_nRooks+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)] & occupied[(int)(!side)];
                    while (x) {appendMove(_nQueens+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //king.
                pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)] & occupied[(int)(!side)];
                while (x) {appendMove(_nKing+(int)(side), pos, popLSB(x), false);}

                return false;
            }
            else if (isInCheckDetailed(side) == 1)
            {
                //single check.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x) {appendMove(_nKing+(int)(side), pos, popLSB(x),true);}

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

                    while (x) {appendMove(_nPawns+(int)(side), pos,popLSB(x), true);}
                }

                //bishops.
                U64 bishops = pieces[_nBishops+(int)(side)];
                while (bishops)
                {
                    pos = popLSB(bishops);
                    x = magicBishopAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nBishops+(int)(side), pos, popLSB(x), true);}
                }

                //knights.
                U64 knights = pieces[_nKnights+(int)(side)];
                while (knights)
                {
                    pos = popLSB(knights);
                    x = knightAttacks(1ull << pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nKnights+(int)(side), pos, popLSB(x), true);}
                }

                //rook.
                U64 rooks = pieces[_nRooks+(int)(side)];
                while (rooks)
                {
                    pos = popLSB(rooks);
                    x = magicRookAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nRooks+(int)(side), pos, popLSB(x), true);}
                }

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nQueens+(int)(side), pos, popLSB(x), true);}
                }

                return true;
            }
            else
            {
                //multiple check. only king moves allowed.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x) {appendMove(_nKing+(int)(side), pos, popLSB(x), true);}

                return true;
            }
        }

        bool generatePseudoMoves(bool side)
        {
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
                }
                if (current.canQueenCastle[(int)(side)] &&
                    !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1])) &&
                    !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]))
                {
                    //queenside castle.
                    pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                    appendMove(_nKing+(int)(side), pos, pos-2, false);
                }

                U64 pinned = getPinnedPieces(side);
                U64 p = (occupied[0] | occupied[1]);

                //knights.
                U64 knights = pieces[_nKnights+(int)(side)] & ~pinned;
                while (knights)
                {
                    pos = popLSB(knights);
                    x = knightAttacks(1ull << pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nKnights+(int)(side), pos, popLSB(x), false);}
                }

                //bishops.
                U64 bishops = pieces[_nBishops+(int)(side)];
                while (bishops)
                {
                    pos = popLSB(bishops);
                    x = magicBishopAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nBishops+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
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

                    while (x) {appendMove(_nPawns+(int)(side), pos,popLSB(x), (pawnPosBoard & pinned)!=0);}
                }

                //rook.
                U64 rooks = pieces[_nRooks+(int)(side)];
                while (rooks)
                {
                    pos = popLSB(rooks);
                    x = magicRookAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nRooks+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nQueens+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);}
                }

                //king.
                pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x) {appendMove(_nKing+(int)(side), pos, popLSB(x), false);}

                return false;
            }
            else if (isInCheckDetailed(side) == 1)
            {
                //single check.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x) {appendMove(_nKing+(int)(side), pos, popLSB(x),true);}

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

                    while (x) {appendMove(_nPawns+(int)(side), pos,popLSB(x), true);}
                }

                //bishops.
                U64 bishops = pieces[_nBishops+(int)(side)];
                while (bishops)
                {
                    pos = popLSB(bishops);
                    x = magicBishopAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nBishops+(int)(side), pos, popLSB(x), true);}
                }

                //knights.
                U64 knights = pieces[_nKnights+(int)(side)];
                while (knights)
                {
                    pos = popLSB(knights);
                    x = knightAttacks(1ull << pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nKnights+(int)(side), pos, popLSB(x), true);}
                }

                //rook.
                U64 rooks = pieces[_nRooks+(int)(side)];
                while (rooks)
                {
                    pos = popLSB(rooks);
                    x = magicRookAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nRooks+(int)(side), pos, popLSB(x), true);}
                }

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x) {appendMove(_nQueens+(int)(side), pos, popLSB(x), true);}
                }

                return true;
            }
            else
            {
                //multiple check. only king moves allowed.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x) {appendMove(_nKing+(int)(side), pos, popLSB(x), true);}

                return true;
            }
        }

        void movePieces()
        {
            //remove piece from start square;
            pieces[currentMove.pieceType] -= 1ull << (currentMove.startSquare);
            zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];
            if (currentMove.pieceType >> 1 == _nPawns >> 1)
            {
                zHashPawns ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];
            }

            //add piece to end square, accounting for promotion.
            pieces[currentMove.finishPieceType] += 1ull << (currentMove.finishSquare);
            zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];
            if (currentMove.finishPieceType >> 1 == _nPawns >> 1)
            {
                zHashPawns ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];
            }

            //remove any captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                pieces[currentMove.capturedPieceType] -= 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)));
                zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)))];
                if (currentMove.capturedPieceType >> 1 == _nPawns >> 1)
                {
                    zHashPawns ^= randomNums[64 * currentMove.capturedPieceType + (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)))];
                }

                //update the game phase.
                phase -= piecePhases[currentMove.capturedPieceType >> 1];
                shiftedPhase = (64 * phase + 3) / 6;
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
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1];
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];
                }
            }
        }

        void unMovePieces()
        {
            //remove piece from destination square.
            pieces[currentMove.finishPieceType] -= 1ull << (currentMove.finishSquare);
            zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];
            if (currentMove.finishPieceType >> 1 == _nPawns >> 1)
            {
                zHashPawns ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];
            }
            
            //add piece to start square.
            pieces[currentMove.pieceType] += 1ull << (currentMove.startSquare);
            zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];
            if (currentMove.pieceType >> 1 == _nPawns >> 1)
            {
                zHashPawns ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];
            }

            //add back captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                pieces[currentMove.capturedPieceType] += 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)));
                zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)))];
                if (currentMove.capturedPieceType >> 1 == _nPawns >> 1)
                {
                    zHashPawns ^= randomNums[64 * currentMove.capturedPieceType + (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)))];
                }

                //update the game phase.
                phase += piecePhases[currentMove.capturedPieceType >> 1];
                shiftedPhase = (64 * phase + 3) / 6;
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
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1];

                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                    zHashPieces ^= randomNums[64 * (_nRooks+(currentMove.pieceType & 1)) + QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];
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
                if ((currentMove.startSquare & 7) == 7)
                {
                    current.canKingCastle[currentMove.pieceType & 1] = false;
                }
                else if ((currentMove.startSquare & 7) == 0)
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
                if ((currentMove.finishSquare & 7) == 7)
                {
                    current.canKingCastle[currentMove.capturedPieceType & 1] = false;
                }
                else if ((currentMove.finishSquare & 7) == 0)
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
        }

        void makeNullMove()
        {
            stateHistory.push_back(current);
            moveHistory.push_back(0);

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
        }

        U64 getAttacksToSquare(bool side, U32 square)
        {
            U64 b = occupied[0] | occupied[1];

            U64 isAttacked = kingAttacks(1ull << square) & pieces[_nKing+(int)(!side)];
            isAttacked |= magicRookAttacks(b,square) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
            isAttacked |= magicBishopAttacks(b,square) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
            isAttacked |= knightAttacks(1ull << square) & pieces[_nKnights+(int)(!side)];
            isAttacked |= pawnAttacks(1ull << square,side) & pieces[_nPawns+(int)(!side)];

            return isAttacked;
        }

        U64 getLeastValuableAttacker(bool side, U64 isAttacked, U32 &attackingPiece)
        {
            for (int i=_nPawns+(int)(side); i >= (int)_nKing+(int)(side); i-=2)
            {
                U64 x = isAttacked & pieces[i];
                if (x)
                {
                    attackingPiece = i;
                    return x & (-x);
                }
            }
            return 0; // no attacker found.
        }

        int seeCaptures(U32 startSquare, U32 finishSquare, U32 pieceType, U32 capturedPieceType)
        {
            //perform static evaluation exchange (SEE).
            //use the currentMove struct.
            int d=0;
            gain[0] = PIECE_VALUES_START[capturedPieceType >> 1];
            U32 attackingPiece = pieceType;
            U64 attackingPieceBB = 1ull << startSquare;
            U64 isAttacked[2] = {getAttacksToSquare(1,finishSquare),
                                 getAttacksToSquare(0,finishSquare)};
            U64 occ = occupied[0] | occupied[1];

            bool side = pieceType & 1;

            do
            {
                d++;
                gain[d] = -gain[d-1] + PIECE_VALUES_START[attackingPiece >> 1];
                if (max(-gain[d-1],gain[d]) < 0) {break;}
                occ ^= attackingPieceBB;
                isAttacked[side] ^= attackingPieceBB;

                side = !side;

                //update possible x-ray attacks.
                isAttacked[side] |= magicRookAttacks(occ,finishSquare) & (pieces[_nRooks+(int)(side)] | pieces[_nQueens+(int)(side)]) & occ;
                isAttacked[side] |= magicBishopAttacks(occ,finishSquare) & (pieces[_nBishops+(int)(side)] | pieces[_nQueens+(int)(side)]) & occ;

                attackingPieceBB = getLeastValuableAttacker(side, isAttacked[side], attackingPiece);
            } while (attackingPieceBB);

            while (--d) {gain[d-1] = -max(-gain[d-1], gain[d]);}

            return gain[0];
        }

        int regularEval()
        {
            int startTotal = 0;
            int endTotal = 0;

            //queens.
            U64 temp = pieces[_nQueens];
            while (temp)
            {
                startTotal += PIECE_VALUES_START[1] + PIECE_TABLES_START[1][63-popLSB(temp)];
            }

            temp = pieces[_nQueens+1];
            while (temp)
            {
                startTotal -= PIECE_VALUES_START[1] + PIECE_TABLES_START[1][popLSB(temp)];
            }

            //rooks.
            temp = pieces[_nRooks];
            while (temp)
            {
                startTotal += PIECE_VALUES_START[2] + PIECE_TABLES_START[2][63-popLSB(temp)];
            }

            temp = pieces[_nRooks+1];
            while (temp)
            {
                startTotal -= PIECE_VALUES_START[2] + PIECE_TABLES_START[2][popLSB(temp)];
            }

            //bishops.
            temp = pieces[_nBishops];
            while (temp)
            {
                startTotal += PIECE_VALUES_START[3] + PIECE_TABLES_START[3][63-popLSB(temp)];
            }

            temp = pieces[_nBishops+1];
            while (temp)
            {
                startTotal -= PIECE_VALUES_START[3] + PIECE_TABLES_START[3][popLSB(temp)];
            }

            //knights.
            temp = pieces[_nKnights];
            while (temp)
            {
                startTotal += PIECE_VALUES_START[4] + PIECE_TABLES_START[4][63-popLSB(temp)];
            }

            temp = pieces[_nKnights+1];
            while (temp)
            {
                startTotal -= PIECE_VALUES_START[4] + PIECE_TABLES_START[4][popLSB(temp)];
            }

            //pawns.
            U64 index = zHashPawns & pawnHashMask;
            if (pawnHash[index].first == zHashPawns)
            {
                startTotal += pawnHash[index].second;
            }
            else
            {
                pawnHash[index].first = zHashPawns;
                pawnHash[index].second = 0;

                temp = pieces[_nPawns];
                while (temp)
                {
                    pawnHash[index].second += PIECE_VALUES_START[5] + PIECE_TABLES_START[5][63-popLSB(temp)];
                }

                temp = pieces[_nPawns+1];
                while (temp)
                {
                    pawnHash[index].second -= PIECE_VALUES_START[5] + PIECE_TABLES_START[5][popLSB(temp)];
                }

                startTotal += pawnHash[index].second;
            }

            endTotal = startTotal;

            //kings.
            int kingPos = __builtin_ctzll(pieces[_nKing]);
            int kingPos2 = __builtin_ctzll(pieces[_nKing+1]);

            startTotal += PIECE_TABLES_START[0][63-kingPos] - PIECE_TABLES_START[0][kingPos2];
            endTotal += PIECE_TABLES_END[0][63-kingPos] - PIECE_TABLES_END[0][kingPos2];

            return (((startTotal * shiftedPhase) + (endTotal * (256 - shiftedPhase))) / 256) * (1-2*(int)(moveHistory.size() & 1));
        }

        int evaluateBoard()
        {
            //see if checkmate or stalemate.
            bool turn = moveHistory.size() & 1;

            bool inCheck = generatePseudoMoves(turn);

            if (moveBuffer.size() > 0)
            {
                return regularEval();
            }
            else if (inCheck)
            {
                //checkmate.
                return -MATE_SCORE;
            }
            else
            {
                //stalemate.
                return 0;
            }
        }

        vector<pair<U32,int> > orderMoves(int depth, U32 bestMove = 0)
        {
            //assumes that getOccupied() has been called immediately before.
            scoredMoves.clear();

            for (int i=0;i<(int)(moveBuffer.size());i++)
            {
                if (moveBuffer[i] == bestMove)
                {
                    //best move from hash table should be checked first.
                    scoredMoves.push_back(pair<U32,int>(moveBuffer[i], INT_MAX));
                }
                else
                {
                    U32 capturedPieceType = (moveBuffer[i] & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                    if (capturedPieceType != 15)
                    {
                        //capture.
                        U32 startSquare = (moveBuffer[i] & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
                        U32 finishSquare = (moveBuffer[i] & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                        U32 pieceType = (moveBuffer[i] & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                        //scale captures for move ordering.
                        scoredMoves.push_back(pair<U32,int>(moveBuffer[i], 15 * seeCaptures(startSquare, finishSquare, pieceType, capturedPieceType)));
                    }
                    else
                    {
                        //non-capture moves.
                        if (moveBuffer[i] == killerMoves[depth][0] || moveBuffer[i] == killerMoves[depth][1])
                        {
                            //killer.
                            //set killer score to arbitrary 100 centipawns.
                            scoredMoves.push_back(pair<U32,int>(moveBuffer[i],100));
                        }
                        else
                        {
                            //non-killer.
                            U32 startSquare = (moveBuffer[i] & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
                            U32 finishSquare = (moveBuffer[i] & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                            U32 pieceType = (moveBuffer[i] & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                            if (pieceType & 1)
                            {
                                if (pieceType >> 1 == _nKing)
                                {
                                    scoredMoves.push_back(pair<U32,int>(moveBuffer[i],0));
                                }
                                else
                                {
                                    scoredMoves.push_back(pair<U32,int>(moveBuffer[i],
                                    PIECE_TABLES_START[pieceType >> 1][finishSquare] -
                                    PIECE_TABLES_START[pieceType >> 1][startSquare]));
                                }
                            }
                            else
                            {
                                if (pieceType >> 1 == _nKing)
                                {
                                    scoredMoves.push_back(pair<U32,int>(moveBuffer[i],0));
                                }
                                else
                                {
                                    scoredMoves.push_back(pair<U32,int>(moveBuffer[i],
                                    PIECE_TABLES_START[pieceType >> 1][63 - finishSquare] -
                                    PIECE_TABLES_START[pieceType >> 1][63 - startSquare]));
                                }
                            }
                        }
                    }
                }
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }

        vector<pair<U32,int> > orderQMoves(const int threshhold = 0)
        {
            //assumes that updateOccupied() has been called immediately before.
            scoredMoves.clear();
            
            for (int i=0;i<(int)moveBuffer.size();i++)
            {
                U32 capturedPieceType = (moveBuffer[i] & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                if (capturedPieceType != 15)
                {
                    //capture.
                    U32 startSquare = (moveBuffer[i] & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
                    U32 finishSquare = (moveBuffer[i] & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                    U32 pieceType = (moveBuffer[i] & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                    int score = seeCaptures(startSquare, finishSquare, pieceType, capturedPieceType);
                    if (score >= threshhold) {scoredMoves.push_back(pair<U32,int>(moveBuffer[i], score));}
                }
                else
                {
                    //non-capture moves.
                    scoredMoves.push_back(pair<U32,int>(moveBuffer[i],0));
                }
            }

            //sort the moves.
            sort(scoredMoves.begin(), scoredMoves.end(), [](auto &a, auto &b) {return a.second > b.second;});
            return scoredMoves;
        }
};

#endif // BOARD_H_INCLUDED
