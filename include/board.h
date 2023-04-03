#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <vector>
#include <bitset>
#include <windows.h>
#include <cctype>

#include "constants.h"
#include "bitboard.h"

#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "slider.h"
#include "magic.h"

#include "evaluation.h"

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
    bool shouldCheck;
};

class Board {
    public:
        U64 pieces[12]={};

        U64 occupied[2]={0,0};
        U64 attacked[2]={0,0};

        vector<gameState> stateHistory;
        vector<U32> moveHistory;

        vector<U32> moveBuffer;

        gameState current = {
            .canKingCastle = {true,true},
            .canQueenCastle = {true,true},
            .enPassantSquare = -1,
        };

        moveInfo currentMove = {};

        const U32 _nKing=0;
        const U32 _nQueens=2;
        const U32 _nRooks=4;
        const U32 _nBishops=6;
        const U32 _nKnights=8;
        const U32 _nPawns=10;

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
        };

        void appendMove(U32 pieceType, U32 startSquare, U32 finishSquare, bool shouldCheck=false)
        {
            //will automatically detect promotion and generate all permutations.
            //will automatically detect captured piece (including en passant).
            U32 newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) |
            (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
            (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) |
            ((U32)(shouldCheck) << MOVEINFO_SHOULDCHECK_OFFSET) |
            MOVEINFO_CAPTUREDPIECETYPE_MASK |
            (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

            //check for captured pieces (including en passant).
            if (((1ull << finishSquare) & occupied[(pieceType+1) & 1])!=0)
            {
                //check for captures.
                U64 x = 1ull << finishSquare;
                for (int i=_nQueens+((pieceType+1) & 1);i<12;i+=2)
                {
                    if ((x & pieces[i]) != 0)
                    {
                        newMove &= ~MOVEINFO_CAPTUREDPIECETYPE_MASK;
                        newMove |= i << MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                        break;
                    }
                }
            }
            else if ((pieceType >> 1 == _nPawns >> 1 && finishSquare >> 3 == 5u-3u*(pieceType & 1u)) && (finishSquare%8 != startSquare%8))
            {
                //en-passant capture.
                newMove |= 1 << MOVEINFO_ENPASSANT_OFFSET;

                //check for captured piece.
                U64 x = 1ull << (finishSquare-8+16*(pieceType & 1));
                for (int i=_nQueens+((pieceType+1) & 1);i<12;i+=2)
                {
                    if ((x & pieces[i]) != 0)
                    {
                        newMove &= ~MOVEINFO_CAPTUREDPIECETYPE_MASK;
                        newMove |= i << MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
                        break;
                    }
                }
            }

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
            attacked[(int)(side)] |= pawnAttacks(pieces[_nPawns+(int)(side)],(int)(side));
        }

        bool isInCheck(bool side)
        {
            //check if the king's square is attacked.
            int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
            U64 b = occupied[0] | occupied[1];

            bool inCheck = kingAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKing+(int)(!side)];
            inCheck |= magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
            inCheck |= magicBishopAttacks(b,kingPos) & (pieces[_nBishops+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
            inCheck |= knightAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKnights+(int)(!side)];
            inCheck |= pawnAttacks(pieces[_nKing+(int)(side)],side) & pieces[_nPawns+(int)(!side)];

            return inCheck;
        }

        U32 isInCheckDetailed(bool side)
        {
            //check if the king's square is attacked.
            int kingPos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
            U64 b = occupied[0] | occupied[1];

            U64 inCheck = kingAttacks(pieces[_nKing+(int)(side)]) & pieces[_nKing+(int)(!side)];
            inCheck |= magicRookAttacks(b,kingPos) & (pieces[_nRooks+(int)(!side)] | pieces[_nQueens+(int)(!side)]);
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

        void generatePseudoMoves(bool side)
        {
            //generate all pseudo-legal moves.
            updateOccupied();
            updateAttacked(!side);

            bool kingInCheck = (bool)(pieces[_nKing+(int)(side)] & attacked[(int)(!side)]);

            if (!kingInCheck)
            {
                //regular moves.

                //king.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x)
                {
                    appendMove(_nKing+(int)(side), pos, popLSB(x));
                }

                //castling.
                if (current.canKingCastle[(int)(side)] &&
                    !(bool)(KING_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1])) &&
                    !(bool)(KING_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]) &&
                    (bool)(pieces[_nRooks+(int)(side)] & KING_ROOK_POS[(int)(side)]))
                {
                    //kingside castle.
                    appendMove(_nKing+(int)(side), pos, pos+2);
                }
                if (current.canQueenCastle[(int)(side)] &&
                    !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1])) &&
                    !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]) &&
                    (bool)(pieces[_nRooks+(int)(side)] & QUEEN_ROOK_POS[(int)(side)]))
                {
                    //queenside castle.
                    appendMove(_nKing+(int)(side), pos, pos-2);
                }

                U64 pinned = getPinnedPieces(side);
                U64 p = (occupied[0] | occupied[1]);

                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nQueens+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);
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
                        appendMove(_nKnights+(int)(side), pos, popLSB(x), ((1ull << pos) & pinned)!=0);
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

                    if (current.enPassantSquare != -1)
                    {
                        x &= (occupied[(int)(!side)] | (1ull << current.enPassantSquare));
                    }
                    else
                    {
                        x &= occupied[(int)(!side)];
                    }

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
                    }
                }
            }
            else if (isInCheckDetailed(side) == 1)
            {
                //single check.

                //king.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x)
                {
                    appendMove(_nKing+(int)(side), pos, popLSB(x),true);
                }

                U64 p = (occupied[0] | occupied[1]);
                //queen.
                U64 queens = pieces[_nQueens+(int)(side)];
                while (queens)
                {
                    pos = popLSB(queens);
                    x = magicQueenAttacks(p,pos) & ~occupied[(int)(side)];
                    while (x)
                    {
                        appendMove(_nQueens+(int)(side), pos, popLSB(x), true);
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

                    if (current.enPassantSquare != -1)
                    {
                        x &= (occupied[(int)(!side)] | (1ull << current.enPassantSquare));
                    }
                    else
                    {
                        x &= occupied[(int)(!side)];
                    }

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
                    }
                }
            }
            else
            {
                //multiple check. only king moves allowed.
                U32 pos = __builtin_ctzll(pieces[_nKing+(int)(side)]);
                U64 x = kingAttacks(pieces[_nKing+(int)(side)]) & ~attacked[(int)(!side)] & ~occupied[(int)(side)];
                while (x)
                {
                    appendMove(_nKing+(int)(side), pos, popLSB(x));
                }
            }
        }

//        void movePieces(moveInfo currentMove)
        void movePieces()
        {
            //remove piece from start square;
            pieces[currentMove.pieceType] -= 1ull << (currentMove.startSquare);

            //add piece to end square, accounting for promotion.
            pieces[currentMove.finishPieceType] += 1ull << (currentMove.finishSquare);

            //remove any captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                pieces[currentMove.capturedPieceType] -= 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)));
            }

            //if castles, then move the rook too.
            if (currentMove.pieceType >> 1 == _nKing >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare))==2)
            {
                if (currentMove.finishSquare-currentMove.startSquare==2)
                {
                    //kingside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= KING_ROOK_POS[currentMove.pieceType & 1];
                    pieces[_nRooks+(currentMove.pieceType & 1)] += KING_ROOK_POS[currentMove.pieceType & 1] >> 2;
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1];
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;
                }
            }
        }

        void unMovePieces()
        {
            //remove piece from destination square.
            pieces[currentMove.finishPieceType] -= 1ull << (currentMove.finishSquare);

            //add piece to start square.
            pieces[currentMove.pieceType] += 1ull << (currentMove.startSquare);

            //add back captured pieces.
            if (currentMove.capturedPieceType != 15)
            {
                pieces[currentMove.capturedPieceType] += 1ull << (currentMove.finishSquare+(int)(currentMove.enPassant)*(-8+16*(currentMove.pieceType & 1)));
            }

            //if castles move the rook back.
            if (currentMove.pieceType >> 1 == _nKing >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare))==2)
            {
                if (currentMove.finishSquare-currentMove.startSquare==2)
                {
                    //kingside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= KING_ROOK_POS[currentMove.pieceType & 1] >> 2;
                    pieces[_nRooks+(currentMove.pieceType & 1)] += KING_ROOK_POS[currentMove.pieceType & 1];
                }
                else
                {
                    //queenside.
                    pieces[_nRooks+(currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;
                    pieces[_nRooks+(currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1];
                }
            }
        }

        void unpackMove(U32 chessMove)
        {
            currentMove.pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            currentMove.startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
            currentMove.finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
            currentMove.enPassant = (chessMove & MOVEINFO_ENPASSANT_MASK);
            currentMove.capturedPieceType = (chessMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
            currentMove.finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
            currentMove.shouldCheck = (chessMove & MOVEINFO_SHOULDCHECK_MASK);
        }

        bool makeMove(U32 chessMove)
        {
            unpackMove(chessMove);

            //move pieces.
            movePieces();

            //check if the move was legal.
            if (!currentMove.shouldCheck)
            {
                //update history.
                stateHistory.push_back(current);
                moveHistory.push_back(chessMove);

                //if double-pawn push, set en-passant square.
                //otherwise, set en-passant square to -1.
                bool x = currentMove.pieceType >> 1 == _nPawns >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare)) == 16;
                current.enPassantSquare = -1 + (int)(x)*(1+currentMove.finishSquare-8+16*(currentMove.pieceType & 1));

                if (currentMove.pieceType >> 1 == _nRooks >> 1)
                {
                    if (currentMove.startSquare == (7 + 56 * (currentMove.pieceType & 1)))
                    {
                        current.canKingCastle[currentMove.pieceType & 1] = false;
                    }
                    else if (currentMove.startSquare == (56 * (currentMove.pieceType & 1)))
                    {
                        current.canQueenCastle[currentMove.pieceType & 1] = false;
                    }
                }
                else if (currentMove.pieceType >> 1 == _nKing >> 1)
                {
                    current.canKingCastle[currentMove.pieceType & 1] = false;
                    current.canQueenCastle[currentMove.pieceType & 1] = false;
                }

                return true;
            }
            else
            {
                updateOccupied();
                if (isInCheck(currentMove.pieceType & 1))
                {
                    unMovePieces();
                    return false;
                }
                else
                {
                    //update history.
                    stateHistory.push_back(current);
                    moveHistory.push_back(chessMove);

                    //if double-pawn push, set en-passant square.
                    //otherwise, set en-passant square to -1.
                    bool x = currentMove.pieceType >> 1 == _nPawns >> 1 && abs((int)(currentMove.finishSquare)-(int)(currentMove.startSquare)) == 16;
                    current.enPassantSquare = -1 + (int)(x)*(1+currentMove.finishSquare-8+16*(currentMove.pieceType & 1));

                    if (currentMove.pieceType >> 1 == _nRooks >> 1)
                    {
                        if (currentMove.startSquare == (7 + 56 * (currentMove.pieceType & 1)))
                        {
                            current.canKingCastle[currentMove.pieceType & 1] = false;
                        }
                        else if (currentMove.startSquare == (56 * (currentMove.pieceType & 1)))
                        {
                            current.canQueenCastle[currentMove.pieceType & 1] = false;
                        }
                    }
                    else if (currentMove.pieceType >> 1 == _nKing >> 1)
                    {
                        current.canKingCastle[currentMove.pieceType & 1] = false;
                        current.canQueenCastle[currentMove.pieceType & 1] = false;
                    }

                    return true;
                }
            }
        }

        void unmakeMove()
        {
            //unmake most recent move and update gameState.
            current = stateHistory.back();
            unpackMove(moveHistory.back());
            unMovePieces();

            stateHistory.pop_back();
            moveHistory.pop_back();
        }

        int evalPieces(int pieceType)
        {
            int total=0;
            U64 temp = pieces[pieceType] ^ (63ull * ((U64)pieceType & 1ull));
            while (temp)
            {
                //only mid-game eval at the moment.
                total += PIECE_VALUES_START[pieceType >> 1] + PIECE_TABLES_START[pieceType >> 1][popLSB(temp)];
            }
            return total;
        }
};

#endif // BOARD_H_INCLUDED
