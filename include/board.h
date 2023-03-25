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

using namespace std;

struct gameState
{
    bool turn;
    bool hasKingMoved[2];
    bool hasKingSideRookMoved[2];
    bool hasQueenSideRookMoved[2];
    int enPassantSquare;
    vector<U64> pieces{vector<U64>(12,0)};
};

struct moveInfo
{
    int pieceType;
    int startSquare;
    int finishSquare;
    int capturedPieceSquare;
    int capturedPieceType;
    int promotionPieceType;
};

class Board {
    public:
        vector<U64> attackTables{vector<U64>(12,0)};

        U64 occupied[2]={0,0};
        U64 attacked[2]={0,0};

        vector<gameState> stateHistory;
        vector<moveInfo> moveHistory;

        vector<moveInfo> moveBuffer;

        gameState current = {
            .turn=0,
            .hasKingMoved={false,false},
            .hasKingSideRookMoved={false,false},
            .hasQueenSideRookMoved={false,false},
        };

        const int _nKing=0;
        const int _nQueens=2;
        const int _nRooks=4;
        const int _nBishops=6;
        const int _nKnights=8;
        const int _nPawns=10;

        Board()
        {
            //default constructor for regular games.
            current.pieces[_nKing]=WHITE_KING;
            current.pieces[_nKing+1]=BLACK_KING;

            current.pieces[_nQueens]=WHITE_QUEENS;
            current.pieces[_nQueens+1]=BLACK_QUEENS;

            current.pieces[_nRooks]=WHITE_ROOKS;
            current.pieces[_nRooks+1]=BLACK_ROOKS;

            current.pieces[_nBishops]=WHITE_BISHOPS;
            current.pieces[_nBishops+1]=BLACK_BISHOPS;

            current.pieces[_nKnights]=WHITE_KNIGHTS;
            current.pieces[_nKnights+1]=BLACK_KNIGHTS;

            current.pieces[_nPawns]=WHITE_PAWNS;
            current.pieces[_nPawns+1]=BLACK_PAWNS;

            updateOccupied();
            updateAttackTables(0);
            updateAttackTables(1);
            updateAttacked();
        };

        void appendMove(int pieceType, int startSquare, int finishSquare)
        {
            //will automatically detect promotion and generate all permutations.
            //will automatically detect captured piece (including en passant).
            moveInfo newMove = {
                .pieceType=pieceType,
                .startSquare=startSquare,
                .finishSquare=finishSquare,
                .capturedPieceSquare=-1,
                .capturedPieceType=-1,
                .promotionPieceType=-1,
            };

            //check for captured pieces (including en passant).
            if ((convertToBitboard(finishSquare) & occupied[(pieceType+1)%2])!=0)
            {
                //regular capture.
                newMove.capturedPieceSquare = finishSquare;
            }
            else if (pieceType/2 == _nPawns/2 && finishSquare/8 == 5-3*(pieceType%2))
            {
                //en-passant capture.
                newMove.capturedPieceSquare = finishSquare-8+16*(pieceType%2);
            }

            if (newMove.capturedPieceSquare>0)
            {
                //find the piece type of the captured piece.
                U64 x = convertToBitboard(newMove.capturedPieceSquare);
                for (int i=_nQueens+(pieceType+1)%2;i<12;i+=2)
                {
                    if ((x & current.pieces[i]) != 0)
                    {
                        //found the captured piece.
                        newMove.capturedPieceType = i;
                        break;
                    }
                }
            }

            //check for pawn promotion.
            if (pieceType/2 == _nPawns/2 && finishSquare/8 == 7-7*(pieceType%2))
            {
                //promotion.
                for (int i=_nQueens+pieceType%2;i<_nPawns;i+=2)
                {
                    newMove.promotionPieceType=i;
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
            occupied[0]=0; occupied[1]=0;
            for (int i=0;i<(int)current.pieces.size();i++)
            {
                occupied[i%2] |= current.pieces[i];
            }
        }

        void updateAttacked()
        {
            attacked[0]=0; attacked[1]=0;
            for (int i=0;i<(int)attackTables.size();i++)
            {
                attacked[i%2] |= attackTables[i];
            }
        }

        bool isInCheck(bool side)
        {
            return bool(current.pieces[_nKing+(int)(side)] & attacked[(int)(!side)]);
        }

        pair<bool,bool> canCastle(bool side)
        {
            pair<bool,bool> res={true,true};

            //king and rook must not have previously moved.
            res.first=!current.hasKingSideRookMoved[(int)(side)] && !current.hasKingMoved[(int)(side)];
            res.second=!current.hasQueenSideRookMoved[(int)(side)] && !current.hasKingMoved[(int)(side)];

            //no pieces between king and rook.
            res.first &= !(bool)(KING_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1]));
            res.second &= !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & (occupied[0] | occupied[1]));

            //king squares cannot be under attack.
            res.first &= !(bool)(KING_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]);
            res.second &= !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked[(int)(!side)]);

            return res;
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
            for (int i=0;i<(int)current.pieces.size();i++)
            {
                temp=bitset<64>(current.pieces[i]).to_string();
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

        void updateAttackTables(bool side)
        {
            //generate tables of attacked squares.
            U64 p = ~(occupied[0] | occupied[1]);

            attackTables[_nKing+side]=kingAttacks(current.pieces[_nKing+(int)(side)]);
            attackTables[_nQueens+side]=queenAttacks(current.pieces[_nQueens+(int)(side)],p);
            attackTables[_nRooks+side]=rookAttacks(current.pieces[_nRooks+(int)(side)],p);
            attackTables[_nBishops+side]=bishopAttacks(current.pieces[_nBishops+(int)(side)],p);
            attackTables[_nKnights+side]=knightAttacks(current.pieces[_nKnights+(int)(side)]);
            attackTables[_nPawns+side]=pawnAttacks(current.pieces[_nPawns+(int)(side)],(int)(side));
        }

        void generatePseudoMoves(bool side)
        {
            //generate all pseudo-legal moves.
            //assume occupancy and attack bitboards are up-to-date.

            //king.

            //get position of king.
            int kingPos = getLSB(current.pieces[_nKing+(int)(side)]);

            //get position of move-squares.
            //the king cannot go to a square which is attacked by the enemy.
            U64 x = attackTables[_nKing+(int)(side)] & ~occupied[(int)(side)];
            x &= ~attacked[(int)(!side)];

            while (x!=0)
            {
                int finishSquare = popLSB(x);
                appendMove(_nKing+(int)(side), kingPos, finishSquare);
            }

            //castling.
            pair<bool,bool> rights = canCastle(side);
            if (rights.first==true)
            {
                //kingside castle.
                appendMove(_nKing+(int)(side), kingPos, kingPos+2);
            }
            if (rights.second==true)
            {
                //queenside castle.
                appendMove(_nKing+(int)(side), kingPos, kingPos-2);
            }

            U64 p = ~(occupied[0] | occupied[1]);
            //queen.
            U64 queens = current.pieces[_nQueens+(int)(side)];

            while (queens!=0)
            {
                int queenPos = popLSB(queens);

                U64 x = queenAttacks(convertToBitboard(queenPos),p);
                x &= ~occupied[(int)(side)];

                while (x!=0)
                {
                    int finishSquare = popLSB(x);
                    appendMove(_nQueens+(int)(side), queenPos, finishSquare);
                }
            }

            //rook.
            U64 rooks = current.pieces[_nRooks+(int)(side)];

            while (rooks!=0)
            {
                int rookPos = popLSB(rooks);

                U64 x = rookAttacks(convertToBitboard(rookPos),p);
                x &= ~occupied[(int)(side)];

                while (x!=0)
                {
                    int finishSquare = popLSB(x);
                    appendMove(_nRooks+(int)(side), rookPos, finishSquare);
                }
            }

            //bishops.
            U64 bishops = current.pieces[_nBishops+(int)(side)];

            while (bishops!=0)
            {
                int bishopPos = popLSB(bishops);

                U64 x = bishopAttacks(convertToBitboard(bishopPos),p);
                x &= ~occupied[(int)(side)];

                while (x!=0)
                {
                    int finishSquare = popLSB(x);
                    appendMove(_nBishops+(int)(side), bishopPos, finishSquare);
                }
            }

            //knights.
            U64 knights = current.pieces[_nKnights+(int)(side)];

            while (knights!=0)
            {
                int knightPos = popLSB(knights);

                U64 x = knightAttacks(convertToBitboard(knightPos));
                x &= ~occupied[(int)(side)];

                while (x!=0)
                {
                    int finishSquare = popLSB(x);
                    appendMove(_nKnights+(int)(side), knightPos, finishSquare);
                }
            }

            //pawns.
            U64 pawns = current.pieces[_nPawns+(int)(side)];

            while (pawns!=0)
            {
                int pawnPos = popLSB(pawns);
                U64 pawnPosBoard = convertToBitboard(pawnPos);

                //en passant square included.
                U64 x = pawnAttacks(pawnPosBoard,side);
                x &= ~occupied[(int)(side)];

                if (current.enPassantSquare != -1)
                {
                    x &= (occupied[(int)(!side)] | convertToBitboard(current.enPassantSquare));
                }
                else
                {
                    x &= occupied[(int)(!side)];
                }

                //move forward.
                if (side==0)
                {
                    x |= (pawnPosBoard << 8) & p;
                    x |= ((((pawnPosBoard & FILE_2) << 8) & p) << 8) & p;
                }
                else
                {
                    x |= (pawnPosBoard >> 8 & p);
                    x |= ((((pawnPosBoard & FILE_7) >> 8) & p) >> 8) & p;
                }

                while (x!=0)
                {
                    int finishSquare = popLSB(x);
                    appendMove(_nPawns+(int)(side), pawnPos,finishSquare);
                }
            }
        }

        void testMoves()
        {
            //test out moves and remove the ones which leave king hanging.
            vector<moveInfo> legalBuffer;

            for (int i=0;i<(int)moveBuffer.size();i++)
            {
                gameState result = movePieces(current, moveBuffer[i]);

                //check if result in check or not.
                bool res = inCheck(result, current.turn);
                if (res==false)
                {
                    //legal move.
                    legalBuffer.push_back(moveBuffer[i]);
                }
            }

            //only include the legal moves.
            moveBuffer = legalBuffer;
        }

        bool inCheck(gameState currentState, bool side)
        {
            //checks if the chosen side is in check.

            U64 p = 0;
            for (int i=0;i<(int)currentState.pieces.size();i++)
            {
                p |= currentState.pieces[i];
            }
            p = ~p;

            U64 totalAttacked=0;

            totalAttacked |=kingAttacks(currentState.pieces[_nKing+(int)(!side)]);
            totalAttacked |=queenAttacks(currentState.pieces[_nQueens+(int)(!side)],p);
            totalAttacked |=rookAttacks(currentState.pieces[_nRooks+(int)(!side)],p);
            totalAttacked |=bishopAttacks(currentState.pieces[_nBishops+(int)(!side)],p);
            totalAttacked |=knightAttacks(currentState.pieces[_nKnights+(int)(!side)]);
            totalAttacked |=pawnAttacks(currentState.pieces[_nPawns+(int)(!side)],(int)(!side));

            return (currentState.pieces[_nKing+(int)(side)] & totalAttacked) != 0;
        }

        gameState movePieces(gameState currentState, moveInfo currentMove)
        {
            gameState test = currentState;

            //remove piece from start square;
            test.pieces[currentMove.pieceType] -= convertToBitboard(currentMove.startSquare);

            //add piece to end square, accounting for promotion.
            int finishPieceType = currentMove.promotionPieceType>=_nQueens ? currentMove.promotionPieceType : currentMove.pieceType;
            test.pieces[finishPieceType] += convertToBitboard(currentMove.finishSquare);

            //remove any captured pieces.
            if (currentMove.capturedPieceType!=-1)
            {
                test.pieces[currentMove.capturedPieceType] -= convertToBitboard(currentMove.capturedPieceSquare);
            }

            //if castles, then move the rook too.
            if (currentMove.pieceType/2 == _nKing/2 && abs(currentMove.finishSquare-currentMove.startSquare)==2)
            {
                if (currentMove.finishSquare-currentMove.startSquare==2)
                {
                    //kingside.
                    test.pieces[_nRooks+currentMove.pieceType%2] -= KING_ROOK_POS[currentMove.pieceType%2];
                    test.pieces[_nRooks+currentMove.pieceType%2] += KING_ROOK_POS[currentMove.pieceType%2] >> 2;
                }
                else
                {
                    //queenside.
                    test.pieces[_nRooks+currentMove.pieceType%2] -= QUEEN_ROOK_POS[currentMove.pieceType%2];
                    test.pieces[_nRooks+currentMove.pieceType%2] += QUEEN_ROOK_POS[currentMove.pieceType%2] << 3;
                }
            }

            return test;
        }

        void makeMove(moveInfo currentMove)
        {
            //update history.
            stateHistory.push_back(current);
            moveHistory.push_back(currentMove);

            //move pieces.
            current = movePieces(current, currentMove);

            //update game-state.

            //if double-pawn push, set en-passant square.
            //otherwise, set en-passant square to -1.
            if (currentMove.pieceType/2 == _nPawns/2 && abs(currentMove.finishSquare-currentMove.startSquare) == 16)
            {
                //en-passant is possible immediately.
                current.enPassantSquare = currentMove.finishSquare-8+16*(currentMove.pieceType%2);
            }
            else
            {
                current.enPassantSquare = -1;
            }

            current.turn = !current.turn;
            current.hasKingMoved[currentMove.pieceType%2] |= currentMove.pieceType/2 == _nKing/2;
            current.hasKingSideRookMoved[currentMove.pieceType%2] |= (currentMove.pieceType/2 == _nRooks/2) && (currentMove.startSquare == 7 + 56 * (currentMove.pieceType%2));
            current.hasQueenSideRookMoved[currentMove.pieceType%2] |= (currentMove.pieceType/2 == _nRooks/2) && (currentMove.startSquare == 0 + 56 * (currentMove.pieceType%2));
        }

        void unmakeMove()
        {
            //unmake most recent move and update gameState.

            current = stateHistory.back();
            stateHistory.pop_back();
            moveHistory.pop_back();
        }
};

#endif // BOARD_H_INCLUDED
