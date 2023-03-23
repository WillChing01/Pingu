#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include <vector>
#include <string>
#include <bitset>
#include <windows.h>
#include <cctype>

#include "constants.h"
#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "slider.h"

using namespace std;

class Board {
    public:
        vector<U64> pieces{vector<U64>(12,0)};
        vector<U64> attackTables{vector<U64>(12,0)};
        vector<U64> moveTables{vector<U64>(12,0)};

        U64 occupied[2]={0,0};

        bool turn=1; //white starts.
        bool inCheck=false;
        bool canKingSideCastle[2]={true,true};
        bool canQueenSideCastle[2]={true,true};

        int kingPos[2]={4,60};

        const int _nKing=0;
        const int _nQueens=2;
        const int _nRooks=4;
        const int _nBishops=6;
        const int _nKnights=8;
        const int _nPawns=10;

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
            updateAttackTables(0);
            updateAttackTables(1);
        };

        void updateOccupied()
        {
            occupied[0]=0; occupied[1]=0;
            for (int i=0;i<(int)pieces.size();i++)
            {
                occupied[i%2]|=pieces[i];
            }
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
            for (int i=0;i<(int)pieces.size();i++)
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

        void updateAttackTables(bool side)
        {
            //generate tables of attacked squares.
            U64 p = ~(occupied[0] | occupied[1]);

            attackTables[_nKing+side]=kingAttacks(pieces[_nKing+side]);
            attackTables[_nQueens+side]=queenAttacks(pieces[_nQueens+side],p);
            attackTables[_nRooks+side]=rookAttacks(pieces[_nRooks+side],p);
            attackTables[_nBishops+side]=bishopAttacks(pieces[_nBishops+side],p);
            attackTables[_nKnights+side]=knightAttacks(pieces[_nKnights+side]);
            attackTables[_nPawns+side]=pawnAttacks(pieces[_nPawns+side]);
        }

        void generateMoves(bool side)
        {
            //generate all legal moves.
            //assume occupancy and attack bitboards are up-to-date.
            if (inCheck==true)
            {
                //evade check.
            }
            else
            {
                //regular moves.

                //XOR with friendly-occupied squares.
                //for pawns need to generate forward moves, and can only take if attacking enemy.

                //king.

                //the king cannot go to a square which is attacked by the enemy.
                U64 attackedSquares=attackTables[_nKing+(int)(!side)];
                attackedSquares |= attackTables[_nQueens+(int)(!side)];
                attackedSquares |= attackTables[_nRooks+(int)(!side)];
                attackedSquares |= attackTables[_nBishops+(int)(!side)];
                attackedSquares |= attackTables[_nKnights+(int)(!side)];
                attackedSquares |= attackTables[_nPawns+(int)(!side)];

                moveTables[_nKing+side]=attackTables[_nKing+side] & ~occupied[(int)(side)];
                moveTables[_nKing+side] &= ~attackedSquares;

                //castling.

                //queen.
                moveTables[_nQueens+side]=attackTables[_nQueens+side] & ~occupied[(int)(side)];

                //rook.
                moveTables[_nRooks+side]=attackTables[_nRooks+side] & ~occupied[(int)(side)];

                //bishops.
                moveTables[_nBishops+side]=attackTables[_nBishops+side] & ~occupied[(int)(side)];

                //knights.
                moveTables[_nKnights+side]=attackTables[_nKnights+side] & ~occupied[(int)(side)];

                //pawns.
                moveTables[_nPawns+side]=attackTables[_nPawns+side] & (~occupied[(int)(side)] & occupied[(int)(!side)]);

                //moving forward.
                if (side==0)
                {
                    //white.
                    moveTables[_nPawns] |= (pieces[_nPawns] << 8) & ~(occupied[0] | occupied[1]);

                    //initial 'jump'.
                    moveTables[_nPawns] |= ((pieces[_nPawns] & FILE_2) << 16) & ~(occupied[0] | occupied[1]);
                }
                else
                {
                    //black.
                    moveTables[_nPawns+1] |= (pieces[_nPawns+1] >> 8) & ~(occupied[0] | occupied[1]);

                    //initial 'jump'.
                    moveTables[_nPawns+1] |= ((pieces[_nPawns+1] & FILE_7) >> 16) & ~(occupied[0] | occupied[1]);
                }

                //en passant.
                //promotion? maybe deal with it during actual move execution.
            }
        }
};

void displayBitboard(U64 bitboard)
{
    string temp=bitset<64>(bitboard).to_string();

    cout << endl;
    for (int i=7;i>=0;i--)
    {
        for (int j=0;j<8;j++)
        {
            cout << " " << temp[63-(8*i+j)];
        } cout << endl;
    } cout << endl;
}


#endif // BOARD_H_INCLUDED
