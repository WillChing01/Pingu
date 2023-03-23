#include <iostream>

#include "board.h"

using namespace std;

int main()
{
    Board board;

    board.display();

    displayBitboard(board.attackTables[board._nKing]);
    displayBitboard(board.attackTables[board._nQueens]);
    displayBitboard(board.attackTables[board._nRooks]);
    displayBitboard(board.attackTables[board._nBishops]);
    displayBitboard(board.attackTables[board._nKnights]);
    displayBitboard(board.attackTables[board._nPawns]);

    board.generateMoves(0);

    displayBitboard(board.moveTables[board._nKing]);
    displayBitboard(board.moveTables[board._nQueens]);
    displayBitboard(board.moveTables[board._nRooks]);
    displayBitboard(board.moveTables[board._nBishops]);
    displayBitboard(board.moveTables[board._nKnights]);
    displayBitboard(board.moveTables[board._nPawns]);

    return 0;
}
