#ifndef FORMAT_H_INCLUDED
#define FORMAT_H_INCLUDED

#include "constants.h"
#include "bitboard.h"
#include "board.h"

const string promotionLetters = "_qrbn";
const string fileSymbols = "abcdefgh";
const string rankSymbols = "12345678";

vector<string> separateByWhiteSpace(string input)
{
    //assume only a single whitespace separates each word.
    vector<string> words; words.push_back("");
    for (int i=0;i<(int)input.length();i++)
    {
        if (input[i] == ' ') {words.push_back("");}
        else {words.back() += input[i];}
    }
    while (words.back() == "") {words.pop_back();}
    return words;
}

string moveToString(U32 chessMove)
{
    string startSquare = toCoord((chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET);
    string finishSquare = toCoord((chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET);

    string res = startSquare + finishSquare;

    int pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
    int finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

    //check for promotion.
    if (pieceType != finishPieceType)
    {
        res += promotionLetters[finishPieceType >> 1];
    }

    return res;
}

U32 stringToMove(Board &b, string input)
{
    //return 0 (null move) if input is not correct.
    if (input.length() != 4 && input.length() != 5) {return 0;}

    size_t startFile = fileSymbols.find(input[0]);
    size_t startRank = rankSymbols.find(input[1]);

    size_t finishFile = fileSymbols.find(input[2]);
    size_t finishRank = rankSymbols.find(input[3]);

    if (startFile == string::npos || startRank == string::npos ||
        finishFile == string::npos || finishRank == string::npos)
    {
        return 0;
    }

    U32 startSquare = 8 * startRank + startFile;
    U32 finishSquare = 8 * finishRank + finishFile;

    size_t finishPiece = 0;

    if (input.length() == 5)
    {
        finishPiece = promotionLetters.find(input[4]) * 2 + (b.moveHistory.size() & 1);
        if (finishPiece == string::npos) {return 0;}
    }

    //check if move is legal.
    b.generatePseudoMoves(b.moveHistory.size() & 1);

    for (int i=0;i<(int)b.moveBuffer.size();i++)
    {
        b.unpackMove(b.moveBuffer[i]);
        if (b.currentMove.startSquare != startSquare ||
            b.currentMove.finishSquare != finishSquare)
        {
            continue;
        }
        if (b.currentMove.pieceType != b.currentMove.finishPieceType &&
            b.currentMove.finishPieceType != (U32)finishPiece)
        {
            continue;
        }
//        if (b.makeMove(b.moveBuffer[i]))
//        {
//            //move is legal and correct.
//            b.unmakeMove();
//            return b.moveBuffer[i];
//        }
    }

    return 0;
}

#endif // FORMAT_H_INCLUDED
