#ifndef FORMAT_H_INCLUDED
#define FORMAT_H_INCLUDED

#include "constants.h"
#include "bitboard.h"
#include "board.h"

const std::string promotionLetters = "_qrbn";
const std::string fileSymbols = "abcdefgh";
const std::string rankSymbols = "12345678";

std::vector<std::string> separateByWhiteSpace(const std::string &input)
{
    //assume only a single whitespace separates each word.
    std::vector<std::string> words; words.push_back("");
    for (const auto &character: input)
    {
        if (character == ' ') {words.push_back("");}
        else {words.back() += character;}
    }
    while (words.back() == "") {words.pop_back();}
    return words;
}

std::string moveToString(U32 chessMove)
{
    std::string startSquare = toCoord((chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET);
    std::string finishSquare = toCoord((chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET);

    std::string res = startSquare + finishSquare;

    int pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
    int finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

    //check for promotion.
    if (pieceType != finishPieceType)
    {
        res += promotionLetters[finishPieceType >> 1];
    }

    return res;
}

U32 stringToMove(Board &b, const std::string &input)
{
    //return 0 (null move) if input is not correct.
    if (input.length() != 4 && input.length() != 5) {return 0;}

    size_t startFile = fileSymbols.find(input[0]);
    size_t startRank = rankSymbols.find(input[1]);

    size_t finishFile = fileSymbols.find(input[2]);
    size_t finishRank = rankSymbols.find(input[3]);

    if (startFile == std::string::npos || startRank == std::string::npos ||
        finishFile == std::string::npos || finishRank == std::string::npos)
    {
        return 0;
    }

    U32 startSquare = 8 * startRank + startFile;
    U32 finishSquare = 8 * finishRank + finishFile;

    size_t finishPiece = 0;

    if (input.length() == 5)
    {
        finishPiece = promotionLetters.find(input[4]) * 2 + (b.moveHistory.size() & 1);
        if (finishPiece == std::string::npos) {return 0;}
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
        return b.moveBuffer[i];
    }

    return 0;
}

#endif // FORMAT_H_INCLUDED
